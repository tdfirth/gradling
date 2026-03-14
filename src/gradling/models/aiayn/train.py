from __future__ import annotations

import re
from itertools import islice
from time import perf_counter

import jax
import numpy as np
import optax
from flax import nnx
from jax import numpy as jnp

import wandb
from gradling import logger
from gradling.data import create_dataset, loader, random_iterator
from gradling.models.aiayn.config import AIAYNConfig
from gradling.models.aiayn.model import AIAYN
from gradling.run import Run

log = logger.get(__name__)


def _loss_fn(model: AIAYN, xs: jax.Array, ys: jax.Array):
    logits = model(xs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, ys).mean()
    return loss, logits


def duration_in_ms(start: int | float, stop: int | float) -> int | float:
    start_ms = start * 1000
    stop_ms = stop * 1000
    return stop_ms - start_ms


EVALUATE_ON_STEP = 200


def format_path(path):
    return "".join(format(p) for p in path)


def path_matches(path, regex):
    result = re.search(regex, format_path(path))
    return result is not None


def _run_training_loop(
    run: Run,
    cfg: AIAYNConfig,
    model: AIAYN,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    train_data: np.memmap,
    dev_data: np.memmap,
) -> None:

    @nnx.jit
    def _train_step(
        model: AIAYN,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        rngs: nnx.Rngs,
        xs: jax.Array,
        ys: jax.Array,
    ):
        grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model, xs, ys)
        metrics.update(loss=loss, logits=logits, labels=ys.astype(jnp.int32))
        optimizer.update(model, grads)

    @nnx.jit
    def _eval_step(
        model: AIAYN,
        metrics: nnx.MultiMetric,
        rngs: nnx.Rngs,
        xs: jax.Array,
        ys: jax.Array,
    ):
        loss, logits = _loss_fn(model, xs, ys)
        metrics.update(loss=loss, logits=logits, labels=ys.astype(jnp.int32))

    # Set up the data iterators
    nrng = np.random.Generator(np.random.PCG64(seed=cfg.seed))
    train_iterator = islice(
        random_iterator(nrng, cfg.batch_size, cfg.n_ctx, train_data),
        cfg.train_steps,
    )
    dev_iterator = loader(
        islice(
            random_iterator(nrng, cfg.batch_size, cfg.n_ctx, dev_data),
            # Plus one because we eval once on step 0 too!
            (cfg.train_steps // EVALUATE_ON_STEP) + 1,
        )
    )

    model.train()
    window_start = perf_counter()

    for step, batch in enumerate(loader(train_iterator)):
        xs, ys = batch
        should_evaluate = step % EVALUATE_ON_STEP == 0
        if not should_evaluate:
            _train_step(model, optimizer, metrics, rngs, xs, ys)
        else:
            log.info(f"Step {step}/{cfg.train_steps}")

            # Measure any time taken waiting for data to be ready.
            data_start = perf_counter()
            jax.effects_barrier()
            data_end = perf_counter()

            train_start = perf_counter()
            _train_step(model, optimizer, metrics, rngs, xs, ys)
            jax.effects_barrier()
            train_end = perf_counter()

            train_metrics = {f"train_{k}": v for k, v in metrics.compute().items()}
            metrics.reset()

            model.eval()
            dev_xs, dev_ys = next(dev_iterator)
            _eval_step(model, metrics, rngs, dev_xs, dev_ys)
            dev_metrics = {f"dev_{k}": v for k, v in metrics.compute().items()}
            metrics.reset()

            window_end = perf_counter()
            window_duration_ms = duration_in_ms(window_start, window_end)
            window_duration_sec = window_duration_ms / 1000
            samples_processed = cfg.batch_size * EVALUATE_ON_STEP

            # Compute frobenius norm for each attention block
            attn_weights = [
                [format_path(path), value]
                for path, value in jax.tree.leaves_with_path(model)
                if path_matches(path, "sa_heads.attn")
            ]

            attn_norms = {
                f"gradients/fnorm/{p}": jnp.linalg.norm(v) for p, v in attn_weights
            }
            # TODO not actually gradients... need to figure out how to retrieve
            # those from optimizer state.
            attn_hists = {
                f"gradients/hist/{p}": wandb.Histogram(v) for p, v in attn_weights
            }

            run.track(
                {
                    **train_metrics,
                    **dev_metrics,
                    **attn_norms,
                    **attn_hists,
                    "timing/ms_per_step": window_duration_ms / EVALUATE_ON_STEP,
                    "timing/samples_per_second": samples_processed
                    / window_duration_sec,
                    "timing/step_wait_ms": duration_in_ms(data_start, data_end),
                    "timing/train_exec_ms": duration_in_ms(train_start, train_end),
                },
                step=step,
            )

            # Get ready for the next round
            model.train()
            window_start = perf_counter()

    log.info("Done training, saving weights")
    run.checkpoint("final", model)


def train(cfg: AIAYNConfig) -> None:
    """Train a GPT2."""

    rngs = nnx.Rngs(cfg.seed)

    log.info("Loading dataset")
    tok, train_data, dev_data = create_dataset("roneneldan/TinyStories")

    log.info("Starting training run with config %s", cfg)
    log.info("Initializing model")
    model = AIAYN(cfg, len(tok.vocab))

    log.info("Initializing optimizer")
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(
            optax.warmup_cosine_decay_schedule(
                init_value=cfg.learning_rate / 10,
                peak_value=cfg.learning_rate,
                warmup_steps=cfg.train_steps // 100,
                decay_steps=cfg.train_steps,
            ),
            cfg.momentum,
            weight_decay=0.1,
        ),
        wrt=nnx.Param,
    )

    log.info("Initializing metrics")
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average("loss")
    )

    log.info("Preparing to train")
    if cfg.dry_run:
        log.info("Dry run, exiting before training")
        return

    run_cfg = cfg.to_dict()
    run_cfg.update(
        {
            "dry_run": cfg.dry_run,
            "run_path": cfg.run_path,
            "checkpoint_label": cfg.checkpoint_label,
        }
    )
    run = Run.from_config("gpt2", run_cfg)

    _run_training_loop(
        run,
        cfg,
        model,
        optimizer,
        metrics,
        rngs,
        train_data,
        dev_data,
    )
    run.finalize()
