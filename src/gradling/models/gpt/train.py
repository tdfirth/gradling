from __future__ import annotations

import re
from time import perf_counter

import jax
import optax
from flax import nnx
from fontTools.misc.iterTools import islice
from jax import numpy as jnp

import wandb
from gradling.data import loader, prepare_training_data, random_iterator
from gradling.models.gpt.common import load_corpus, log
from gradling.models.gpt.config import GPTConfig
from gradling.models.gpt.model import GPT
from gradling.run import Run
from gradling.tokenizers import CharacterTokenizer


def _loss_fn(model: GPT, xs: jax.Array, ys: jax.Array):
    logits = model(xs)
    loss = optax.softmax_cross_entropy(logits, ys).mean()
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
    cfg: GPTConfig,
    model: GPT,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    train_data: jax.Array,
    dev_data: jax.Array,
) -> None:

    @nnx.jit
    def _train_step(
        model: GPT,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        rngs: nnx.Rngs,
        xs: jax.Array,
        ys: jax.Array,
    ):
        grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model, xs, nnx.one_hot(ys, model.n_vocab))
        metrics.update(loss=loss, logits=logits, labels=ys)
        optimizer.update(model, grads)

    @nnx.jit
    def _eval_step(
        model: GPT,
        metrics: nnx.MultiMetric,
        rngs: nnx.Rngs,
        xs: jax.Array,
        ys: jax.Array,
    ):
        # xs, ys = sample_batch(rngs, dev_data, cfg.batch_size * 4, cfg.n_ctx)
        loss, logits = _loss_fn(model, xs, nnx.one_hot(ys, model.n_vocab))
        metrics.update(loss=loss, logits=logits, labels=ys)

    # Set up the data iterators
    train_iterator = islice(
        random_iterator(rngs, cfg.batch_size, cfg.n_ctx, train_data), cfg.train_steps
    )
    dev_iterator = loader(
        islice(
            random_iterator(rngs, cfg.batch_size * 4, cfg.n_ctx, dev_data),
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


def train(cfg: GPTConfig) -> None:
    """Train a GPT."""

    rngs = nnx.Rngs(cfg.seed)

    log.info("Loading data")
    corpus = load_corpus()

    log.info("Training tokenizer")
    tok = CharacterTokenizer.train(corpus)
    log.info("Creating test, dev split")
    train_data, dev_data = prepare_training_data(tok, corpus)

    log.info("Starting training run with config %s", cfg)
    log.info("Initializing model")
    model = GPT(cfg, len(tok.vocab))

    log.info("Initializing optimizer")
    optimizer = nnx.Optimizer(
        model,
        # optax.adamw(cfg.learning_rate, cfg.momentum, weight_decay=0.1),
        optax.adamw(
            optax.warmup_cosine_decay_schedule(
                init_value=0.01,
                peak_value=0.01,
                warmup_steps=100,
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
    run = Run.from_config("gpt", run_cfg)

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
