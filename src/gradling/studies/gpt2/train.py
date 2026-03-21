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
from gradling.data import load, loader, random_iterator
from gradling.run import Run
from gradling.studies.gpt2.config import GPTConfig
from gradling.studies.gpt2.model import GPT2

log = logger.get(__name__)


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


def _accumulate_micro_batches(step_fn, init_carry, xs, ys, micro_batch_size):
    n = xs.shape[0] // micro_batch_size
    micro_xs = xs.reshape(n, micro_batch_size, -1)
    micro_ys = ys.reshape(n, micro_batch_size, -1)

    def body(i, carry):
        return step_fn(carry, micro_xs[i], micro_ys[i])

    return jax.lax.fori_loop(0, n, body, init_carry), n


def _run_training_loop(
    run: Run,
    cfg: GPTConfig,
    model: GPT2,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    train_data: np.memmap,
    dev_data: np.memmap,
) -> None:

    @nnx.jit
    def _train_step(
        model: GPT2,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        rngs: nnx.Rngs,
        xs: jax.Array,
        ys: jax.Array,
    ):
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)

        def pure_loss(params, xs_i, ys_i):
            m = nnx.merge(graphdef, params, rest)
            logits = m(xs_i)
            return optax.softmax_cross_entropy_with_integer_labels(logits, ys_i).mean()

        grad_fn = jax.value_and_grad(pure_loss)

        def train_body(carry, xs_i, ys_i):
            acc_grads, acc_loss = carry
            loss, grads = grad_fn(params, xs_i, ys_i)
            acc_grads = jax.tree.map(jnp.add, acc_grads, grads)
            return acc_grads, acc_loss + loss

        zero_grads = jax.tree.map(jnp.zeros_like, params)
        (acc_grads, acc_loss), n = _accumulate_micro_batches(
            train_body, (zero_grads, jnp.array(0.0)), xs, ys, cfg.micro_batch_size
        )

        acc_grads = jax.tree.map(lambda g: g / n, acc_grads)
        acc_loss = acc_loss / n

        optimizer.update(model, acc_grads)
        metrics.update(loss=acc_loss)

    @nnx.jit
    def _eval_step(
        model: GPT2,
        metrics: nnx.MultiMetric,
        rngs: nnx.Rngs,
        xs: jax.Array,
        ys: jax.Array,
    ):
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)

        def pure_loss(params, xs_i, ys_i):
            m = nnx.merge(graphdef, params, rest)
            logits = m(xs_i)
            return optax.softmax_cross_entropy_with_integer_labels(logits, ys_i).mean()

        def eval_body(acc_loss, xs_i, ys_i):
            return acc_loss + pure_loss(params, xs_i, ys_i)

        acc_loss, n = _accumulate_micro_batches(
            eval_body, jnp.array(0.0), xs, ys, cfg.micro_batch_size
        )
        metrics.update(loss=acc_loss / n)

    nrng = np.random.Generator(np.random.PCG64(seed=cfg.seed))
    train_iterator = islice(
        random_iterator(nrng, cfg.batch_size, cfg.n_ctx, train_data),
        cfg.train_steps,
    )
    dev_iterator = loader(
        islice(
            random_iterator(nrng, cfg.batch_size, cfg.n_ctx, dev_data),
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
            tokens_processed = cfg.batch_size * cfg.n_ctx * EVALUATE_ON_STEP

            attn_weights = [
                [format_path(path), value]
                for path, value in jax.tree.leaves_with_path(model)
                if path_matches(path, "sa.attn")
            ]

            attn_norms = {
                f"weights/fnorm/{p}": jnp.linalg.norm(v) for p, v in attn_weights
            }
            attn_hists = {
                f"weights/hist/{p}": wandb.Histogram(v) for p, v in attn_weights
            }

            run.track(
                {
                    **train_metrics,
                    **dev_metrics,
                    **attn_norms,
                    **attn_hists,
                    "timing/ms_per_step": window_duration_ms / EVALUATE_ON_STEP,
                    "timing/tokens_per_second": tokens_processed / window_duration_sec,
                    "timing/step_wait_ms": duration_in_ms(data_start, data_end),
                    "timing/train_exec_ms": duration_in_ms(train_start, train_end),
                },
                step=step,
            )

            model.train()
            window_start = perf_counter()

    log.info("Done training, saving weights")
    run.checkpoint("final", model)


def train(run: Run[GPTConfig]) -> None:
    cfg = run.cfg

    rngs = nnx.Rngs(cfg.seed)

    assert cfg.batch_size % cfg.micro_batch_size == 0, (
        f"batch_size ({cfg.batch_size}) must be divisible by "
        f"micro_batch_size ({cfg.micro_batch_size})"
    )
    grad_accum_steps = cfg.batch_size // cfg.micro_batch_size
    log.info(
        "Gradient accumulation: %d micro-batches of %d",
        grad_accum_steps,
        cfg.micro_batch_size,
    )

    log.info("Loading dataset")
    tok, train_data, dev_data = load(
        run.ctx.root, "HuggingFaceFW/fineweb-edu", config="sample-10BT"
    )

    log.info("Starting training run with config %s", cfg)
    log.info("Initializing model")
    model = GPT2(cfg, len(tok.vocab))

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
