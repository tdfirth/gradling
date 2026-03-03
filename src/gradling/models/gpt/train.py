from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import optax
from flax import nnx

from gradling.data import prepare_training_data, sample_batch
from gradling.models.gpt.common import load_corpus, log
from gradling.models.gpt.config import GPTConfig
from gradling.models.gpt.model import GPT
from gradling.run import Run
from gradling.tokenizers import CharacterTokenizer


def _loss_fn(model: GPT, xs: jax.Array, ys: jax.Array):
    logits = model(xs)
    loss = optax.softmax_cross_entropy(logits, ys).mean()
    return loss, logits


@nnx.jit(static_argnums=(0,))
def _train_step(
    cfg: GPTConfig,
    model: GPT,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    train_data: jax.Array,
):
    xs, ys = sample_batch(rngs, train_data, cfg.batch_size, cfg.n_ctx)
    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, xs, nnx.one_hot(ys, model.n_vocab))
    metrics.update(loss=loss, logits=logits, labels=ys)
    optimizer.update(model, grads)


@nnx.jit(static_argnums=(0,))
def _eval_step(
    cfg: GPTConfig,
    model: GPT,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    dev_data: jax.Array,
):
    xs, ys = sample_batch(rngs, dev_data, cfg.batch_size, cfg.n_ctx)
    loss, logits = _loss_fn(model, xs, nnx.one_hot(ys, model.n_vocab))
    metrics.update(loss=loss, logits=logits, labels=ys)


def _run_training_loop(
    run: Run,
    cfg: GPTConfig,
    model: GPT,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    train_data: jax.Array,
    dev_data: jax.Array,
    metric_sinks: list[Callable[[str, Any], None]],
) -> None:
    def track(metric: str, value: Any):
        for sink in metric_sinks:
            sink(metric, value)

    eval_cfg = cfg.model_copy(update={"batch_size": 512})
    for step in range(cfg.train_steps):
        if step % 100 == 0:
            log.info(f"Training for step {step}/{cfg.train_steps}")

        model.train()
        _train_step(cfg, model, optimizer, metrics, rngs, train_data)

        if step % 100 == 0:
            for metric, value in metrics.compute().items():
                track(f"train_{metric}", value)
            metrics.reset()

            model.eval()
            _eval_step(eval_cfg, model, metrics, rngs, dev_data)
            for metric, value in metrics.compute().items():
                track(f"dev_{metric}", value)
            metrics.reset()

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
        optax.adamw(cfg.learning_rate, cfg.momentum),
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

    run_cfg = cfg.model_dump(mode="json")
    run_cfg.update(
        {
            "dry_run": cfg.dry_run,
            "run_path": cfg.run_path,
            "checkpoint_label": cfg.checkpoint_label,
        }
    )
    run = Run.from_config(run_cfg, family=cfg.config_name())

    def log_loss(metric: str, value: Any):
        log.info(f"{metric}: {value:.4f}")

    _run_training_loop(
        run,
        cfg,
        model,
        optimizer,
        metrics,
        rngs,
        train_data,
        dev_data,
        [log_loss],
    )
    run.finalize()
