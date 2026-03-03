from collections.abc import Callable

import jax
import optax
from flax import nnx

from gradling import logger
from gradling.data import sample_batch
from gradling.models.gpt import GPT, Config
from gradling.run import Run

log = logger.get(__name__)


def loss_fn(model, logits, ys):
    logits = model(logits)
    loss = optax.softmax_cross_entropy(logits, ys).mean()
    return loss, logits


@nnx.jit(static_argnums=(0,))
def train_step(
    cfg: Config,
    model: GPT,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    train_data: jax.Array,
):
    xs, ys = sample_batch(rngs, train_data, cfg.batch_size, cfg.n_ctx)
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, xs, nnx.one_hot(ys, cfg.n_vocab))
    metrics.update(loss=loss, logits=logits, labels=ys)
    optimizer.update(model, grads)


@nnx.jit(static_argnums=(0,))
def eval_step(
    cfg: Config,
    model: GPT,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    dev_data: jax.Array,
):
    xs, ys = sample_batch(rngs, dev_data, cfg.batch_size, cfg.n_ctx)
    loss, logits = loss_fn(model, xs, nnx.one_hot(ys, cfg.n_vocab))
    metrics.update(loss=loss, logits=logits, labels=ys)


def train(
    run: Run,
    cfg: Config,
    model: GPT,
    optimizer,
    metrics,
    rngs: nnx.Rngs,
    train_data: jax.Array,
    dev_data: jax.Array,
    metric_sinks: list[Callable],
):
    def track(metric: str, value: any):
        for sink in metric_sinks:
            sink(metric, value)

    eval_cfg = cfg._replace(batch_size=512)
    for step in range(cfg.train_steps):
        if step % 100 == 0:
            log.info(f"Training for step {step}/{cfg.train_steps}")

        model.train()

        train_step(cfg, model, optimizer, metrics, rngs, train_data)

        if step % 100 == 0:
            # Record training metrics.
            for metric, value in metrics.compute().items():
                track(f"train_{metric}", value)
            metrics.reset()

            # Run through a batch of 512 for dev loss.
            model.eval()
            eval_step(eval_cfg, model, metrics, rngs, dev_data)

            # Record dev metrics
            for metric, value in metrics.compute().items():
                track(f"dev_{metric}", value)
            metrics.reset()

    log.info("Done training, saving weights")
    run.checkpoint("final", model)
