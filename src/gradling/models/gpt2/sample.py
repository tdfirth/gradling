from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from jax import random
from transformers import TokenizersBackend

from gradling import logger
from gradling.data import (
    create_dataset,
    random_iterator,
)
from gradling.models.gpt2.config import GPT2Config
from gradling.models.gpt2.model import GPT2
from gradling.run import Run

log = logger.get(__name__)


class _SampleState(NamedTuple):
    key: jax.Array
    output: jax.Array
    ctx_buf: jax.Array


def sample_tokens(
    model: nnx.Module,
    tok: TokenizersBackend,
    inputs: jax.Array,
    max_tokens: int = 1024,
) -> None:
    @nnx.jit
    def _sample(ctx: jax.Array):
        key = random.key(42)
        B, T = ctx.shape
        output = jnp.empty((B, max_tokens), dtype=jnp.uint16)
        ctx_buf = ctx.copy()

        def fn(i: int, state: _SampleState):
            logits = model(state.ctx_buf)[:, -1, :]
            key, sk = random.split(state.key)
            next_tokens = random.categorical(sk, logits)
            output = state.output.at[:, i].set(next_tokens)
            ctx_buf = jnp.concat([state.ctx_buf[:, 1:T], next_tokens[:, None]], axis=1)
            return _SampleState(key=key, output=output, ctx_buf=ctx_buf)

        state = nnx.fori_loop(
            0, max_tokens, fn, _SampleState(key=key, output=output, ctx_buf=ctx_buf)
        )
        return state.output

    output = _sample(inputs)
    for i, sample in enumerate(output):
        print(f"Sample {i}")
        print(tok.decode(sample.tolist()))
        print("")


def sample(cfg: GPT2Config) -> None:
    """Sample from a GPT2."""

    if not cfg.run_path:
        msg = "run_path is required for sample command."
        raise ValueError(msg)

    log.info("Loading run")
    run = Run.from_path(Path(cfg.run_path))
    cfg = GPT2Config(**run.cfg)

    log.info("Loading dataset")
    tok, _, dev_data = create_dataset("roneneldan/TinyStories")

    nrng = np.random.Generator(np.random.PCG64(seed=cfg.seed))
    it = random_iterator(nrng, cfg.batch_size, cfg.n_ctx, dev_data)

    log.info("Initializing model")
    model = GPT2(cfg, len(tok.vocab))

    log.info("Restoring weights")
    run.load_checkpoint(cfg.checkpoint_label, model)

    log.info("Preparing inputs")
    xs, _ = next(it)
    model.eval()
    sample_tokens(model, tok, jnp.array(xs[0:1]).astype(jnp.int32))
    run.finalize()
