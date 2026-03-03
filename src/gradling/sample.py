from typing import NamedTuple

import jax
from flax import nnx
from jax import numpy as jnp
from jax import random

from gradling.tokenizers import Tokenizer


class State(NamedTuple):
    key: jax.Array
    output: jax.Array
    ctx_buf: jax.Array


def sample(model: nnx.Module, tok: Tokenizer, input: jax.Array, max_tokens=1024):
    @nnx.jit
    def _sample(ctx):
        key = random.key(42)
        B, T = ctx.shape
        output = jnp.empty((B, max_tokens))
        ctx_buf = ctx.copy()

        def fn(i: int, state: State):
            logits = model(state.ctx_buf)[:, -1, :]
            key, sk = random.split(state.key)
            next_tokens = random.categorical(sk, logits)
            output = state.output.at[:, i].set(next_tokens)
            ctx_buf = jnp.concat([state.ctx_buf[:, 1:T], next_tokens[:, None]], axis=1)
            return State(key=key, output=output, ctx_buf=ctx_buf)

        state = nnx.fori_loop(
            0, max_tokens, fn, State(key=key, output=output, ctx_buf=ctx_buf)
        )

        return state.output

    # inputs, _ = next(loader)
    output = _sample(input)

    for i, o in enumerate(output):
        print(f"Sample {i}")
        print(tok.decode(o.tolist()))
        print("")
