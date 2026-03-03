from collections.abc import Iterator

import jax
from flax import nnx
from jax import numpy as jnp

from gradling.dir import ROOT
from gradling.tokenizers import Tokenizer

DATA = ROOT / "data"
NAMES = DATA / "names.txt"
SHAKESPEARE = DATA / "shakespeare.txt"


def prepare_training_data(tok: Tokenizer, corpus: str) -> tuple[jax.Array, jax.Array]:
    train_n = int(len(corpus) * 0.9)
    train = jnp.array(tok.encode("".join(list(corpus[:train_n]))), dtype=jnp.int32)
    dev = jnp.array(tok.encode("".join(list(corpus[train_n:]))), dtype=jnp.int32)
    return train, dev


def sample_batch(rngs: nnx.Rngs, data: jax.Array, batch_size: int, ctx_length: int):
    offsets = jax.random.randint(rngs(), (batch_size,), 0, len(data) - ctx_length)
    xs = data[offsets[:, None] + jnp.arange(ctx_length)[None, :]]
    ys = data[(offsets + 1)[:, None] + jnp.arange(ctx_length)[None, :]]
    return xs, ys


def make_loader(
    rngs: nnx.Rngs, batch_size: int, ctx_length: int, data: jax.Array
) -> Iterator[tuple[jax.Array, jax.Array]]:
    static_params = ((batch_size,), 0, len(data) - ctx_length)
    while True:
        offsets = jax.random.randint(rngs(), *static_params)
        xs = data[offsets[:, None] + jnp.arange(ctx_length)[None, :]]
        ys = data[(offsets + 1)[:, None] + jnp.arange(ctx_length)[None, :]]
        yield xs, ys
