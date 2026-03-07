from collections.abc import Iterator
from queue import Queue
from threading import Thread

import jax
from flax import nnx
from jax import numpy as jnp

from gradling.dir import ROOT
from gradling.tokenizers import Tokenizer

DATA = ROOT / "data"
NAMES = DATA / "names.txt"
SHAKESPEARE = DATA / "shakespeare.txt"


# TODO these all create the data on device immediately, need to move to a numpy
# version and then rely on the loader to move things to device.
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


def random_iterator(
    rngs: nnx.Rngs, batch_size: int, ctx_length: int, data: jax.Array
) -> Iterator[tuple[jax.Array, jax.Array]]:
    static_params = ((batch_size,), 0, len(data) - ctx_length)
    while True:
        offsets = jax.random.randint(rngs(), *static_params)
        xs = data[offsets[:, None] + jnp.arange(ctx_length)[None, :]]
        ys = data[(offsets + 1)[:, None] + jnp.arange(ctx_length)[None, :]]
        yield xs, ys


def loader(batch_iterator: Iterator, size: int = 2):
    q = Queue(maxsize=size)

    def produce():
        try:
            for batch in batch_iterator:
                q.put(jax.device_put(batch))
        except Exception as e:
            q.put(e)
        finally:
            q.put(None)

    Thread(target=produce, daemon=True).start()

    while (batch := q.get()) is not None:
        if isinstance(batch, Exception):
            raise batch  # Re-raise the exception on the main thread.
        yield batch
