from collections.abc import Iterator
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import cast

import jax
import numpy as np
from datasets import Dataset, load_dataset
from flax import nnx
from jax import numpy as jnp
from tqdm import tqdm
from transformers import AutoTokenizer, TokenizersBackend

from gradling import logger
from gradling.dir import ROOT
from gradling.tokenizers import Tokenizer

DATA = ROOT / "data"
CACHE = DATA / "cache"
NAMES = DATA / "names.txt"
SHAKESPEARE = DATA / "shakespeare.txt"

CACHE.mkdir(parents=True, exist_ok=True)

log = logger.get(__name__)


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


def tokenize_dataset(tokenizer, dataset: Dataset, path: Path) -> np.memmap:
    log.info("Tokenizing data")
    tokenized = dataset.map(
        lambda batch: tokenizer(batch["text"], return_attention_mask=False),
        batched=True,
        remove_columns=["text"],
        num_proc=32,
    )
    tokenized_samples = tokenized["input_ids"]
    token_count = sum(len(s) + 1 for s in tokenized_samples)

    log.info("Writing to memory mapped file")
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(token_count,))
    # TODO this is hardcoded for GPT2 tokenizer
    eot = tokenizer("<|endoftext|>")["input_ids"][0]
    idx = 0
    for sample in tqdm(
        tokenized_samples, desc="Writing tokens", position=0, leave=True
    ):
        arr[idx : idx + len(sample)] = sample
        idx += len(sample)
        arr[idx : idx + 1] = eot
        idx += 1

    return arr


def create_dataset(dataset_name: str) -> tuple[TokenizersBackend, np.memmap, np.memmap]:
    file_name = dataset_name.replace("/", "-")
    test_path = CACHE / f"{file_name}-train.npy"
    dev_path = CACHE / f"{file_name}-dev.npy"
    tokenizer = cast(TokenizersBackend, AutoTokenizer.from_pretrained("gpt2"))
    log.info("Checking if dataset already exists")
    if test_path.exists() and dev_path.exists():
        log.info("Loading cached dataset")
        test = np.memmap(test_path, dtype=np.uint16)
        dev = np.memmap(dev_path, dtype=np.uint16)
        return tokenizer, test, dev
    log.info("Downloading dataset")
    dataset = load_dataset(dataset_name).shuffle(seed=42)
    log.info("Tokenizing test")
    test = tokenize_dataset(tokenizer, dataset["train"], test_path)
    log.info("Tokenizing dev")
    dev = tokenize_dataset(tokenizer, dataset["validation"], dev_path)
    return tokenizer, test, dev


def jax_random_iterator(
    rngs: nnx.Rngs, batch_size: int, ctx_length: int, data: jax.Array
) -> Iterator[tuple[jax.Array, jax.Array]]:
    static_params = ((batch_size,), 0, len(data) - ctx_length)
    while True:
        offsets = jax.random.randint(rngs(), *static_params)
        xs = data[offsets[:, None] + jnp.arange(ctx_length)[None, :]]
        ys = data[(offsets + 1)[:, None] + jnp.arange(ctx_length)[None, :]]
        yield xs, ys


def random_iterator(
    rng: np.random.Generator, batch_size: int, ctx_length: int, data: np.ndarray
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    while True:
        offsets = rng.integers(0, len(data) - ctx_length, size=batch_size)
        xs = data[offsets[:, None] + np.arange(ctx_length)[None, :]]
        ys = data[(offsets + 1)[:, None] + np.arange(ctx_length)[None, :]]
        yield xs, ys


_SENTINEL = object()


def loader(batch_iterator: Iterator, size: int = 2):
    q = Queue(maxsize=size)

    def produce():
        try:
            for batch in batch_iterator:
                q.put(jax.device_put(jax.tree.map(lambda x: x.astype(np.int32), batch)))
        except Exception as e:
            q.put(e)
        finally:
            q.put(_SENTINEL)

    Thread(target=produce, daemon=True).start()

    while (batch := q.get()) is not _SENTINEL:
        if isinstance(batch, Exception):
            raise batch  # Re-raise the exception on the main thread.
        yield batch
