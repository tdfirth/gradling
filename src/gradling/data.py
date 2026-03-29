from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, cast

import jax
import numpy as np
import tomlkit
from datasets import Dataset, load_dataset
from flax import nnx
from jax import numpy as jnp
from tqdm import tqdm
from transformers import AutoTokenizer, TokenizersBackend

from gradling import logger
from gradling.context import ROOT
from gradling.tokenizers import Tokenizer

log = logger.get(__name__)

DATA = ROOT / "data"
NAMES = DATA / "names.txt"
SHAKESPEARE = DATA / "shakespeare.txt"

DATASET_FILE = "dataset.toml"


@dataclass(frozen=True)
class DatasetMeta:
    source: str
    repo: str
    tokenizer_name: str
    dtype: str
    train_tokens: int
    dev_tokens: int
    config: str | None = None


def dataset_dir(root: Path, dataset_name: str, config: str | None = None) -> Path:
    d = root / "data" / Path(*dataset_name.split("/"))
    if config is not None:
        d = d / config
    return d


def read_meta(path: Path) -> DatasetMeta:
    doc = tomlkit.parse((path / DATASET_FILE).read_text()).unwrap()
    ds = doc["dataset"]
    return DatasetMeta(
        source=ds["source"],
        repo=ds["repo"],
        tokenizer_name=ds["tokenizer"],
        dtype=ds["dtype"],
        train_tokens=ds["train"]["tokens"],
        dev_tokens=ds["dev"]["tokens"],
        config=ds.get("config"),
    )


def write_meta(path: Path, meta: DatasetMeta) -> None:
    doc: dict[str, Any] = {
        "dataset": {
            "source": meta.source,
            "repo": meta.repo,
            "tokenizer": meta.tokenizer_name,
            "dtype": meta.dtype,
            "train": {"tokens": meta.train_tokens},
            "dev": {"tokens": meta.dev_tokens},
        }
    }
    if meta.config is not None:
        doc["dataset"]["config"] = meta.config
    (path / DATASET_FILE).write_text(tomlkit.dumps(doc))


def prepare(root: Path, dataset_name: str, config: str | None = None) -> DatasetMeta:
    d = dataset_dir(root, dataset_name, config)
    toml_path = d / DATASET_FILE
    train_path = d / "train.npy"
    dev_path = d / "dev.npy"

    if toml_path.exists() and train_path.exists() and dev_path.exists():
        log.info("Dataset %s already prepared", dataset_name)
        return read_meta(d)

    d.mkdir(parents=True, exist_ok=True)

    tokenizer = cast(TokenizersBackend, AutoTokenizer.from_pretrained("gpt2"))

    label = f"{dataset_name} ({config})" if config else dataset_name
    log.info("Downloading dataset %s", label)
    ds = load_dataset(dataset_name, name=config).shuffle(seed=42)

    if "validation" in ds:
        train_split = ds["train"]
        dev_split = ds["validation"]
    else:
        log.info("No validation split found, splitting train 99/1")
        split = ds["train"].train_test_split(test_size=0.01, seed=42)
        train_split = split["train"]
        dev_split = split["test"]

    log.info("Tokenizing train split")
    train = tokenize_dataset(tokenizer, train_split, train_path)

    log.info("Tokenizing dev split")
    dev = tokenize_dataset(tokenizer, dev_split, dev_path)

    _, dataset_short = dataset_name.split("/", 1)
    repo_suffix = f"-{config}" if config else ""
    meta = DatasetMeta(
        source=dataset_name,
        config=config,
        repo=f"tdfirth/{dataset_short}{repo_suffix}",
        tokenizer_name="gpt2",
        dtype="uint16",
        train_tokens=len(train),
        dev_tokens=len(dev),
    )
    write_meta(d, meta)
    log.info("Dataset prepared at %s", d)
    return meta


def load(
    root: Path, dataset_name: str, config: str | None = None
) -> tuple[TokenizersBackend, np.memmap, np.memmap]:
    d = dataset_dir(root, dataset_name, config)
    toml_path = d / DATASET_FILE

    if not toml_path.exists():
        raise FileNotFoundError(
            f"No dataset.toml for {dataset_name!r}. "
            f"Run 'gradling datasets prepare {dataset_name}' first."
        )

    meta = read_meta(d)
    train_path = d / "train.npy"
    dev_path = d / "dev.npy"

    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(
            f"Data files missing for {dataset_name!r}. "
            f"Run 'gradling datasets pull {dataset_name}' first."
        )

    tokenizer = cast(
        TokenizersBackend, AutoTokenizer.from_pretrained(meta.tokenizer_name)
    )
    train = np.memmap(train_path, dtype=np.dtype(meta.dtype))
    dev = np.memmap(dev_path, dtype=np.dtype(meta.dtype))
    return tokenizer, train, dev


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
        remove_columns=dataset.column_names,
        num_proc=32,
    )
    tokenized_samples = tokenized["input_ids"]
    eot = np.array([tokenizer("<|endoftext|>")["input_ids"][0]], dtype=np.uint16)

    log.info("Writing to memory mapped file")
    with open(path, "wb") as f:
        for sample in tqdm(
            tokenized_samples, desc="Writing tokens", position=0, leave=True
        ):
            np.array(sample, dtype=np.uint16).tofile(f)
            eot.tofile(f)

    return np.memmap(path, dtype=np.uint16, mode="r")


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


def loader(batch_iterator: Iterator, size: int = 2, dtype=np.int32):
    q = Queue(maxsize=size)

    def produce():
        try:
            for batch in batch_iterator:
                q.put(jax.device_put(jax.tree.map(lambda x: x.astype(dtype), batch)))
        except Exception as e:
            q.put(e)
        finally:
            q.put(_SENTINEL)

    Thread(target=produce, daemon=True).start()

    while (batch := q.get()) is not _SENTINEL:
        if isinstance(batch, Exception):
            raise batch
        yield batch
