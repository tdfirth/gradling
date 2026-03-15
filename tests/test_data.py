from itertools import islice
from pathlib import Path

import jax
import numpy as np
import pytest
from flax import nnx

from gradling.data import (
    DatasetMeta,
    dataset_dir,
    jax_random_iterator,
    load,
    loader,
    prepare_training_data,
    read_meta,
    write_meta,
)
from gradling.tokenizers import CharacterTokenizer

CORPUS = "The quick brown fox jumped over the lazy dog."


@pytest.fixture
def tok():
    yield CharacterTokenizer.train(CORPUS)


def test_prepare_training_data(tok):
    train, dev = prepare_training_data(tok, CORPUS)
    assert tok.decode(train.tolist()) == "The quick brown fox jumped over the lazy"
    assert tok.decode(dev.tolist()) == " dog."


def test_empty_loader():
    got = list(loader(iter([])))
    assert got == []


def test_single_item_loader():
    data = np.ones((2, 2))
    got = list(loader(iter([data])))
    assert len(got) == 1
    assert np.all(got[0] == data)


def test_loader(tok):
    n = 20
    train, _ = prepare_training_data(tok, CORPUS)
    batch_generator = jax_random_iterator(nnx.Rngs(0), 8, 8, train)
    slice = list(islice(batch_generator, n))
    want = np.array(slice)

    def batch_it():
        yield from want

    got = np.array(list(loader(batch_it())))
    assert len(got) == n
    assert np.all(want == got)


def test_loader_transfers_to_device(tok):
    n = 4
    train, _ = prepare_training_data(tok, CORPUS)
    batch_generator = jax_random_iterator(nnx.Rngs(0), 8, 8, train)
    slice = list(islice(batch_generator, n))

    def batch_it():
        yield from np.array(slice)

    for batch in loader(batch_it()):
        assert isinstance(batch, jax.Array)


def test_handles_exception_in_iterator():
    def explodes():
        yield np.ones((2, 2))
        yield np.ones((2, 2))
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        list(loader(explodes()))


def test_dataset_dir():
    root = Path("/project")
    assert dataset_dir(root, "roneneldan/TinyStories") == (
        Path("/project/data/roneneldan/TinyStories")
    )


def test_write_read_meta_roundtrip(tmp_path):
    meta = DatasetMeta(
        source="roneneldan/TinyStories",
        repo="tdfirth/TinyStories",
        tokenizer_name="gpt2",
        dtype="uint16",
        train_tokens=100,
        dev_tokens=50,
    )
    write_meta(tmp_path, meta)
    got = read_meta(tmp_path)
    assert got == meta


def test_load_errors_without_toml(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset.toml"):
        load(tmp_path, "roneneldan/TinyStories")


def test_load_errors_without_npy_files(tmp_path):
    d = tmp_path / "data" / "roneneldan" / "TinyStories"
    d.mkdir(parents=True)
    meta = DatasetMeta(
        source="roneneldan/TinyStories",
        repo="tdfirth/TinyStories",
        tokenizer_name="gpt2",
        dtype="uint16",
        train_tokens=100,
        dev_tokens=50,
    )
    write_meta(d, meta)

    with pytest.raises(FileNotFoundError, match="datasets pull"):
        load(tmp_path, "roneneldan/TinyStories")
