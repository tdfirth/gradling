from itertools import islice

import jax
import numpy as np
import pytest
from flax import nnx

from gradling.data import loader, prepare_training_data, random_iterator
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
    batch_generator = random_iterator(nnx.Rngs(0), 8, 8, train)
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
    batch_generator = random_iterator(nnx.Rngs(0), 8, 8, train)
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
