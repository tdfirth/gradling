from flax import nnx

from gradling.data import make_loader, prepare_training_data
from gradling.tokenizers import CharacterTokenizer

CORPUS = "The quick brown fox jumped over the lazy dog."


def test_prepare_training_data():
    tok = CharacterTokenizer.train(CORPUS)
    train, dev = prepare_training_data(tok, CORPUS)
    assert tok.decode(train.tolist()) == "The quick brown fox jumped over the lazy"
    assert tok.decode(dev.tolist()) == " dog."


def test_loader():
    tok = CharacterTokenizer.train(CORPUS)
    train, _ = prepare_training_data(tok, CORPUS)
    rng = nnx.Rngs(0)
    loader = make_loader(rng, 4, 8, train)
    xs, ys = next(loader)
    for batch in range(4):
        _xs = xs[batch]
        _ys = ys[batch]
        assert len(_xs) == 8
        assert len(_ys) == 8

        xdec = tok.decode(_xs.tolist())
        ydec = tok.decode(_ys.tolist())
        assert xdec in CORPUS
        assert ydec in CORPUS
        assert xdec[1:] == ydec[:-1]


def test_loader_can_take_many():
    tok = CharacterTokenizer.train(CORPUS)
    train, _ = prepare_training_data(tok, CORPUS)
    rng = nnx.Rngs(0)
    loader = make_loader(rng, 4, 8, train)
    for _ in range(100):
        xs, ys = next(loader)
        assert xs.shape == (4, 8)
        assert ys.shape == (4, 8)
