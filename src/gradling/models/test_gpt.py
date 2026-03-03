import pytest
from flax import nnx

from gradling.data import make_loader, prepare_training_data
from gradling.models.gpt import GPT, Config
from gradling.tokenizers import CharacterTokenizer

CORPUS = "The quick brown fox jumped over the lazy dog."


@pytest.fixture
def setup():
    tok = CharacterTokenizer.train(CORPUS)
    train, _ = prepare_training_data(tok, CORPUS)
    rngs = nnx.Rngs(0)
    loader = make_loader(rngs, 8, 8, train)

    def model(**kwargs):
        cfg = Config(
            seed=42,
            batch_size=32,
            n_vocab=len(tok.vocab),
            n_ctx=8,
            n_emb=32,
            head_size=32,
            num_heads=4,
            num_blocks=3,
            dropout=0.1,
            learning_rate=1e-3,
            momentum=0.9,
            train_steps=1,
        )
        cfg = cfg._replace(**kwargs)
        return GPT(cfg)

    yield model, loader


def test_sweep_head_size(subtests, setup):
    cases = [32, 64, 128]
    model, loader = setup

    for case in cases:
        with subtests.test(msg=f"head_size {case}"):
            xs, _ = next(loader)
            model(head_size=case, n_emb=case)(xs)
