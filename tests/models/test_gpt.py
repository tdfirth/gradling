import pytest
from flax import nnx

from gradling.data import make_loader, prepare_training_data
from gradling.models.gpt import GPT, ModelConfig
from gradling.tokenizers import CharacterTokenizer

CORPUS = "The quick brown fox jumped over the lazy dog."


@pytest.fixture
def setup():
    tok = CharacterTokenizer.train(CORPUS)
    train, _ = prepare_training_data(tok, CORPUS)
    rngs = nnx.Rngs(0)
    loader = make_loader(rngs, 2, 4, train)

    def model(**kwargs):
        cfg = ModelConfig(
            seed=42,
            batch_size=2,
            n_vocab=len(tok.vocab),
            n_ctx=4,
            n_emb=8,
            head_size=8,
            num_heads=2,
            num_blocks=1,
            dropout=0.1,
            learning_rate=1e-3,
            momentum=0.9,
            train_steps=1,
        )
        cfg = cfg._replace(**kwargs)
        return GPT(cfg)

    yield model, loader


def test_sweep_embedding_size(subtests, setup):
    cases = [8, 16]
    model, loader = setup

    for case in cases:
        with subtests.test(msg=f"n_emb {case}"):
            xs, _ = next(loader)
            model(n_emb=case)(xs)
