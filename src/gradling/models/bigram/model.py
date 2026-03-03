from flax import nnx

from gradling.config import Config


class BigramConfig(Config):
    seed: int
    n_emb: int


class RuntimeBigramConfig(BigramConfig):
    n_vocab: int


class Bigram(nnx.Module):
    def __init__(self, cfg: RuntimeBigramConfig):
        self.cfg = cfg
        self.emb = nnx.Embed(
            num_embeddings=cfg.n_vocab, features=cfg.n_emb, rngs=nnx.Rngs(cfg.seed)
        )

    def __call__(self, xs):
        return self.emb(xs)
