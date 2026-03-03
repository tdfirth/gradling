from typing import NamedTuple

from flax import nnx


class Config(NamedTuple):
    seed: int
    n_vocab: int
    n_emb: int


class Bigram(nnx.Module):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.emb = nnx.Embed(
            num_embeddings=cfg.n_vocab, features=cfg.n_emb, rngs=nnx.Rngs(cfg.seed)
        )

    def __call__(self, xs):
        return self.emb(xs)
