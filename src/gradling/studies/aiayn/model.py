from typing import Literal, TypeGuard, get_args

import jax
from flax import nnx
from jax import numpy as jnp

from gradling import logger
from gradling.modules import LayerNorm, MultiHeadAttention
from gradling.modules.glu import Geglu, Swiglu
from gradling.studies.aiayn.config import AIAYNConfig

log = logger.get(__name__)

NonLinearity = Literal["relu", "swiglu", "geglu"]


def scale_by(n: int, factor: float) -> int:
    return int(256 * (n * factor // 256))


class FeedForward(nnx.Module):
    def __init__(
        self, n_emb: int, dropout: float, rngs: nnx.Rngs, *, nl: NonLinearity = "relu"
    ):
        if nl == "relu":
            log.debug("Using relu FFN")
            self.net = nnx.Sequential(
                nnx.Linear(
                    in_features=n_emb, out_features=4 * n_emb, rngs=rngs, use_bias=True
                ),
                nnx.relu,
                nnx.Linear(
                    in_features=4 * n_emb, out_features=n_emb, rngs=rngs, use_bias=True
                ),
                nnx.Dropout(dropout, rngs=rngs),
            )
        else:
            log.debug(f"Using {nl} FFN")
            nl_mod = Swiglu if nl == "swiglu" else Geglu
            self.net = nnx.Sequential(
                nl_mod(d_model=n_emb, d_hidden=scale_by(n_emb, 8 / 3), rngs=rngs),
                nnx.Dropout(dropout, rngs=rngs),
            )

    def __call__(self, x: jax.Array):
        return self.net(x)


class AttentionBlock(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        n_ctx: int,
        n_emb: int,
        head_size: int,
        dropout: float,
        rngs: nnx.Rngs,
        *,
        nl: NonLinearity = "relu",
    ):
        self.sa_heads = MultiHeadAttention(
            n_ctx,
            n_embd=n_emb,
            num_heads=num_heads,
            rngs=rngs,
            dropout=dropout,
        )
        self.ff = FeedForward(n_emb, dropout, rngs, nl=nl)
        self.ln1 = LayerNorm(n_emb)
        self.ln2 = LayerNorm(n_emb)

    def __call__(self, x: jax.Array):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


def validate_non_linearity(nl: str) -> NonLinearity:
    def is_non_linearity(nl: str) -> TypeGuard[NonLinearity]:
        return nl in get_args(NonLinearity)

    if not is_non_linearity(nl):
        raise ValueError(f"Unsupported nonlinearity {nl}")

    return nl


class AIAYN(nnx.Module):
    def __init__(self, cfg: AIAYNConfig, n_vocab: int):
        self.cfg = cfg
        self.n_vocab = n_vocab
        rngs = nnx.Rngs(cfg.seed)
        self.tok_emb = nnx.Embed(num_embeddings=n_vocab, features=cfg.n_emb, rngs=rngs)
        self.pos_emb = nnx.Embed(
            num_embeddings=cfg.n_ctx, features=cfg.n_emb, rngs=rngs
        )
        self.blocks = nnx.Sequential(
            *[
                AttentionBlock(
                    cfg.num_heads,
                    cfg.n_ctx,
                    cfg.n_emb,
                    cfg.head_size,
                    cfg.dropout,
                    rngs,
                    nl=validate_non_linearity(cfg.nl),
                )
                for _ in range(cfg.num_blocks)
            ],
            nnx.LayerNorm(cfg.n_emb, rngs=rngs),
        )
        self.lm_head = nnx.Linear(
            in_features=cfg.n_emb, out_features=n_vocab, rngs=rngs
        )

    def __call__(self, x: jax.Array):
        _, T = x.shape
        tok_embs = self.tok_emb(x)
        pos_embs = self.pos_emb(jnp.arange(0, T))
        x = tok_embs + pos_embs
        x = self.blocks(x)
        x = self.lm_head(x)
        return x
