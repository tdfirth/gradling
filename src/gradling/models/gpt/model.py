from typing import NamedTuple

import jax
from flax import nnx
from jax import numpy as jnp

from gradling.modules import FeedForward, LayerNorm, MultiHeadAttention


class ModelConfig(NamedTuple):
    seed: int
    batch_size: int
    n_vocab: int
    n_ctx: int
    n_emb: int
    head_size: int
    num_heads: int
    num_blocks: int
    dropout: float
    learning_rate: float
    momentum: float
    train_steps: int


class AttentionBlock(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        n_ctx: int,
        n_emb: int,
        head_size: int,
        dropout: float,
        rngs: nnx.Rngs,
    ):
        self.sa_heads = MultiHeadAttention(
            n_ctx,
            n_embd=n_emb,
            num_heads=num_heads,
            rngs=rngs,
            dropout=dropout,
        )
        self.ff = FeedForward(n_emb, dropout, rngs)
        self.ln1 = LayerNorm(n_emb, rngs)
        self.ln2 = LayerNorm(n_emb, rngs)

    def __call__(self, x: jax.Array):
        x = self.ln1(x)
        x = x + self.sa_heads(x)
        x = self.ln2(x)
        x = x + self.ff(x)
        return x


class GPT(nnx.Module):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        rngs = nnx.Rngs(cfg.seed)
        self.tok_emb = nnx.Embed(
            num_embeddings=cfg.n_vocab, features=cfg.n_emb, rngs=rngs
        )
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
                )
                for _ in range(cfg.num_blocks)
            ],
            nnx.LayerNorm(cfg.n_emb, rngs=rngs),
        )
        self.lm_head = nnx.Linear(
            in_features=cfg.n_emb, out_features=cfg.n_vocab, rngs=rngs
        )

    def __call__(self, x: jax.Array):
        _, T = x.shape
        tok_embs = self.tok_emb(x)
        pos_embs = self.pos_emb(jnp.arange(0, T))
        x = tok_embs + pos_embs
        x = self.blocks(x)
        x = self.lm_head(x)
        return x
