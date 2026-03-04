import jax
from flax import nnx
from jax import numpy as jnp

from gradling.models.gpt import GPTConfig
from gradling.modules import LayerNorm, MultiHeadAttention


class FeedForward(nnx.Module):
    def __init__(self, n_emb: int, dropout: float, rngs: nnx.Rngs):
        self.net = nnx.Sequential(
            nnx.Linear(in_features=n_emb, out_features=4 * n_emb, rngs=rngs),
            nnx.relu,
            nnx.Linear(in_features=4 * n_emb, out_features=n_emb, rngs=rngs),
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
    def __init__(self, cfg: GPTConfig, n_vocab: int):
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
