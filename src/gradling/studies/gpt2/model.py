import math

import jax
from flax import nnx
from jax import numpy as jnp

from gradling.modules import LayerNorm, MultiHeadAttention
from gradling.modules.activations import gelu
from gradling.studies.gpt2.config import GPTConfig

_INIT_STDDEV = 0.02


def _init(stddev: float = _INIT_STDDEV) -> nnx.initializers.Initializer:
    return nnx.initializers.normal(stddev=stddev)


def _residual_init(n_layers: int) -> nnx.initializers.Initializer:
    return nnx.initializers.normal(stddev=_INIT_STDDEV / math.sqrt(2 * n_layers))


class FeedForward(nnx.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        rngs: nnx.Rngs,
    ):
        self.fc = nnx.Linear(
            in_features=d_model,
            out_features=4 * d_model,
            rngs=rngs,
            use_bias=True,
            kernel_init=_init(),
        )
        self.proj = nnx.Linear(
            in_features=4 * d_model,
            out_features=d_model,
            rngs=rngs,
            use_bias=True,
            kernel_init=_residual_init(n_layers),
        )
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = gelu(self.fc(x))
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        cfg: GPTConfig,
        rngs: nnx.Rngs,
    ):
        self.sa = MultiHeadAttention(
            cfg.n_ctx,
            n_embd=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            rngs=rngs,
            kernel_init=_init(),
            proj_kernel_init=_residual_init(cfg.n_layers),
        )
        self.ff = FeedForward(cfg.d_model, cfg.n_layers, cfg.dropout, rngs)
        self.ln1 = LayerNorm(cfg.d_model)
        self.ln2 = LayerNorm(cfg.d_model)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT2(nnx.Module):
    def __init__(self, cfg: GPTConfig, n_vocab: int):
        self.cfg = cfg
        self.n_vocab = n_vocab
        rngs = nnx.Rngs(cfg.seed)

        self.tok_emb = nnx.Embed(
            num_embeddings=n_vocab,
            features=cfg.d_model,
            rngs=rngs,
            embedding_init=_init(),
        )
        self.pos_emb = nnx.Embed(
            num_embeddings=cfg.n_ctx,
            features=cfg.d_model,
            rngs=rngs,
            embedding_init=_init(),
        )
        self.blocks = nnx.Sequential(
            *[TransformerBlock(cfg, rngs) for _ in range(cfg.n_layers)],
        )
        self.ln_f = LayerNorm(cfg.d_model)

    def __call__(self, x: jax.Array) -> jax.Array:
        _, T = x.shape
        tok_embs = self.tok_emb(x)
        pos_embs = self.pos_emb(jnp.arange(0, T))
        x = tok_embs + pos_embs
        x = self.blocks(x)
        x = self.ln_f(x)
        x = x @ self.tok_emb.embedding.value.T
        return x
