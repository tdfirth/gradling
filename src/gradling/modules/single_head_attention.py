import jax
from flax import nnx
from jax import numpy as jnp


class SingleHeadAttention(nnx.Module):
    def __init__(
        self,
        ctx_len: int,
        n_embd: int,
        dropout: float,
        rngs: nnx.Rngs,
        head_size: int | None = None,
    ):
        self.n_ctx = ctx_len
        self.n_embd = n_embd
        self.head_size = n_embd if head_size is None else head_size
        self.key = nnx.Linear(
            in_features=n_embd, out_features=self.head_size, use_bias=False, rngs=rngs
        )
        self.query = nnx.Linear(
            in_features=n_embd, out_features=self.head_size, use_bias=False, rngs=rngs
        )
        self.value = nnx.Linear(
            in_features=n_embd, out_features=self.head_size, use_bias=False, rngs=rngs
        )
        self.mask = nnx.Variable(jnp.tril(jnp.ones((ctx_len, ctx_len))) == 0)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.proj = nnx.Linear(
            in_features=self.head_size,
            out_features=self.head_size,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = jnp.einsum("bqc,bkc->bqk", q, k) * self.n_embd**-0.5
        wei = jnp.where(self.mask[...], -jnp.inf, wei)
        wei = nnx.softmax(wei, axis=-1)
        wei = self.dropout(wei)
        return self.proj(wei @ v)
