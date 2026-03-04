import jax
from flax import nnx
from jax import numpy as jnp


class MultiHeadAttention(nnx.Module):
    """The rough plan is as follows:
    - First we need to create the three linear layers in one big matrix with an
      extra dimension for the head number.
    - Then we need to implement the attention algorithm against that along the
      head dimension.
    - Then we need to reshape the output to be out expected output shape.

    input = (B, T, C)
    attn = (n_heads, T, C // n_heads)
    """

    def __init__(
        self,
        ctx_len: int,
        n_embd: int,
        num_heads: int,
        dropout: float,
        rngs: nnx.Rngs,
    ):
        assert n_embd % num_heads == 0
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads
        self.attn = nnx.Linear(
            in_features=n_embd,
            out_features=n_embd * 3,
            use_bias=False,
            rngs=rngs,
        )
        self.mask = nnx.Variable(
            jnp.tril(jnp.ones((ctx_len, ctx_len))).reshape(1, 1, ctx_len, ctx_len) == 0
        )
        self.proj = nnx.Linear(in_features=n_embd, out_features=n_embd, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jax.Array):
        B, T, C = x.shape

        # Split the projection into three along the channel dimension.
        key, query, value = jnp.split(self.attn(x), 3, axis=2)

        # Reshape k, q and v to (B, T, num_heads, head_dim)
        key = key.reshape(B, T, self.num_heads, self.head_dim)
        query = query.reshape(B, T, self.num_heads, self.head_dim)
        value = value.reshape(B, T, self.num_heads, self.head_dim)

        # (B, Q, num_heads, head_dim) @ (B, K, num_heads, head_dim) -> (B, num_heads, Q, K)
        wei = jnp.einsum("bqhe,bkhe->bhqk", query, key) * (1 / jnp.sqrt(self.head_dim))

        # Causal mask (e.g. q at 1 can only attend to k at 0 and 1).
        wei = jnp.where(self.mask[...], -jnp.inf, wei)

        # Turn wei into a probability distribution along K.
        wei = nnx.softmax(wei, axis=-1)

        # (B, num_heads, Q, K) @ (B, K, num_heads, head_dim) -> (B, Q, num_heads, head_dim)
        x = jnp.einsum("bhqk,bkhe->bqhe", wei, value)

        x = x.reshape(B, T, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x
