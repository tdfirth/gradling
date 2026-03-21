import jax
from flax import nnx
from jax import numpy as jnp

Initializer = nnx.initializers.Initializer


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
        kernel_init: Initializer | None = None,
        proj_kernel_init: Initializer | None = None,
    ):
        assert n_embd % num_heads == 0
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads

        attn_kwargs: dict = dict(
            in_features=n_embd,
            out_features=n_embd * 3,
            use_bias=False,
            rngs=rngs,
        )
        if kernel_init is not None:
            attn_kwargs["kernel_init"] = kernel_init

        self.attn = nnx.Linear(**attn_kwargs)
        self.mask = nnx.Variable(
            jnp.tril(jnp.ones((ctx_len, ctx_len))).reshape(1, 1, ctx_len, ctx_len) == 0
        )

        proj_kwargs: dict = dict(in_features=n_embd, out_features=n_embd, rngs=rngs)
        if proj_kernel_init is not None:
            proj_kwargs["kernel_init"] = proj_kernel_init
        elif kernel_init is not None:
            proj_kwargs["kernel_init"] = kernel_init

        self.proj = nnx.Linear(**proj_kwargs)
        self.a_dropout = nnx.Dropout(dropout, rngs=rngs)
        self.r_dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jax.Array):
        B, T, C = x.shape

        # Split the projection into three along the channel dimension.
        key, query, value = jnp.split(self.attn(x), 3, axis=2)

        # Reshape k, q and v to (B, T, num_heads, head_dim)
        key = key.reshape(B, T, self.num_heads, self.head_dim)
        query = query.reshape(B, T, self.num_heads, self.head_dim)
        value = value.reshape(B, T, self.num_heads, self.head_dim)

        # (B, Q, nh, hd) @ (B, K, nh, hd) -> (B, nh, Q, K)
        attn = jnp.einsum("bqhe,bkhe->bhqk", query, key) * (1 / jnp.sqrt(self.head_dim))

        # Causal mask (e.g. q at 1 can only attend to k at 0 and 1).
        attn = jnp.where(self.mask[...], -jnp.inf, attn)

        # Turn wei into a probability distribution along K.
        attn = nnx.softmax(attn, axis=-1)
        attn = self.a_dropout(attn)

        # (B, nh, Q, K) @ (B, K, nh, hd) -> (B, Q, nh, hd)
        x = jnp.einsum("bhqk,bkhe->bqhe", attn, value)

        x = x.reshape(B, T, C)
        x = self.proj(x)
        x = self.r_dropout(x)
        return x
