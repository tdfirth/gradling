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
        wei = nnx.softmax(wei)
        wei = self.dropout(wei)
        return self.proj(wei @ v)


# TODO karpathy only had num_heads, and the single head size is governed by the
# embedding dimenson divided by the number of heads... here we are instead
# allowing those things to be independent, which means we're projecting from
# embedding size to whatever the head size is, and then back again... I have no
# idea what is standard yet.
# One source of the deviation is that I made single head attention first, where
# it seemed we wanted this ability, but perhaps that is not standard in multi
# head attention?
class MultiHeadAttention(nnx.Module):
    def __init__(
        self,
        ctx_len: int,
        n_embd: int,
        num_heads: int,
        dropout: float,
        rngs: nnx.Rngs,
    ):
        assert n_embd % num_heads == 0
        self.heads = nnx.List(
            [
                SingleHeadAttention(
                    ctx_len, n_embd, dropout, rngs, head_size=n_embd // num_heads
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nnx.Linear(in_features=n_embd, out_features=n_embd, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x: jax.Array):
        xs = [h(x) for h in self.heads]
        x = jnp.concat(xs, axis=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x
