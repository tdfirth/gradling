import jax
from flax import nnx
from jax import numpy as jnp


class LayerNorm(nnx.Module):
    def __init__(self, size: int, rngs: nnx.Rngs):
        self.gamma = nnx.Param(jnp.ones((size,)))
        self.beta = nnx.Param(jnp.zeros((size,)))

    def __call__(self, x: jax.Array):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-6)
        x = x * self.gamma[...] + self.beta[...]
        return x
