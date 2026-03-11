import jax
from flax import nnx
from jax import numpy as jnp


class RMSNorm(nnx.Module):
    def __init__(self, n: int, e=1e-9):
        self.e = e
        self.g = nnx.Param(jnp.ones((n,)))

    def __call__(self, x: jax.Array):
        mean = jnp.mean(x**2, axis=-1)
        rms = jnp.sqrt(mean + self.e)
        return (x / rms[..., None]) * self.g[...]
