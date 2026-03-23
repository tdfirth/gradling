import jax
from flax import nnx
from jax import numpy as jnp


class LayerNorm(nnx.Module):
    def __init__(self, size: int):
        self.gamma = nnx.Param(jnp.ones((size,)))
        self.beta = nnx.Param(jnp.zeros((size,)))

    def __call__(self, x: jax.Array):
        in_dtype = x.dtype
        # LayerNorm is numerically unstable, and so we ensure it's in float32.
        # bf16 has only 3 decimal digits of precistion and this is not enough
        # to represent variance.
        x = x.astype(jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + 1e-6)
        # Jax will actually prompote gamma and beta to float32 anyway, but best
        # to be explicit
        x = x * self.gamma[...].astype(jnp.float32) + self.beta[...].astype(jnp.float32)
        x = x.astype(in_dtype)
        return x
