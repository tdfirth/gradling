from flax import nnx
from jax import numpy as jnp
from jax import random

from gradling.modules.rms_norm import RMSNorm


def test_shape(subtests):
    rngs = nnx.Rngs(42)
    dims = [
        (2, 4, 8),
        (4, 8, 16),
        (8, 16, 32),
    ]
    for dim in dims:
        _, _, C = dim
        with subtests.test(msg=""):
            x = random.normal(rngs(), dim)
            rms = RMSNorm(C)
            assert rms(x).shape == dim


def test_rms_eq_one_is_identity(subtests):
    # If RMS comes out as one, then the layer should act as the identity
    # function as x/1 = x. Note that this relies on gamma being initialized as
    # ones.
    dim = (1, 1, 8)
    xs = [
        # If the squares sum to 8, then rms should also be 1.
        jnp.ones(dim),
        jnp.array([[[0, 0, 0, 0, 0, 0, 2, 2]]]),
        jnp.array([[[2, 1, 1, 1, 1, 0, 0, 0]]]),
    ]
    for x in xs:
        _, _, C = dim
        with subtests.test(msg=""):
            rms = RMSNorm(C)
            assert jnp.all(rms(x) == x)


def test_scale_invariance(subtests):
    rngs = nnx.Rngs(42)
    dim = (2, 4, 8)
    x = random.normal(rngs(), dim)
    for k in range(1, 6):
        _, _, C = dim
        with subtests.test(msg=""):
            rms = RMSNorm(C)
            assert jnp.allclose(rms(x), rms(x * k))


def test_handles_zero_mean():
    dim = _, _, C = (2, 4, 8)
    x = jnp.zeros(dim)
    rms = RMSNorm(C)
    assert jnp.allclose(rms(x), x)
