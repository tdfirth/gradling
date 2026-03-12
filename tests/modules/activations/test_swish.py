from flax import nnx
from jax import numpy as jnp
from jax import random

from gradling.modules.activations.sigmoid import sigmoid
from gradling.modules.activations.swish import swish


def test_swish_values():
    input = jnp.array([-1e10, -1, 0, 1, 1e10])
    expect = jnp.array(
        [0, -sigmoid(jnp.array([-1])).item(), 0, sigmoid(jnp.array([1])).item(), 1e10]
    )
    assert jnp.allclose(swish(input), expect)


def test_swish_known_implementation():
    key = random.key(42)
    for i in range(4):
        key, subkey = random.split(key)
        x = random.normal(subkey, (i + 4,)) * (i + 1)
        want = nnx.swish(x)
        got = swish(x)
        assert jnp.allclose(want, got)
