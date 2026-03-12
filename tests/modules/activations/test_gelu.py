from flax import nnx
from jax import numpy as jnp
from jax import random

from gradling.modules.activations.gelu import gelu


def test_gelu_values_approximate():
    input = jnp.array([-1e10, 0, 1e10])
    expect = jnp.array([0, 0, 1e10])
    assert jnp.allclose(gelu(input), expect)


def test_gelu_values_precise():
    input = jnp.array([-1e6, 0, 1e6])
    expect = jnp.array([0, 0, 1e6])
    assert jnp.allclose(gelu(input, False), expect)


def test_gelu_known_implementation_approximate():
    key = random.key(42)

    for i in range(4):
        key, subkey = random.split(key)
        x = random.normal(subkey, (i + 4,)) * (i + 1)
        want = nnx.gelu(x)
        got = gelu(x)
        assert jnp.allclose(want, got)


def test_gelu_known_implementation_precise():
    key = random.key(42)

    for i in range(4):
        key, subkey = random.split(key)
        x = random.normal(subkey, (i + 4,)) * (i + 1)
        want = nnx.gelu(x, False)
        got = gelu(x, False)
        assert jnp.allclose(want, got)
