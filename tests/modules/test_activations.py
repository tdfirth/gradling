from flax import nnx
from jax import lax, random
from jax import numpy as jnp

from gradling.modules.activations import gelu, relu, sigmoid, swish


def test_relu():
    x = jnp.array([-1, 0, 1])
    assert jnp.all(relu(x) == jnp.array([0, 0, 1]))

    x = jnp.array([-1e-5, 0, 1e-5])
    assert jnp.all(relu(x) == jnp.array([0, 0, 1e-5]))


def test_sigmoid_values():
    input = jnp.array([-1e10, 0, 1e10])
    expect = jnp.array([0, 0.5, 1])
    assert jnp.allclose(sigmoid(input), expect)


def test_sigmoid_against_known_implementation(subtests):
    for i in range(6):
        key = random.PRNGKey(i)
        x = random.normal(key, (i, 8))
        want = lax.logistic(x)
        got = sigmoid(x)
        assert jnp.allclose(want, got)


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
