from jax import lax, random
from jax import numpy as jnp

from gradling.modules.activations.sigmoid import sigmoid


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
