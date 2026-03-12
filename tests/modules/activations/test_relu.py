from jax import numpy as jnp

from gradling.modules.activations.relu import relu


def test_relu():
    x = jnp.array([-1, 0, 1])
    assert jnp.all(relu(x) == jnp.array([0, 0, 1]))

    x = jnp.array([-1e-5, 0, 1e-5])
    assert jnp.all(relu(x) == jnp.array([0, 0, 1e-5]))
