import pytest
from flax import nnx
from jax import numpy as jnp
from jax import random

from gradling.modules import LayerNorm


@pytest.fixture()
def rngs():
    yield nnx.Rngs(0)


def test_dims(subtests, rngs):
    dims = [
        (16, 32),
        (32, 64),
        (32, 128),
        (16, 8, 32),
        (32, 8, 64),
        (32, 8, 128),
    ]
    for dim in dims:
        with subtests.test(msg=f"{dim}"):
            C = dim[-1]
            ln = LayerNorm(C, rngs)
            xs = random.normal(rngs(), dim)
            out = ln(xs)

            assert out.shape == dim


def test_unit_variance(subtests, rngs):
    dim = (32, 8, 128)
    for i in range(6):
        with subtests.test(msg=f"example {i}"):
            C = dim[-1]
            ln = LayerNorm(C, rngs)
            xs = random.normal(rngs(), dim)
            out = ln(xs)
            var = jnp.var(out, axis=-1)
            assert var.shape == (32, 8)
            assert jnp.allclose(var, 1)


def test_zero_mean(subtests, rngs):
    dim = (32, 8, 128)
    for i in range(6):
        with subtests.test(msg=f"example {i}"):
            C = dim[-1]
            ln = LayerNorm(C, rngs)
            xs = random.normal(rngs(), dim)
            out = ln(xs)
            mean = jnp.mean(out, axis=-1)
            assert mean.shape == (32, 8)
            assert jnp.allclose(mean, 0, atol=1e-7)


def test_gamma_and_beta(subtests, rngs):
    dim = (32, 8, 128)
    for i in range(6):
        with subtests.test(msg=f"example {i}"):
            C = dim[-1]
            ln = LayerNorm(C, rngs)
            ln.gamma = jnp.ones((128,)) * i
            ln.beta = jnp.ones((128,)) * i
            xs = random.normal(rngs(), dim)
            out = ln(xs)

            # Mean should scale linearly with beta
            mean = jnp.mean(out, axis=-1)
            assert mean.shape == (32, 8)
            assert jnp.allclose(mean, i, atol=1e-7)

            # Variance scales with the square of gamma
            var = jnp.var(out, axis=-1)
            assert var.shape == (32, 8)
            assert jnp.allclose(var, i**2, atol=1e-7)


def test_handles_zero_var_and_mean(rngs):
    dim = (32, 8, C := 128)
    ln = LayerNorm(C, rngs)
    xs = jnp.zeros(dim)
    out = ln(xs)
    assert out.shape == dim
