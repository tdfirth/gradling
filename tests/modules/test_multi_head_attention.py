import jax
import pytest
from flax import nnx
from jax import numpy as jnp
from jax import random

from gradling.modules import MultiHeadAttention


@pytest.fixture
def rngs():
    yield nnx.Rngs(0)


def test_multi_head_attention_dims(subtests, rngs):
    dims = [
        (1, 4, 8),
        (4, 16, 32),
    ]
    num_heads = [1, 2]
    # num_heads = [2]
    for dim in dims:
        for nh in num_heads:
            with subtests.test(f"dims={dim}, num_heads={nh}"):
                B, T, C = dim
                xs = random.normal(rngs(), dim)
                attn = MultiHeadAttention(T, C, nh, 0.1, rngs)
                out = attn(xs)
                assert out.shape == dim


def test_multi_head_attention_future_token_indpendence(subtests, rngs):
    jnp.set_printoptions(linewidth=200, precision=4, suppress=True)
    # If we change some tokens at step t, then the output before t should not
    # be affected.
    nh = 2
    B, T, C = dim = (1, 6, 16)
    attn = MultiHeadAttention(T, C, nh, 0.0, rngs)
    xs1 = random.normal(rngs(), dim)
    xs2 = xs1.copy()
    out1 = attn(xs1)

    for t in range(T):
        with subtests.test(f"t {t + 1}/{T}"):
            # Set dropout to 0 so that output is not randomly affected.
            _xs2 = xs2.at[:, t, :].set(random.normal(rngs(), (B, C)))
            out2 = attn(_xs2)
            # Everything up to t should be identical.
            assert jnp.allclose(out1[:, :t, :], out2[:, :t, :])
            # Everything from t onwards shold have changed.
            assert not jnp.allclose(out1[:, t:, :], out2[:, t:, :])


def test_multi_head_attention_gradient_causality(subtests, rngs):
    nh = 2
    B, T, C = dim = (1, 4, 4)
    attn = MultiHeadAttention(T, C, nh, 0, rngs)
    xs = random.normal(rngs(), dim)
    J = jax.jacobian(lambda x: attn(x))(xs)

    for t in range(T):
        assert jnp.allclose(J[:, t, :, :, t + 1 :, :], 0)
