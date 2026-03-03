import jax
import pytest
from flax import nnx
from jax import numpy as jnp
from jax import random

from gradling.modules import MultiHeadAttention, SingleHeadAttention


@pytest.fixture
def rngs():
    yield nnx.Rngs(0)


def test_single_head_attention_dims(subtests, rngs):
    dims = [
        (1, 4, 8),
        (2, 6, 16),
    ]
    for dim in dims:
        with subtests.test(f"dims={dim}"):
            B, T, C = dim
            attn = SingleHeadAttention(T, C, 0.1, rngs)
            xs = random.normal(rngs(), dim)
            assert attn(xs).shape == dim


def test_single_head_attention_future_token_indpendence(subtests, rngs):
    jnp.set_printoptions(linewidth=200, precision=4, suppress=True)
    # If we change some tokens at step t, then the output before t should not
    # be affected.
    B, T, C = dim = (1, 6, 16)
    attn = SingleHeadAttention(T, C, 0.0, rngs)
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


def test_single_head_attention_gradient_causality(subtests, rngs):
    B, T, C = dim = (1, 4, 4)
    attn = SingleHeadAttention(T, C, 0, rngs)
    xs = random.normal(rngs(), dim)
    J = jax.jacobian(lambda x: attn(x))(xs)

    for t in range(T):
        assert jnp.allclose(J[:, t, :, :, t + 1 :, :], 0)


def test_multi_head_attention_dims(subtests, rngs):
    dims = [
        (1, 4, 8),
        (2, 6, 16),
    ]
    num_heads = [1, 2]
    for dim in dims:
        for nh in num_heads:
            with subtests.test(f"dims={dim}, num_heads={nh}"):
                B, T, C = dim
                xs = random.normal(rngs(), dim)
                attn = MultiHeadAttention(T, C, nh, 0.1, rngs)
                out = attn(xs)
                assert out.shape == dim


# def test_multi_head_attention_rejects_invalid_head_size_num_head_combination(rngs):
#    with pytest.raises(AssertionError):
#        attn = MultiHeadAttention(8, 4, 8, 5, 0.1, rngs)


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
