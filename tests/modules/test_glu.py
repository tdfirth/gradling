from flax import nnx
from jax import numpy as jnp

from gradling.modules.activations import gelu, swish
from gradling.modules.glu import Geglu, Reglu, Swiglu


def test_reglu():
    rngs = nnx.Rngs(42)
    dm, dh = (8, 12)
    reglu = Reglu(d_model=dm, d_hidden=dh, rngs=rngs)
    reglu.W.kernel = jnp.ones((dm, dh))
    reglu.V.kernel = jnp.ones((dm, dh))
    reglu.W2.kernel = jnp.ones((dh, dm))

    x_dim = (2, 4, 8)
    x = jnp.ones(x_dim)
    assert reglu(x).shape == x_dim
    assert jnp.allclose(reglu(jnp.zeros(x_dim)), jnp.zeros(x_dim))
    assert jnp.allclose(reglu(jnp.full(x_dim, -1)), jnp.zeros(x_dim))
    # The 768 is surprisng but correct. When the matrix multiplications on the
    # arrays of ones you get 8x8x12 = 768
    assert jnp.allclose(reglu(x), jnp.full(x_dim, 768))

    # Test the gate works accurately.
    x = x.at[0].set(-1)
    assert jnp.allclose(reglu(x), jnp.full(x_dim, 768).at[0].set(0))


def test_geglu():
    rngs = nnx.Rngs(42)
    dm, dh = (8, 8)
    geglu = Geglu(d_model=dm, d_hidden=dh, rngs=rngs)
    # Using identity for this one as it's easier to reason about expected
    # outputs.
    geglu.W.kernel = jnp.identity(dh)
    geglu.V.kernel = jnp.identity(dh)
    geglu.W2.kernel = jnp.identity(dh)

    x_dim = (2, 4, 8)
    x = jnp.ones(x_dim)
    assert geglu(x).shape == x_dim
    assert jnp.allclose(geglu(jnp.zeros(x_dim)), jnp.zeros(x_dim))
    assert jnp.allclose(geglu(jnp.full(x_dim, -1e10)), jnp.zeros(x_dim))

    # Test the gate works accurately.
    x = x.at[0].set(-1e10)
    assert jnp.allclose(geglu(x), jnp.full(x_dim, gelu(jnp.array([1]))).at[0].set(0))


def test_swiglu():
    rngs = nnx.Rngs(42)
    dm, dh = (8, 8)
    swiglu = Swiglu(d_model=dm, d_hidden=dh, rngs=rngs)
    swiglu.W.kernel = jnp.identity(dh)
    swiglu.V.kernel = jnp.identity(dh)
    swiglu.W2.kernel = jnp.identity(dh)

    x_dim = (2, 4, 8)
    x = jnp.ones(x_dim)
    assert swiglu(x).shape == x_dim
    assert jnp.allclose(swiglu(jnp.zeros(x_dim)), jnp.zeros(x_dim))
    assert jnp.allclose(swiglu(jnp.full(x_dim, -1e10)), jnp.zeros(x_dim))

    # Test the gate works accurately.
    x = x.at[0].set(-1e10)
    assert jnp.allclose(swiglu(x), jnp.full(x_dim, swish(jnp.array([1]))).at[0].set(0))
