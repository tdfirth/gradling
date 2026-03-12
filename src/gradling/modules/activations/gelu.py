import math

import jax
from jax import lax

_SQRT_2 = math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def gelu(x: jax.Array, approximate: bool = True) -> jax.Array:
    if approximate:
        return x * 0.5 * (1 + lax.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x**3)))
    else:
        # 1 + erf(z) = erfc(-z)
        cdf = 0.5 * lax.erfc(-x / _SQRT_2)
        return x * cdf
