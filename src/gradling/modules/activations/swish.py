import jax

from gradling.modules.activations.sigmoid import sigmoid


def swish(x: jax.Array) -> jax.Array:
    return x * sigmoid(x)
