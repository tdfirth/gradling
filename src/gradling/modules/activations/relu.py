import jax


def relu(x: jax.Array) -> jax.Array:
    return x.clip(0)
