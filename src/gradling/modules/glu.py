from collections.abc import Callable

import jax
from flax import nnx

from gradling.modules.activations import gelu, relu, swish


class Glu(nnx.Module):
    def __init__(
        self, activation: Callable, d_model: int, d_hidden: int, rngs: nnx.Rngs
    ):
        self.activation = activation
        self.W = nnx.Linear(
            in_features=d_model, out_features=d_hidden, use_bias=False, rngs=rngs
        )
        self.V = nnx.Linear(
            in_features=d_model, out_features=d_hidden, use_bias=False, rngs=rngs
        )
        self.W2 = nnx.Linear(
            in_features=d_hidden, out_features=d_model, use_bias=False, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.activation(self.W(x))
        value = self.V(x)
        return self.W2(gate * value)


class Reglu(Glu):
    def __init__(self, d_model: int, d_hidden: int, rngs: nnx.Rngs):
        super().__init__(activation=relu, d_model=d_model, d_hidden=d_hidden, rngs=rngs)


class Geglu(Glu):
    def __init__(self, d_model: int, d_hidden: int, rngs: nnx.Rngs):
        super().__init__(activation=gelu, d_model=d_model, d_hidden=d_hidden, rngs=rngs)


class Swiglu(Glu):
    def __init__(self, d_model: int, d_hidden: int, rngs: nnx.Rngs):
        super().__init__(
            activation=swish, d_model=d_model, d_hidden=d_hidden, rngs=rngs
        )
