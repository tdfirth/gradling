import jax
from flax import nnx


class FeedForward(nnx.Module):
    def __init__(self, n_emb: int, dropout: float, rngs: nnx.Rngs):
        self.net = nnx.Sequential(
            nnx.Linear(in_features=n_emb, out_features=4 * n_emb, rngs=rngs),
            nnx.relu,
            nnx.Linear(in_features=4 * n_emb, out_features=n_emb, rngs=rngs),
            nnx.Dropout(dropout, rngs=rngs),
        )

    def __call__(self, x: jax.Array):
        return self.net(x)
