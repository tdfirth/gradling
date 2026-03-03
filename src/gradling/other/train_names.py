import functools as ft
import itertools as it
from collections.abc import Iterator
from dataclasses import dataclass
from functools import reduce
from random import shuffle

import jax
from jax import numpy as jnp
from jax import random

from gradling.data import NAMES


def make_rng(seed=42):
    root_key = random.key(seed)
    return map(ft.partial(random.fold_in, root_key), it.count())


@jax.tree_util.register_pytree_with_keys_class
class dot_dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

    def tree_flatten_with_keys(self):
        keys = tuple(sorted(self))
        return tuple((jax.tree_util.DictKey(k), self[k]) for k in keys), keys

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(zip(keys, values))


@dataclass
class Hyperparams:
    rng_seed: int
    ctx_length: int
    vocab_size: int
    emb_size: int
    hidden_size: int


@dataclass
class Dataset:
    xs: jax.Array
    ys: jax.Array


class Tokenizer:
    def __init__(self, vocabulary: list[str]):
        self.vocabulary = vocabulary
        self.itos = {i: s for i, s in enumerate(vocabulary)}
        self.stoi = {s: i for i, s in enumerate(vocabulary)}

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode_one(self, s: str):
        return self.stoi[s]

    def encode(self, string: list[str]):
        return [self.stoi[s] for s in string]

    def decode_one(self, i: int):
        return self.itos[i]

    def decode(self, ints: list[int]):
        return [self.itos[n] for n in ints]

    @classmethod
    def from_list(cls, words):
        vocab = sorted(list(set("".join(words))))
        vocab.insert(0, ".")
        return cls(vocab)


def load_names():
    with open(NAMES) as f:
        names = f.read().splitlines()
        return names


def create_examples(params: Hyperparams, tok: Tokenizer, words: list[str]):
    l = params.ctx_length
    examples = []
    for word in words:
        padded = f"{l * '.'}{word}."
        for i in range(len(word) + 1):
            x = padded[i : i + l]
            y = padded[i + l]
            examples.append([tok.encode(list(x)), tok.encode_one(y)])

    shuffle(examples)
    xs, ys = [], []
    for x, y in examples:
        xs.append(x)
        ys.append(y)
    return jnp.array(xs), jnp.array(ys)


def create_datasets(xs, ys):
    n = len(xs)
    n_80 = int(0.8 * n)
    n_90 = int(0.9 * n)

    x_train, y_train = xs[:n_80], ys[:n_80]
    x_dev, y_dev = xs[n_80:n_90], ys[n_80:n_90]
    x_test, y_test = xs[n_90:], ys[n_90:]
    return (
        Dataset(xs=x_train, ys=y_train),
        Dataset(xs=x_dev, ys=y_dev),
        Dataset(xs=x_test, ys=y_test),
    )


def cross_entropy_loss(logits: jax.Array, targets: jax.Array):
    assert len(logits) == len(targets)
    shifted = logits - logits.max(1, keepdims=True)
    log_probs = shifted - jnp.log(jnp.exp(shifted).sum(1, keepdims=True))
    return -log_probs[jnp.arange(len(targets)), targets].mean()


def linear(rng: Iterator, fan_in: int, fan_out: int):
    return dot_dict(
        W=random.normal(next(rng), (fan_in, fan_out)) / (5 * fan_in**0.5 / 3),
        b=jnp.zeros((fan_out,)),
        bnorm=dot_dict(
            gamma=random.normal(next(rng), (fan_out,)), beta=jnp.zeros((fan_out,))
        ),
    )


def init_weights(params: Hyperparams, rng):
    return dot_dict(
        emb=random.normal(next(rng), (params.vocab_size, params.emb_size)) * 0.01,
        hidden=[
            linear(rng, params.ctx_length * params.emb_size, params.hidden_size),
            linear(rng, params.hidden_size, params.hidden_size),
        ],
        output=random.normal(next(rng), (params.hidden_size, params.vocab_size))
        / (params.hidden_size**0.5),
    )


def bnorm_state(size: int):
    return dot_dict(
        running_mean=jnp.zeros((size,)),
        running_var=jnp.ones((size,)),
    )


def init_state(params: Hyperparams):
    return dot_dict(
        hidden=[
            bnorm_state(params.hidden_size),
            bnorm_state(params.hidden_size),
        ]
    )


def update(current, new):
    mom = 0.1
    return ((1 - mom) * current) + (mom * new)


def model(weights, state, x, eval=False):
    # New state
    new_bnstate = []
    # Look up positional embeddings and flatten
    x = weights.emb[x]
    b, *rest = x.shape
    x = jnp.reshape(x, (b, reduce(lambda a, b: a * b, rest)))
    # For each hidden layer
    for i, layer in enumerate(weights.hidden):
        # Apply the weights and bias
        x = x @ layer.W + layer.b
        # Batch norm
        bnorm_state = state.hidden[i]
        if eval:
            xmean = bnorm_state.running_mean
            xvar = bnorm_state.running_var
        else:
            xmean = jnp.mean(x, 0)
            xvar = jnp.var(x, 0)
            new_bnstate.append(
                dot_dict(
                    running_mean=update(bnorm_state.running_mean, xmean),
                    running_var=update(bnorm_state.running_var, xvar),
                )
            )
        # Normalize x
        x = (x - xmean) / (xvar + 1e-5) ** 0.5
        x = x * layer.bnorm.gamma + layer.bnorm.beta
        # Apply the nonlinearity.
        x = jax.lax.tanh(x)

    logits = x @ weights.output
    return logits, dot_dict(hidden=new_bnstate)


@jax.jit
def train_step(
    weights,
    state,
    x,
    y,
):
    def loss(W):
        logits, new_state = model(W, state, x)
        return cross_entropy_loss(logits, y), new_state

    return jax.value_and_grad(loss, has_aux=True)(weights)


@jax.jit
def val_loss(weights, state, x, y):
    logits, _ = model(weights, state, x, eval=True)
    return cross_entropy_loss(logits, y)


@jax.jit
def sample_one(key, weights, state, ctx):
    logits, _ = model(weights, state, ctx, eval=True)
    return random.categorical(key, logits)


def sample(rng, params: Hyperparams, tok: Tokenizer, weights, state, n=5):
    for _ in range(n):
        context = ["."] * params.ctx_length
        result = []
        while True:
            choice = sample_one(
                next(rng), weights, state, jnp.array([tok.encode(context)])
            ).item()
            next_token = tok.decode_one(choice)
            if next_token == ".":
                break
            result.append(next_token)
            context.append(next_token)
            context = context[1:]
        print("".join(result))


def main() -> None:
    names = load_names()
    tok = Tokenizer.from_list(names)
    params = Hyperparams(
        rng_seed=42,
        ctx_length=8,
        vocab_size=tok.vocab_size,
        emb_size=24,
        hidden_size=200,
    )
    rng = make_rng(params.rng_seed)
    xs, ys = create_examples(params, tok, names)
    train, dev, test = create_datasets(xs, ys)
    weights, state = init_weights(params, rng), init_state(params)
    steps = 50_000
    batch_size = 32
    for step in range(steps):
        mask = random.randint(
            next(rng), (batch_size,), minval=0, maxval=(len(train.xs) - 1)
        )
        X, Y = train.xs[mask], train.ys[mask]
        (loss, new_state), grad = train_step(
            weights,
            state,
            X,
            Y,
        )
        state = new_state
        lr = 0.1 if step <= 40000 else 0.01
        weights = jax.tree.map(lambda w, g: w + -lr * g, weights, grad)

        if step % 100 == 0:
            vloss = val_loss(weights, state, dev.xs, dev.ys)
            print(f"{step}/{steps}: loss = {vloss:.4f}")
        # print(jax.tree_util.tree_structure(grad))

    sample(rng, params, tok, weights, state, n=50)
