from dataclasses import dataclass
from functools import reduce
from random import shuffle

from jax import random
from jax import numpy as jnp

from gradling.data import NAMES


class KeyGenerator:
    def __init__(self):
        self.key = random.key(42)

    def one(self):
        key, subkey = random.split(self.key)
        self.key = key
        return subkey

    def many(self, n):
        key, *subkeys = random.split(self.key, n=n)
        self.key = key
        return subkeys


@dataclass
class Hyperparams:
    ctx_length: int
    vocab_size: int
    emb_size: int


@dataclass
class Context:
    params: Hyperparams
    kg: KeyGenerator


@dataclass
class Dataset:
    xs: jnp.array
    ys: jnp.array


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
        vocab = list(set("".join(words)))
        sorted(vocab)
        vocab.insert(0, ".")
        return cls(vocab)


def load_names():
    with open(NAMES, "r") as f:
        names = f.read().splitlines()
        return names


def create_examples(ctx: Context, tok: Tokenizer, words: list[str]):
    l = ctx.params.ctx_length
    examples = []
    for word in words:
        padded = f"{l*'.'}{word}."
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


class Embedding:
    def __init__(self, ctx: Context):
        key = ctx.kg.one()
        self.C = random.normal(key, (ctx.params.vocab_size, ctx.params.emb_size))

    def __call__(self, x):
        self.out = self.C[x]
        return self.out

    def parameters(self):
        return [self.C]


class Flatten:
    def __call__(self, x):
        b, *rest = x.shape
        dim = reduce(lambda a, b: a * b, rest)
        self.out = x.reshape((b, dim))
        return self.out

    def parameters(self):
        return []


class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for p in l.parameters() for l in self.layers]


def main() -> None:
    print("Hello from gradling!")
    names = load_names()
    tok = Tokenizer.from_list(names)
    params = Hyperparams(ctx_length=8, vocab_size=tok.vocab_size, emb_size=24)
    kg = KeyGenerator()
    ctx = Context(params=params, kg=kg)
    xs, ys = create_examples(ctx, tok, names)
    train, dev, test = create_datasets(xs, ys)

    model = Sequential(Embedding(ctx), Flatten())
    print(model(train.xs[0:32]).shape)
