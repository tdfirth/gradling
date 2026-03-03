import json
from pathlib import Path
from typing import Self


class CharacterTokenizer:
    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self._encoder = {c: i for i, c in enumerate(vocab)}
        self._decoder = {i: c for i, c in enumerate(vocab)}

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, path: Path) -> Self:
        with open(path) as f:
            vocab = json.load(f)
            return cls(vocab)

    @classmethod
    def train(cls, corpus: str) -> Self:
        vocab = sorted(list(set(corpus)))
        return cls(vocab)

    def encode(self, input: str) -> list[int]:
        return [self._encoder[c] for c in input]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self._decoder[t] for t in tokens)
