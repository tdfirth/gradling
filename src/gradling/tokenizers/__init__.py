from typing import Protocol

from gradling.tokenizers.character import CharacterTokenizer

__all__ = ["CharacterTokenizer", "Tokenizer"]


class Tokenizer(Protocol):
    def encode(self, input: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...
