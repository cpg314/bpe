from __future__ import annotations
from abc import ABC, abstractmethod
import itertools
import pickle
from typing import Iterator

Token = str
TokenID = int


class AbstractTokenizer(ABC):
    @abstractmethod
    def tokenize(self, line: str) -> list[TokenID]: ...
    def tokenize_text(self, text: str) -> list[TokenID]:
        return list(
            itertools.chain.from_iterable(
                self.tokenize(line) for line in text.split("\n")
            )
        )

    @abstractmethod
    def tokens_as_strings(self, tokens: list[TokenID]) -> list[Token]: ...
    @staticmethod
    @abstractmethod
    def train(lines: Iterator[str], vocab_size: int) -> AbstractTokenizer: ...

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> AbstractTokenizer:
        with open(path, "rb") as f:
            return pickle.load(f)
