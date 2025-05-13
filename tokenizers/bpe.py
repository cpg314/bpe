"""
Heavily inspired from https://huggingface.co/learn/llm-course/en/chapter6/5
"""

from collections import Counter
from time import perf_counter
from dataclasses import dataclass
from typing import Iterator

import structlog
from bidict import bidict
from tqdm import tqdm

from tokenizers import AbstractTokenizer, TokenID, Token

log = structlog.get_logger()

Word = str

Splits = dict[Word, list[str]]
Pair = tuple[str, str]
Merges = dict[Pair, str]

APPROXIMATE_VOCAB_SIZE = False


def compute_pair_freqs(splits: Splits, freqs: Counter) -> Counter:
    pair_freqs = Counter()
    for word, freq in freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def get_words(line: str) -> list[str]:
    line = line.strip()
    line = "".join(e for e in line.lower() if e.isalnum() or e.isspace())
    return [w for w in line.split(" ") if len(w) > 0]


@dataclass
class Tokenizer(AbstractTokenizer):
    merges: Merges
    tokens: bidict[int, str]

    def __str__(self):
        return f"Tokenizer with {len(self.tokens)} tokens and {len(self.merges)} merge rules"

    def tokenize(self, line: str) -> list[TokenID]:
        words = get_words(line)
        if len(words) == 0:
            return []
        splits: list[list[str]] = list(map(list, words))
        # Apply the merge rules in order
        for pair, merge in self.merges.items():
            for split in splits:
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split[i] = merge
                        del split[i + 1]
                    else:
                        i += 1

        splits = [s + [" "] for s in splits]
        splits[-1].pop()
        return [
            self.tokens.inverse.get(letter, -1) for split in splits for letter in split
        ]

    def tokens_as_strings(self, tokens: list[TokenID]) -> list[Token]:
        return [self.tokens.get(token, "UNK") for token in tokens]

    @staticmethod
    def train(lines: Iterator[str], **kwargs) -> AbstractTokenizer:  # pyright: ignore
        vocab_size = kwargs["vocab_size"]
        log.info("Training tokenizer")
        log.info("Processing data")
        # Compute word frequencies
        freqs = Counter()
        for line in tqdm(lines):
            freqs.update(get_words(line))

        # Initial vocabulary and words splits
        vocab = set([letter for word in freqs for letter in word])
        splits = {word: [c for c in word] for word in freqs}
        log.info("Done processing data, computing merges")

        # Record list of merges to build the tokenizer
        merges: Merges = {}

        start = perf_counter()
        # stats = open("stats.csv", "w")
        stats = None
        progress = tqdm()
        while len(vocab) < vocab_size:
            progress.update()
            progress.set_description(f"Vocab size: {len(vocab)}")
            pair_freqs = compute_pair_freqs(splits, freqs)
            if not pair_freqs:
                break

            most_common_pair, _ = pair_freqs.most_common(1)[0]
            merges[most_common_pair] = "".join(most_common_pair)
            log.debug(f"Merging {most_common_pair}")
            a, b = most_common_pair

            joined = "".join(most_common_pair)

            # Update splits
            for word, split in splits.items():
                if len(split) == 1:
                    continue

                i = 0
                while i < len(split) - 1:
                    if split[i] == a and split[i + 1] == b:
                        split[i : i + 2] = [joined]
                    else:
                        i += 1
                splits[word] = split

            # Could also skip this and leave unused tokens in the vocab, and instead just add `joined`
            if not APPROXIMATE_VOCAB_SIZE:
                vocab = set()
                for split in splits.values():
                    vocab.update(split)
            else:
                vocab.add(joined)

            log.debug(f"Vocabulary: {vocab}")
            if stats is not None and len(vocab) % 100 == 0:
                stats.write(f"{len(vocab)},{perf_counter() - start},{len(merges)}\n")
        if stats is not None:
            stats.close()

        vocab.add(" ")
        tokens = sorted(vocab)
        progress.close()
        log.info(f"Stopped after {progress.n} iterations, with {len(tokens)} tokens")
        return Tokenizer(merges, tokens=bidict(enumerate(tokens)))
