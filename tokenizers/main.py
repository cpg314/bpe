import argparse
import itertools
import logging

import structlog
from tqdm import tqdm

import tokenizers.bpe as bpe_py
from tokenizers.utils import timeit, visualize

from .tokenizers import Tokenizer as TokenizerRs

log = structlog.get_logger()


if __name__ == "__main__":
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO)
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=300)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--tokenizer", choices=["py", "rs"], required=True)
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()

    Tokenizer = bpe_py.Tokenizer if args.tokenizer == "py" else TokenizerRs

    filename = f"tokenizer-{args.vocab_size}.tokenizer"
    try:
        if args.no_cache:
            raise FileNotFoundError()
        tokenizer = Tokenizer.load(filename)
        log.warn("Loading pretrained tokenizer", filename=filename)
    except FileNotFoundError:
        with timeit("train"):
            iterator = itertools.chain.from_iterable(
                open(filename) for filename in args.corpus
            )
            tokenizer = Tokenizer.train(iterator, vocab_size=args.vocab_size)
        tokenizer.save(filename)
    log.info(tokenizer)

    with timeit("tokenize") as t:
        total_tokens = 0
        with open("books/11.txt") as f:
            if args.parallel:
                total_tokens = len(tokenizer.tokenize_text(f.read()))
            else:
                for line in tqdm(f):
                    total_tokens += len(tokenizer.tokenize(line))
        log.info(
            f"Total tokens: {total_tokens}, {round(total_tokens / t(), 2)} tokens/s"
        )

    text = "Deep into that darkness peering, long I stood there wondering, fearing, Doubting, dreaming dreams no mortal ever dared to dream before"
    text = "".join(s for s in text if s.isalpha() or s.isspace())
    with open("out.html", "w") as f:
        f.write(visualize(tokenizer, text))
