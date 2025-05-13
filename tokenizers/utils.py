import random
from contextlib import contextmanager
from time import perf_counter

import structlog
from matplotlib import colormaps

import tokenizers

log = structlog.get_logger()


def visualize(tokenizer: tokenizers.AbstractTokenizer, text: str) -> str:
    tokens = tokenizer.tokenize(text)
    tokens_chars = tokenizer.tokens_as_strings(tokens)
    cmap = colormaps["Pastel1"]
    html = """
    <style>
        body {
            font-family: Roboto;
        }
        div {
            padding: 2px;
            display: inline-block;
            margin-top: 5px;
            margin-right: 2px;
        }
    </style>
    """
    random.seed(42)
    colors = [[int(col) for col in cmap(i % cmap.N, bytes=True)] for i in range(cmap.N)]
    random.shuffle(colors)
    for i, t in enumerate(tokens_chars):
        color = colors[i % cmap.N]
        html += f"<div style='background-color: rgb({color[0]}, {color[1]}, {color[2]}); '>{t}</div>"

    return html


@contextmanager
def timeit(what: str):
    log.info(f"Starting `{what}`")
    start = perf_counter()
    yield lambda: perf_counter() - start
    log.info(f"Took {round(perf_counter() - start, 2)} s to run `{what}`")
