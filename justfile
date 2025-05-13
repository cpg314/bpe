run:
    uv run python -m tokenizers.main --vocab-size 5000 --corpus books/train/* --tokenizer py --tokenize-doc books/11.txt 
check:
    ruff check
    ruff format --check
    uv run pyright
    cargo fmt --check 
    cargo group-imports
    cargo clippy
    cargo machete
