.PHONY: all check format fix ty test

all: check

check:
	@uv run ruff check --output-format concise --exclude notebooks .; uv run ty check --output-format concise --exclude notebooks

format:
	uv run ruff format .

fix:
	uv run ruff check . --fix

ty:
	uv run ty check

test:
	uv run pytest
