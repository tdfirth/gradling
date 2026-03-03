.PHONY: all check format fix ty test clean

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
	uv run pytest tests

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache .ty_cache build dist .coverage htmlcov .hypothesis
