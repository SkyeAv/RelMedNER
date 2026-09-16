.PHONY: test lint fmt

test:
	uv run pytest

lint:
	uv run ruff check ./src ./tests
	uv run ruff format --check ./src ./tests

fmt:
	uv run ruff check --fix ./src ./tests
	uv run ruff format ./src ./tests
