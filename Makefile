.PHONY: test lint

test:
	uv run pytest

lint:
	uv run ruff format ./src
