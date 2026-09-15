.PHONY: test lint typecheck

test:
	uv run pytest

lint:
	uv run ruff format ./src

typecheck:
	uv run pyright
