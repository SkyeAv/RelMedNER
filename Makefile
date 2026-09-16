.PHONY: pytest pylint

pytest:
	uv run pytest

pylint:
	uv run ruff format ./src
