.PHONY: pytest pylint scala-test scala-lint scala-fmt

test:
	uv run pytest

lint:
	uv run ruff format ./src
