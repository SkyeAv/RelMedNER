.PHONY: pytest pylint scala-test scala-lint scala-fmt

pytest:
	uv run pytest

pylint:
	uv run ruff format ./src

scala-test:
	cd ./scala && mill test

scala-lint:
	cd ./scala && mill __.checkFormat && mill compile

scala-fmt:
	cd ./scala && mill __.reformat
