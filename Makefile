.PHONY: test test-fast lint fmt deploy teardown

# the full gate: every test, measured, with a floor so new code cannot land untested.
# 90 leaves headroom under the current 94% while still failing on a regression.
COV_FLAGS := --cov=relmedner --cov-report=term-missing --cov-fail-under=90

test:
	uv run pytest $(COV_FLAGS)

# iteration loop: parallel, no coverage measurement
test-fast:
	uv run pytest -q

lint:
	uv run ruff check ./src ./tests
	uv run ruff format --check ./src ./tests

fmt:
	uv run ruff check --fix ./src ./tests
	uv run ruff format ./src ./tests

# deploy runs from a checkout ON the head host (wenceslaus) — native docker, no podman socket.
# the laptop only edits and pushes; see plans/flink-remote-cluster.md
deploy:
	uv run relmedner deploy-cluster

teardown:
	uv run relmedner deploy-cluster --teardown
