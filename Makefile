.PHONY: test test-fast lint fmt deploy teardown prepush

# the full gate: every test, measured, with a floor so new code cannot land untested.
# 90 leaves headroom under the current 94% while still failing on a regression.
COV_FLAGS := --cov=relmedner --cov-report=term-missing --cov-fail-under=90

# worker fan-out. 4 is the laptop-safe default (matches the pyproject addopts ceiling);
# big hosts pass XDIST=auto: `make test XDIST=auto`. The later -n flag wins over addopts.
XDIST ?= 4

test:
	uv run pytest -n $(XDIST) $(COV_FLAGS)

# iteration loop: parallel, no coverage measurement
test-fast:
	uv run pytest -q -n $(XDIST)

lint:
	uv run ruff check ./src ./tests
	uv run ruff format --check ./src ./tests

# push boundary: static checks + the fast suite, no coverage measurement. The measured
# 90% coverage gate belongs to the wenceslaus full gate (make test / remote-gate.sh cov);
# the laptop never pays for it.
prepush: lint test-fast

fmt:
	uv run ruff check --fix ./src ./tests
	uv run ruff format ./src ./tests

# deploy runs from a checkout ON the head host (wenceslaus) — native docker, no podman socket.
# the laptop only edits and pushes; see plans/flink-remote-cluster.md
deploy:
	uv run relmedner deploy-cluster

teardown:
	uv run relmedner deploy-cluster --teardown
