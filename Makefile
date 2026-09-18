.PHONY: test lint fmt deploy teardown

test:
	uv run pytest

lint:
	uv run ruff check ./src ./tests
	uv run ruff format --check ./src ./tests

fmt:
	uv run ruff check --fix ./src ./tests
	uv run ruff format ./src ./tests

deploy:
	DOCKER_HOST=unix:///run/user/$$(id -u)/podman/podman.sock uv run relmedner deploy-cluster

teardown:
	DOCKER_HOST=unix:///run/user/$$(id -u)/podman/podman.sock uv run relmedner deploy-cluster --teardown
