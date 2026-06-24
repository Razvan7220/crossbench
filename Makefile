.PHONY: install dev test lint fmt docker-build docker-push run

install:
	pip install -e .

dev:
	pip install -e ".[all]"

test:
	pytest --cov=cross_bench --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/

fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/

docker-build:
	bash scripts/build_docker_images.sh

run:
	cross-bench run $(ARGS)
