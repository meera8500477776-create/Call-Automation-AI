.PHONY: help install test test-unit test-property test-integration test-e2e test-cov test-fast test-slow coverage lint format type-check clean

help:
	@echo "Available commands:"
	@echo "  make install          - Install project with dev dependencies"
	@echo "  make test             - Run all tests"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-property    - Run property-based tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-e2e         - Run end-to-end tests only"
	@echo "  make test-cov         - Run tests with coverage report"
	@echo "  make test-fast        - Run fast tests (skip slow tests)"
	@echo "  make test-slow        - Run slow tests only"
	@echo "  make coverage         - Generate and open HTML coverage report"
	@echo "  make lint             - Run linter (ruff)"
	@echo "  make format           - Format code with black"
	@echo "  make type-check       - Run type checker (mypy)"
	@echo "  make clean            - Remove generated files"

install:
	pip install -e ".[dev]"

test:
	pytest

test-unit:
	pytest -m unit

test-property:
	pytest -m property

test-integration:
	pytest -m integration

test-e2e:
	pytest -m e2e

test-cov:
	pytest --cov=src --cov-report=term-missing --cov-report=html

test-fast:
	pytest -m "not slow"

test-slow:
	pytest -m slow

coverage:
	pytest --cov=src --cov-report=html
	@echo "Opening coverage report..."
	@open htmlcov/index.html || xdg-open htmlcov/index.html || echo "Please open htmlcov/index.html manually"

lint:
	ruff check src tests

format:
	black src tests

type-check:
	mypy src

clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
