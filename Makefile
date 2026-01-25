.PHONY: help install setup test lint format clean run-example

help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make setup          - Setup development environment"
	@echo "  make test           - Run tests"
	@echo "  make test-cov       - Run tests with coverage"
	@echo "  make lint           - Run linters (flake8, mypy)"
	@echo "  make format         - Format code with black"
	@echo "  make clean          - Clean up cache files"
	@echo "  make prepare-data   - Prepare and upload dataset"
	@echo "  make run-example    - Run inference example"

install:
	pip install -r requirements.txt

setup: install
	@echo "Development environment setup complete!"
	@echo "Next steps:"
	@echo "  1. Copy .env.example to .env"
	@echo "  2. Fill in your AWS credentials"
	@echo "  3. Run: make prepare-data"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	flake8 src/ scripts/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ scripts/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov/
	rm -rf .coverage

prepare-data:
	python scripts/prepare_dataset.py --train-size 1000 --val-size 150

run-example:
	python scripts/example_inference.py --model-id google/flan-t5-base

.DEFAULT_GOAL := help
