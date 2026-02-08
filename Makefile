.PHONY: help install setup test lint format clean run-example
.PHONY: docker-up docker-down docker-prod docker-prod-down docker-logs docker-health docker-clean

help:
	@echo "Available commands:"
	@echo ""
	@echo "Development:"
	@echo "  make install        - Install dependencies"
	@echo "  make setup          - Setup development environment"
	@echo "  make test           - Run tests"
	@echo "  make test-cov       - Run tests with coverage"
	@echo "  make lint           - Run linters (flake8, mypy)"
	@echo "  make format         - Format code with black"
	@echo "  make clean          - Clean up cache files"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up            - Start API (development mode)"
	@echo "  make docker-down          - Stop API"
	@echo "  make docker-logs          - View container logs"
	@echo "  make docker-health        - Check container health"
	@echo "  make docker-prod          - Start API (production mode)"
	@echo "  make docker-prod-down     - Stop production API"
	@echo "  make docker-clean         - Remove all containers and volumes"
	@echo ""
	@echo "Data:"
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
	python scripts/example_inference.py

# ============================================
# Docker Commands
# ============================================

docker-up:
	@echo "Starting API in development mode..."
	docker-compose up -d
	@echo ""
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"
	@echo ""
	@echo "Run 'make docker-logs' to view logs"
	@echo "Run 'make docker-health' to check health"

docker-down:
	@echo "Stopping API..."
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-health:
	@echo "Container Health Status:"
	@docker-compose ps
	@echo ""
	@echo "Detailed Health:"
	@docker inspect --format='{{.Name}}: {{.State.Health.Status}}' $$(docker-compose ps -q) 2>/dev/null || echo "No health checks configured"

# ============================================
# Production
# ============================================

docker-prod:
	@echo "Starting API in PRODUCTION mode..."
	@echo "⚠️  Make sure .env has production credentials!"
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo ""
	@echo "Production API started"
	@echo "API accessible at: http://localhost:8000 (configure reverse proxy for public access)"

docker-prod-down:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml down

docker-clean:
	@echo "⚠️  This will remove ALL containers, volumes, and cached images!"
	docker-compose down -v
	docker system prune -af --volumes
	@echo "Cleanup complete"

# ============================================
# Utilities
# ============================================

docker-build:
	@echo "Building API image..."
	docker-compose build api

docker-shell:
	docker-compose exec api bash

docker-stats:
	docker stats $$(docker-compose ps -q)

.DEFAULT_GOAL := help
