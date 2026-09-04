.DEFAULT_GOAL := help
SHELL := /bin/bash
PY ?= python3

.PHONY: help install install-dev init-db ingest train pipeline predict api web \
        demo test lint fmt clean docker-up docker-down docker-etl

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	$(PY) -m pip install -r requirements.txt

install-dev: ## Install runtime + dev dependencies
	$(PY) -m pip install -r requirements-dev.txt
	cd frontend && npm install

init-db: ## Create tables
	$(PY) -m stockseer.cli init-db

ingest: ## Pull prices + news for the configured universe
	$(PY) -m stockseer.cli ingest

train: ## Train the direction model with walk-forward validation
	$(PY) -m stockseer.cli train

pipeline: ## Ingest -> train -> predict, end to end
	$(PY) -m stockseer.cli pipeline

predict: ## Score every ticker in the warehouse
	$(PY) -m stockseer.cli predict

api: ## Run the API with reload on :8000
	$(PY) -m stockseer.cli serve --reload --port 8000

web: ## Run the dashboard dev server on :5173
	cd frontend && npm run dev

demo: ## Full offline demo — no network, no API keys, no database server
	DATA_SOURCE=synthetic NEWS_SOURCE=synthetic $(PY) -m stockseer.cli pipeline
	@echo "Now run 'make api' and 'make web'."

test: ## Run the test suite
	$(PY) -m pytest tests -q

lint: ## Ruff
	$(PY) -m ruff check stockseer tests

fmt: ## Ruff autofix
	$(PY) -m ruff check --fix stockseer tests

clean: ## Remove caches, artifacts and the dev database
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov artifacts/models artifacts/*.db
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

docker-up: ## Postgres + API + dashboard (dashboard on :8080)
	docker compose up --build -d

docker-etl: ## Run ingest+train inside the compose network
	docker compose --profile jobs run --rm etl

docker-down: ## Stop everything (keeps volumes)
	docker compose down
