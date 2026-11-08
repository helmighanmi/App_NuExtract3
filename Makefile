SHELL := /bin/bash
.DEFAULT_GOAL := help
PY := uv run

.PHONY: help install lock api ui test lint fmt typecheck check docker-build up up-gpu down logs vllm sample eval

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies (uv sync)
	uv sync

lock: ## Refresh uv.lock
	uv lock

api: ## Run FastAPI with reload on :8080
	$(PY) uvicorn nuextract_app.api.main:app --host 0.0.0.0 --port 8080 --reload

ui: ## Run Streamlit on :8501
	$(PY) streamlit run apps/streamlit/app.py

test: ## Run tests
	$(PY) pytest

lint: ## Ruff lint
	$(PY) ruff check .

fmt: ## Format code
	$(PY) ruff format .
	$(PY) ruff check --fix .

typecheck: ## Mypy
	$(PY) mypy src

check: lint typecheck test ## CI-equivalent checks

docker-build: ## Build the CPU image
	docker compose build

up: ## Start API + Streamlit (external vLLM or CPU fallback config)
	docker compose up --build -d

up-gpu: ## Start vLLM + API + Streamlit (NVIDIA GPU required)
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d

down: ## Stop the stack
	docker compose down

logs: ## Tail logs
	docker compose logs -f --tail=100

vllm: ## Serve NuExtract3 with vLLM (bare metal)
	./deploy/vllm/serve.sh

sample: ## Generate a synthetic invoice PDF in data/
	$(PY) scripts/make_sample_invoice.py

eval: ## Run evaluation: make eval ARGS="--dataset eval/datasets/example.jsonl"
	$(PY) python eval/run_eval.py $(ARGS)
