.PHONY: up down logs install test lint run-llm run-vlm run-embedding

COMPOSE = docker compose -f docker-compose.local.yml --env-file .env.local

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

install:
	pip install -e ".[dev]"

test:
	pytest -v

lint:
	ruff check src tests

run-llm:
	python -m mlx_pipeline.run --config configs/llm_lora.yaml

run-vlm:
	python -m mlx_pipeline.run --config configs/vlm_lora.yaml

run-embedding:
	python -m mlx_pipeline.run --config configs/embedding_sft.yaml
