.PHONY: up down logs install test lint run-llm run-vlm run-embedding sweep-llm

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

sweep-llm:
	python scripts/sweep.py --config configs/llm_lora.yaml

run-pii:
	python -m mlx_pipeline.run --config configs/pii_masking.yaml

sweep-pii:
	python scripts/sweep.py --config configs/pii_masking.yaml

pii-data:
	python scripts/generate_pii_dataset.py --n 3000 --seed 7 --out data/samples/korean_pii.jsonl
