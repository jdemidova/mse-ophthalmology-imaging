.PHONY: setup data-check train eval predict docker-build docker-train mlflow-up wandb-login api-up down clean

setup:
	uv sync

data-check:
	uv run python -m seg_project.data.validation --config config/data.yaml

train:
	uv run python -m seg_project.training.train --config config/train.yaml

eval:
	uv run python -m seg_project.training.evaluate --config config/eval.yaml

predict:
	uv run python -m seg_project.inference.predict --config config/infer.yaml

docker-build:
	docker compose build

docker-train:
	docker compose --profile train run --rm trainer

mlflow-up:
	docker compose --profile tracking up -d mlflow

api-up:
	docker compose --profile api up --build api

down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache reports/predictions/*


.PHONY: lint format

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff check --fix src tests scripts
	uv run ruff format src tests scripts

.PHONY: test

test:
	uv run pytest

.PHONY: mlflow-up

mlflow-up:
	docker compose up -d mlflow