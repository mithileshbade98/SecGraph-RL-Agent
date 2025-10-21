.PHONY: help setup neo4j-up neo4j-down ingest index run train-dpo train-ppo eval ui api monitor clean test

help:
	@echo "SecGraph-RL Agent - Available targets:"
	@echo "  make setup       - Install dependencies with Poetry"
	@echo "  make neo4j-up    - Start Neo4j ARM64 container"
	@echo "  make neo4j-down  - Stop Neo4j container"
	@echo "  make ingest      - Generate synthetic data and load graph"
	@echo "  make index       - Build FAISS vector index from tool cards"
	@echo "  make run         - Run end-to-end agent query"
	@echo "  make train-dpo   - Train DPO on audit pairs"
	@echo "  make train-ppo   - Train PPO on verifiable math tasks"
	@echo "  make eval        - Run evaluation suite"
	@echo "  make ui          - Launch Streamlit UI"
	@echo "  make api         - Run FastAPI server"
	@echo "  make monitor     - Generate drift monitoring report"
	@echo "  make test        - Run unit tests"
	@echo "  make clean       - Clean artifacts and caches"

setup:
	@echo "Installing dependencies with Poetry..."
	poetry install
	poetry run pre-commit install
	@echo "Setup complete!"

neo4j-up:
	@echo "Starting Neo4j ARM64 container..."
	docker compose -f docker/neo4j-arm64-compose.yml up -d
	@echo "Waiting for Neo4j to be ready..."
	sleep 10
	@echo "Neo4j available at http://localhost:7474"

neo4j-down:
	@echo "Stopping Neo4j container..."
	docker compose -f docker/neo4j-arm64-compose.yml down

ingest:
	@echo "Generating synthetic data..."
	poetry run python -m reason_agent.cli.ingest_cli --config configs/base.yaml
	@echo "Data ingestion complete!"

index:
	@echo "Building FAISS vector index..."
	poetry run python -m reason_agent.cli.index_cli --config configs/embeddings.yaml
	@echo "Index built successfully!"

run:
	@echo "Running agent query..."
	poetry run python -m reason_agent.cli.run_agent_cli --config configs/base.yaml --query "Detect multi-account abuse with shared devices"

train-dpo:
	@echo "Training DPO from audit pairs..."
	poetry run python -m reason_agent.cli.train_rl_cli --config configs/rl/dpo.yaml --mode dpo

train-ppo:
	@echo "Training PPO on verifiable tasks..."
	poetry run python -m reason_agent.cli.train_rl_cli --config configs/rl/ppo.yaml --mode ppo

eval:
	@echo "Running evaluation suite..."
	poetry run python -m reason_agent.cli.evaluate_cli --config configs/eval.yaml

ui:
	@echo "Launching Streamlit UI..."
	poetry run streamlit run reason_agent/ui/app.py

api:
	@echo "Starting FastAPI server..."
	poetry run uvicorn reason_agent.serving.api:app --host 0.0.0.0 --port 8000 --reload

monitor:
	@echo "Generating drift monitoring report..."
	poetry run python -m reason_agent.monitoring.drift --runs-dir artifacts/runs

test:
	@echo "Running unit tests..."
	poetry run pytest

clean:
	@echo "Cleaning artifacts and caches..."
	rm -rf artifacts/faiss/* artifacts/models/* artifacts/runs/*
	rm -rf data/synthetic/*.parquet
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"
