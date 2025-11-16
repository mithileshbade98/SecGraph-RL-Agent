# Multi-stage build for SecGraph-RL Agent
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /build

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root

# Final stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /app/data /app/artifacts /app/configs && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories with proper ownership
RUN mkdir -p \
    data/synthetic \
    data/audits \
    artifacts/faiss \
    artifacts/models \
    artifacts/runs \
    models \
    logs && \
    chown -R appuser:appuser \
    data \
    artifacts \
    models \
    logs

# Make CLI scripts and shell scripts executable
RUN chmod +x reason_agent/cli/*.py && \
    chmod +x scripts/*.sh scripts/*.py 2>/dev/null || true

USER appuser

# Download model during build (cached for faster subsequent builds)
RUN python3 scripts/download_model.py || echo "Model download will be attempted at runtime"

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "reason_agent.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
