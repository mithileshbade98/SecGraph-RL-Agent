#!/bin/bash
# Quick start script for SecGraph-RL Agent with Docker Compose

set -e

echo "=========================================="
echo "  SecGraph-RL Agent - Quick Start"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: docker-compose is not installed"
    echo "Please install docker-compose and try again"
    exit 1
fi

echo "✓ Docker is running"
echo "✓ docker-compose is available"
echo ""

# Clean up any existing containers
echo "Cleaning up existing containers..."
docker-compose down -v 2>/dev/null || true
echo ""

# Build and start services
echo "Building and starting services..."
echo "This will:"
echo "  1. Build the application Docker image"
echo "  2. Download TinyLlama-1.1B model (~2GB, cached for future runs)"
echo "  3. Start Neo4j database"
echo "  4. Generate synthetic security data (50+ anomaly types)"
echo "  5. Load data into Neo4j graph"
echo "  6. Build FAISS vector index"
echo "  7. Start API service (port 8000)"
echo "  8. Start Streamlit UI (port 8501)"
echo ""
echo "This may take 5-10 minutes on first run (includes model download)..."
echo ""

docker-compose up --build -d

echo ""
echo "=========================================="
echo "  Waiting for services to be ready..."
echo "=========================================="
echo ""

# Wait for Neo4j
echo -n "Waiting for Neo4j..."
until curl -s http://localhost:7474 > /dev/null 2>&1; do
    echo -n "."
    sleep 2
done
echo " ✓"

# Wait for data initialization to complete
echo "Waiting for data initialization (this takes a few minutes)..."
last_step=0
last_status=""
while true; do
    logs=$(docker-compose logs --no-color data-init 2>/dev/null | tail -n 50)

    if echo "$logs" | grep -qi "Data initialization complete"; then
        echo "  100% - Data initialization complete"
        break
    fi

    step=$(echo "$logs" | grep -o '\\[[0-9]/5\\]' | tail -n1 | tr -d '[]' | cut -d'/' -f1)
    if [[ -n "$step" && "$step" -gt "$last_step" ]]; then
        last_step=$step
        percent=$(( step * 20 ))
        status_line=$(echo "$logs" | grep "\\[$step/5\\]" | tail -n1)
        status=${status_line#*] }
        printf "  %3d%% - Step %d/5: %s\n" "$percent" "$step" "$status"
    fi

    latest=$(echo "$logs" | tail -n1)
    if [[ -n "$latest" && "$latest" != "$last_status" ]]; then
        latest_clean=${latest#*| }
        echo "      ↳ ${latest_clean}"
        last_status="$latest"
    fi

    if docker-compose ps data-init 2>/dev/null | grep -q "Exit 0"; then
        echo "  Data initialization container exited successfully."
        break
    fi

    if docker-compose ps data-init 2>/dev/null | grep -Eq "Exit [1-9]"; then
        echo "  ⚠️  data-init container exited with a non-zero status. Check logs with 'docker-compose logs data-init'."
        break
    fi

    sleep 5
done
echo " ✓"

# Wait for API
echo -n "Waiting for API service..."
until curl -s http://localhost:8000/healthz > /dev/null 2>&1; do
    echo -n "."
    sleep 2
done
echo " ✓"

# Wait for UI
echo -n "Waiting for Streamlit UI..."
until curl -s http://localhost:8501 > /dev/null 2>&1; do
    echo -n "."
    sleep 2
done
echo " ✓"

echo ""
echo "=========================================="
echo "  🎉 SecGraph-RL Agent is ready!"
echo "=========================================="
echo ""
echo "Access the application:"
echo "  📊 Streamlit UI:  http://localhost:8501"
echo "  🔌 API Docs:      http://localhost:8000/docs"
echo "  🗄️  Neo4j Browser: http://localhost:7474 (user: neo4j, pass: secgraph123)"
echo ""
echo "Quick commands:"
echo "  View logs:        docker-compose logs -f"
echo "  Stop services:    docker-compose down"
echo "  Restart:          docker-compose restart"
echo "  Clean up:         docker-compose down -v"
echo ""
echo "Example API queries:"
echo "  curl http://localhost:8000/healthz"
echo "  curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{\"query\":\"Detect multi-account abuse\"}'"
echo ""
echo "=========================================="
