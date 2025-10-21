#!/bin/bash
# Docker entrypoint script for SecGraph-RL Agent

set -e

echo "=========================================="
echo "SecGraph-RL Agent - Docker Entrypoint"
echo "=========================================="

# Check which service to run
SERVICE=${SERVICE_NAME:-api}

echo "Service: $SERVICE"
echo "Neo4j URI: ${NEO4J_URI:-not set}"

# Wait for Neo4j if needed
if [ "$SERVICE" != "data-init" ] && [ "$SERVICE" != "skip-wait" ]; then
    echo "Waiting for Neo4j to be ready..."
    until curl -s http://neo4j:7474 > /dev/null 2>&1; do
        echo "  Neo4j not ready yet, waiting..."
        sleep 2
    done
    echo "Neo4j is ready!"
fi

# Execute based on service type
case "$SERVICE" in
    "data-init")
        echo "Running data initialization..."
        exec "$@"
        ;;
    "api")
        echo "Starting API service..."
        exec python -m uvicorn reason_agent.serving.api:app --host 0.0.0.0 --port 8000
        ;;
    "ui")
        echo "Starting Streamlit UI..."
        exec streamlit run reason_agent/ui/app.py --server.port=8501 --server.address=0.0.0.0
        ;;
    *)
        echo "Running custom command: $@"
        exec "$@"
        ;;
esac
