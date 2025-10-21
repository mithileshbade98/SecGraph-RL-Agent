# Docker Deployment Guide

This guide covers running SecGraph-RL Agent using Docker Compose for one-command deployment.

## 🚀 Quick Start (One Command)

```bash
./start.sh
```

**That's it!** This single script will:
1. ✅ Build Docker images
2. ✅ Start Neo4j database
3. ✅ Generate synthetic security data (50+ anomaly types)
4. ✅ Load data into Neo4j graph
5. ✅ Build FAISS vector index
6. ✅ Start API service (port 8000)
7. ✅ Start Streamlit UI (port 8501)

**First run takes 5-10 minutes** for data generation and model downloads.

---

## 📋 Prerequisites

- **Docker Desktop** (or Docker Engine + Docker Compose)
  - Mac: [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
  - Windows: [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
  - Linux: [Docker Engine](https://docs.docker.com/engine/install/)

- **Minimum Requirements:**
  - 4 GB RAM (8 GB recommended)
  - 10 GB disk space
  - Internet connection (for initial image pull and model downloads)

---

## 🏗️ Architecture

The Docker Compose setup includes 4 services:

```
┌─────────────────────────────────────────┐
│  secgraph-ui (Streamlit)                │
│  Port: 8501                             │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│  secgraph-api (FastAPI)                 │
│  Port: 8000                             │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│  secgraph-neo4j (Graph DB)              │
│  Ports: 7474 (HTTP), 7687 (Bolt)        │
└─────────────────────────────────────────┘

  secgraph-data-init (One-time setup)
  - Generates synthetic data
  - Loads Neo4j graph
  - Builds FAISS index
  - Creates sample audit pairs
```

---

## 📦 Services Detail

### 1. **neo4j** (Graph Database)
- **Image**: `neo4j:5.14-community`
- **Ports**:
  - `7474` - Neo4j Browser (Web UI)
  - `7687` - Bolt protocol
- **Credentials**:
  - Username: `neo4j`
  - Password: `secgraph123`
- **Data**: Persisted in Docker volume `neo4j_data`

### 2. **data-init** (Initialization Service)
- **Purpose**: One-time data setup
- **Actions**:
  1. Waits for Neo4j to be ready
  2. Generates ~2000+ synthetic security events
  3. Loads events into Neo4j as temporal graph
  4. Extracts temporal features (burstiness, velocity, etc.)
  5. Builds FAISS vector index for tool retrieval
  6. Creates 10 sample preference pairs for DPO training
- **Status**: Exits after completion (restart: no)
- **Marker**: Creates `/app/data/.init_complete` when done

### 3. **api** (FastAPI Service)
- **Purpose**: REST API for agent queries
- **Port**: `8000`
- **Endpoints**:
  - `GET /healthz` - Health check
  - `POST /query` - Run agent query
  - `GET /trace/{trace_id}` - Get reasoning trace
  - `GET /stats` - System statistics
  - `GET /docs` - Swagger UI
- **Dependencies**: Waits for `data-init` to complete

### 4. **ui** (Streamlit Dashboard)
- **Purpose**: Interactive web UI
- **Port**: `8501`
- **Features**:
  - Query tab - Run agent queries
  - Graph tab - Inspect temporal graph
  - RL Training tab - Train PPO/DPO
  - Audits tab - Create preference pairs
  - Drift tab - Monitor performance
- **Dependencies**: Waits for `api` to be healthy

---

## 🎯 Usage

### Start Everything

```bash
# Option 1: Use the quick start script
./start.sh

# Option 2: Manual docker-compose
docker-compose up --build
```

### Access Services

Once running:

| Service | URL | Credentials |
|---------|-----|-------------|
| **Streamlit UI** | http://localhost:8501 | None |
| **API Docs** | http://localhost:8000/docs | None |
| **Neo4j Browser** | http://localhost:7474 | neo4j / secgraph123 |

### Example API Calls

```bash
# Health check
curl http://localhost:8000/healthz

# Run agent query
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Detect multi-account abuse with shared devices",
    "max_steps": 10,
    "enable_verification": true
  }'

# Get system stats
curl http://localhost:8000/stats
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f ui
docker-compose logs -f neo4j
docker-compose logs data-init  # One-time init logs
```

---

## 🔧 Management Commands

### Stop Services

```bash
docker-compose down
```

### Restart Services

```bash
docker-compose restart
```

### Rebuild After Code Changes

```bash
docker-compose up --build
```

### Clean Everything (including data)

```bash
docker-compose down -v
```

**Warning**: This deletes all volumes including Neo4j data!

### Reset and Reinitialize

```bash
# Stop and clean
docker-compose down -v

# Restart (will regenerate all data)
docker-compose up --build
```

---

## 🐛 Troubleshooting

### Neo4j Won't Start

**Error**: `Neo4j is unavailable`

**Solutions**:
```bash
# Check Neo4j logs
docker-compose logs neo4j

# Restart Neo4j
docker-compose restart neo4j

# Clean volumes and restart
docker-compose down -v
docker-compose up neo4j
```

### Data Initialization Fails

**Error**: `data-init` container exits with error

**Solutions**:
```bash
# View initialization logs
docker-compose logs data-init

# Common issues:
# 1. Neo4j not ready - wait longer and retry
# 2. Out of memory - increase Docker memory to 4GB+
# 3. Network issues - check internet connection

# Manually run initialization
docker-compose run --rm data-init python scripts/init-data.py
```

### API or UI Won't Start

**Error**: `Waiting for data initialization...` stuck

**Solutions**:
```bash
# Check if data-init completed
docker-compose ps data-init

# Should show "Exit 0" or "exited (0)"
# If still running, wait for completion

# Check for .init_complete marker
docker-compose exec api ls -la /app/data/.init_complete
```

### Port Already in Use

**Error**: `Bind for 0.0.0.0:8000 failed: port is already allocated`

**Solutions**:
```bash
# Find what's using the port
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Kill the process or change ports in docker-compose.yml
# For example, change "8000:8000" to "8001:8000"
```

### Out of Disk Space

**Error**: `no space left on device`

**Solutions**:
```bash
# Clean up unused Docker resources
docker system prune -a --volumes

# Check Docker disk usage
docker system df
```

---

## 📊 What Data is Generated?

The `data-init` service creates:

### 1. **Synthetic Security Events** (`data/synthetic/security_events.parquet`)
- **~2000+ events** across 50+ anomaly types
- **Normal baseline**: 100 users with typical behavior
- **Anomalies**:
  - Multi-email free-tier churn (5 accounts)
  - Shared device fan-out (10 accounts)
  - IP rotation abuse (20 IPs)
  - Velocity violations (100 logins/hour)
  - Signup storms (50 accounts/10 minutes)
  - 20+ other fraud patterns

### 2. **Neo4j Graph** (persisted in Docker volume)
- **Nodes**: Users, Emails, Devices, IPs, Sessions
- **Relationships**: USES, LOGS_IN_FROM, HAS_SESSION, FROM_IP
- **Temporal attributes**: `valid_from`, `valid_to`, `observed_at`
- **Clusters**: Detected shared device clusters (3+ accounts)

### 3. **Temporal Features** (`data/synthetic/temporal_features.parquet`)
- Rolling window aggregations (7/14/30 days)
- Burstiness coefficients
- Exponential decay counts
- Velocity metrics (events/hour, location entropy)

### 4. **FAISS Vector Index** (`artifacts/faiss/`)
- **10+ tool embeddings** (policy_checker, cluster_detector, etc.)
- **384-dimensional** MiniLM embeddings
- **Flat index** (exact search, CPU-optimized)

### 5. **Sample Audit Pairs** (`data/audits/pairs.jsonl`)
- **10 preference pairs** for DPO training
- Format: `{prompt, chosen, rejected, confidence}`

---

## 🔒 Security Notes

### Credentials

**Default credentials are for local development only!**

For production:
```yaml
# Change in docker-compose.yml:
environment:
  - NEO4J_AUTH=neo4j/YOUR_SECURE_PASSWORD

# Or use .env file:
NEO4J_PASSWORD=YOUR_SECURE_PASSWORD
```

### Network Exposure

By default, all ports are exposed to `localhost` only.

To restrict further:
```yaml
ports:
  - "127.0.0.1:8000:8000"  # API only on localhost
  - "127.0.0.1:8501:8501"  # UI only on localhost
```

### Volumes

Data is persisted in named Docker volumes:
- `neo4j_data` - Graph database
- `shared_data` - Generated datasets
- `shared_artifacts` - Models and indices

**Backup volumes:**
```bash
# Export Neo4j data
docker run --rm --volumes-from secgraph-neo4j \
  -v $(pwd):/backup \
  ubuntu tar czf /backup/neo4j_backup.tar.gz /data

# Import Neo4j data
docker run --rm --volumes-from secgraph-neo4j \
  -v $(pwd):/backup \
  ubuntu tar xzf /backup/neo4j_backup.tar.gz
```

---

## 🔄 CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/docker-build.yml
name: Docker Build

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Build Docker images
        run: docker-compose build

      - name: Start services
        run: docker-compose up -d

      - name: Wait for services
        run: |
          sleep 60
          curl -f http://localhost:8000/healthz

      - name: Run tests
        run: docker-compose exec -T api pytest

      - name: Cleanup
        run: docker-compose down -v
```

### GitLab CI

```yaml
# .gitlab-ci.yml
test:
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker-compose build
    - docker-compose up -d
    - sleep 60
    - curl -f http://localhost:8000/healthz
    - docker-compose down -v
```

---

## 🚀 Production Deployment

For production, see the Kubernetes manifests in `reason_agent/serving/k8s/`:

```bash
# Apply Kubernetes configs
kubectl apply -f reason_agent/serving/k8s/

# Or use Helm chart (if created)
helm install secgraph-rl-agent ./charts/secgraph-rl-agent
```

**Production considerations:**
- Use managed Neo4j (Aura) or AKS-hosted Neo4j
- Replace FAISS with Azure AI Search
- Add authentication/authorization
- Enable TLS/HTTPS
- Configure resource limits
- Set up monitoring (Prometheus, Grafana)
- Use Azure Cosmos DB for graph storage
- Use Azure Data Explorer for time-series data

---

## 📞 Support

**Issues with Docker setup?**

1. Check [Troubleshooting](#-troubleshooting) section
2. View logs: `docker-compose logs`
3. Open issue on GitHub with:
   - Docker version: `docker --version`
   - Docker Compose version: `docker-compose --version`
   - OS and architecture
   - Full error logs

**Common Solutions:**
- Increase Docker memory to 4GB+
- Ensure ports 7474, 7687, 8000, 8501 are free
- Restart Docker Desktop
- Clean up: `docker system prune -a --volumes`

---

## 📝 Configuration

### Environment Variables

Override defaults by creating `.env` file:

```bash
# Copy example
cp .env.example .env

# Edit values
vim .env
```

### Custom Configurations

Mount custom configs:

```yaml
# docker-compose.yml
services:
  api:
    volumes:
      - ./my-configs:/app/configs:ro
```

### Scale Services

```bash
# Run multiple API instances
docker-compose up --scale api=3
```

---

## ✅ Health Checks

All services have health checks:

```bash
# Check service status
docker-compose ps

# Should show:
# NAME                STATUS
# secgraph-neo4j      Up (healthy)
# secgraph-data-init  Exited (0)
# secgraph-api        Up (healthy)
# secgraph-ui         Up (healthy)
```

Individual health endpoints:

```bash
# Neo4j
curl http://localhost:7474

# API
curl http://localhost:8000/healthz

# UI
curl http://localhost:8501
```

---

**Built with Docker for one-command deployment. Ready for local dev and production scale.**
