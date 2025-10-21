# 🚀 QuickStart Guide - Get Running in 60 Seconds

## One Command Deployment

```bash
./start.sh
```

That's literally it!

## What Happens Behind the Scenes

```
┌─────────────────────────────────────────────────────────────┐
│ ./start.sh                                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Build Docker Images (~2 minutes first time)          │
│  ✓ Python 3.10 base image                                   │
│  ✓ Install Poetry dependencies                              │
│  ✓ Copy application code                                    │
│  ✓ Create app user and directories                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Start Neo4j Database (~30 seconds)                   │
│  ✓ Pull neo4j:5.14-community image                          │
│  ✓ Start Neo4j on ports 7474, 7687                          │
│  ✓ Wait for health check (http://localhost:7474)            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Initialize Data (~3-5 minutes first time)            │
│  ✓ Generate synthetic security events                       │
│    - 100 normal users with typical behavior                 │
│    - 5 accounts doing free-tier churn                       │
│    - 10 accounts sharing devices                            │
│    - 20 IPs in rotation abuse                               │
│    - 100 logins/hour velocity violations                    │
│    - 50 accounts in signup storm                            │
│    - 20+ other fraud patterns                               │
│    Total: ~2000+ events                                     │
│                                                              │
│  ✓ Load into Neo4j temporal graph                           │
│    - Create nodes (Users, Emails, Devices, IPs, Sessions)   │
│    - Create relationships with temporal attributes          │
│    - Detect shared device clusters                          │
│    - Build graph indices                                    │
│                                                              │
│  ✓ Extract temporal features                                │
│    - Rolling window aggregations (7/14/30 days)             │
│    - Burstiness coefficients                                │
│    - Velocity metrics                                       │
│    - Exponential decay counts                               │
│                                                              │
│  ✓ Build FAISS vector index                                 │
│    - Embed 10+ tool cards (MiniLM-L6-v2)                    │
│    - Create Flat index (384-dim, cosine)                    │
│    - Save to artifacts/faiss/                               │
│                                                              │
│  ✓ Create sample audit pairs                                │
│    - 10 preference pairs for DPO training                   │
│    - Format: {prompt, chosen, rejected, confidence}         │
│    - Save to data/audits/pairs.jsonl                        │
│                                                              │
│  ✓ Mark completion (data/.init_complete)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Start API Service (~10 seconds)                      │
│  ✓ Wait for data initialization to complete                 │
│  ✓ Load tools registry (10+ tools)                          │
│  ✓ Initialize reasoner components                           │
│  ✓ Start FastAPI on port 8000                               │
│  ✓ Health check: GET /healthz                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Start Streamlit UI (~10 seconds)                     │
│  ✓ Wait for API to be healthy                               │
│  ✓ Load UI components (5 tabs)                              │
│  ✓ Connect to Neo4j                                         │
│  ✓ Start Streamlit on port 8501                             │
│  ✓ Health check: GET http://localhost:8501                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 🎉 READY!                                                    │
│                                                              │
│  📊 Streamlit UI:  http://localhost:8501                     │
│  🔌 API Docs:      http://localhost:8000/docs                │
│  🗄️  Neo4j:        http://localhost:7474                     │
│                                                              │
│  Credentials: neo4j / secgraph123                            │
└─────────────────────────────────────────────────────────────┘
```

## Total Time

- **First Run**: 5-10 minutes
  - Docker image build: ~2 min
  - Model downloads: ~1 min
  - Data generation: ~2-3 min
  - Graph loading: ~1 min
  - Index building: ~1 min

- **Subsequent Runs**: ~30 seconds
  - Uses cached images
  - Uses persisted data volumes
  - Only starts services

## What You Get

### 1. **API Service** (http://localhost:8000)

```bash
# Health check
curl http://localhost:8000/healthz

# Run agent query
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Detect multi-account abuse with shared devices",
    "max_steps": 10
  }'

# Get reasoning trace
curl http://localhost:8000/trace/{trace_id}

# System stats
curl http://localhost:8000/stats
```

### 2. **Interactive UI** (http://localhost:8501)

**Tab 1: Query**
- Run agent queries
- View reasoning traces step-by-step
- See verifiable rewards
- Download trace JSON

**Tab 2: Graph**
- Inspect temporal graph snapshots
- View node counts by type
- Detect shared device clusters
- Run policy checks

**Tab 3: RL Training**
- Train PPO on math tasks
- Train DPO on audit pairs
- View reward curves

**Tab 4: Audits**
- Compare two reasoning traces
- Select preferred trace
- Create DPO preference pairs

**Tab 5: Drift**
- Monitor tool success rates
- Track reward EWMA trends
- View drift alerts

### 3. **Neo4j Graph** (http://localhost:7474)

**Pre-loaded with:**
- 150+ User nodes
- 160+ Email nodes
- 45+ Device nodes
- 80+ IP nodes
- 500+ Session nodes
- Temporal relationships with `valid_from`, `valid_to`, `observed_at`

**Example Cypher query:**
```cypher
// Find accounts sharing devices
MATCH (u1:User)-[:USES]->(d:Device)<-[:USES]-(u2:User)
WHERE u1.id < u2.id
RETURN d.id AS device, count(*) AS accounts
ORDER BY accounts DESC
LIMIT 10
```

## Example Workflow

### 1. Run a Query via UI

1. Open http://localhost:8501
2. Go to "Query" tab
3. Enter: "Detect multi-account abuse with shared devices"
4. Click "Run Query"
5. View:
   - Reasoning steps
   - Tool calls and results
   - Verifiable rewards
   - Full trace JSON

### 2. Run a Query via API

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Detect multi-account abuse",
    "enable_verification": true
  }' | jq '.'
```

### 3. Inspect the Graph

1. Open http://localhost:7474
2. Login: neo4j / secgraph123
3. Run query:
```cypher
MATCH (u:User)-[r:USES]->(d:Device)
RETURN u, r, d
LIMIT 25
```

### 4. Train RL Models

**Via UI:**
1. Go to "RL Training" tab
2. Click "Train PPO" or "Train DPO"
3. View training curves

**Via CLI:**
```bash
# Access running container
docker exec -it secgraph-api bash

# Train PPO
python -m reason_agent.cli.train_rl_cli --mode ppo --episodes 100

# Train DPO
python -m reason_agent.cli.train_rl_cli --mode dpo --epochs 1
```

## Stopping and Restarting

### Stop Everything

```bash
docker-compose down
```

Services stop, but data is preserved in volumes.

### Restart

```bash
docker-compose up
```

Services start instantly (no data regeneration).

### Clean Slate (Delete All Data)

```bash
docker-compose down -v
./start.sh  # Regenerates everything
```

## Troubleshooting

### "Port already in use"

```bash
# Check what's using port 8000
lsof -i :8000

# Or change port in docker-compose.yml
# ports:
#   - "8001:8000"  # Use 8001 instead
```

### "Neo4j won't start"

```bash
# View logs
docker-compose logs neo4j

# Restart Neo4j
docker-compose restart neo4j

# Increase Docker memory to 4GB+ in Docker Desktop settings
```

### "Data init takes too long"

First run downloads models (~500MB) and generates data. This is normal.

```bash
# View progress
docker-compose logs -f data-init

# Expected output:
# [1/5] Generating synthetic data...
# [2/5] Loading data into Neo4j...
# [3/5] Extracting temporal features...
# [4/5] Building FAISS index...
# [5/5] Creating sample audit pairs...
# ✓ Data initialization complete!
```

### "UI shows 'Waiting for data...'"

Data init is still running. Wait for it to complete.

```bash
# Check data-init status
docker-compose ps data-init

# Should eventually show "Exited (0)"
```

## Next Steps

- **Explore UI**: Try all 5 tabs
- **Run Queries**: Test different anomaly detection scenarios
- **Inspect Graph**: Browse Neo4j for fraud patterns
- **Train Models**: Experiment with PPO/DPO
- **Read Docs**: Check DOCKER.md and README.md
- **Customize**: Edit configs in `configs/`
- **Scale Up**: Deploy to Kubernetes with manifests in `reason_agent/serving/k8s/`

## Resources

- **Main README**: [README.md](README.md)
- **Docker Guide**: [DOCKER.md](DOCKER.md)
- **API Docs**: http://localhost:8000/docs (after starting)
- **GitHub**: [Issues & Discussions](https://github.com/your-org/secgraph-rl-agent)

---

**Built for simplicity. Designed for scale. Ready in 60 seconds.**
