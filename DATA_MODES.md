# 📊 Data Modes Guide

SecGraph-RL Agent supports 3 data generation modes optimized for different scenarios.

---

## 🎯 **Data Modes**

### **1. Lightweight Mode** (200 events)
**Use Case:** Quick testing, minimal resources
**Target:** Initial exploration, CI/CD
**Time:** ~1 minute
**Memory:** <500MB

**Configuration:**
```yaml
DATA_MODE=lightweight
USE_PARALLEL=false  # Not needed for small dataset
```

**Data Generated:**
- 20 normal users (3-8 sessions each)
- 3 multi-email fraud cases
- 5 shared device cases
- 5 IP rotation cases
- 3 additional anomaly types
- **Total:** ~200 events

---

### **2. Demo Mode** ⭐ (800-1000 events) **[DEFAULT]**
**Use Case:** **UI demonstrations, populated graphs, realistic showcase**
**Target:** Technical presentations, stakeholder demos
**Time:** ~2-3 minutes (with M1 parallel processing)
**Memory:** <1.5GB

**Configuration:**
```yaml
DATA_MODE=demo
USE_PARALLEL=true  # Recommended for M1 Mac
```

**Data Generated:**
- 80 normal users (5-12 sessions each) → **~600 events**
- 10 multi-email fraud cases
- 15 shared device cases
- 15 IP rotation cases
- 50 velocity violations
- 20 signup storms
- 15 additional anomaly types (3-7 events each) → **~60 events**
- **Total:** ~800-1000 events

**Why This Mode:**
✅ UI graphs are **fully populated**
✅ Fraud clusters are **clearly visible**
✅ Temporal patterns are **meaningful**
✅ Still fast enough for demos (<3 min init)
✅ Shows diversity of anomaly types

---

### **3. Full Mode** (3500+ events)
**Use Case:** Production testing, performance benchmarks
**Target:** Load testing, research
**Time:** ~5-8 minutes (with parallel processing)
**Memory:** <3GB

**Configuration:**
```yaml
DATA_MODE=full
USE_PARALLEL=true  # Highly recommended
```

**Data Generated:**
- 100 normal users (10-50 sessions each)
- 5 multi-email fraud cases
- 10 shared device cases
- 20 IP rotation cases
- 100 velocity violations
- 50 signup storms
- 20 additional anomaly types (3-8 events each)
- **Total:** ~3500+ events

---

## ⚡ **Parallel Processing**

### **M1 Mac Optimization**

Set `USE_PARALLEL=true` to enable multiprocessing:

**Benefits:**
- Uses all CPU cores (up to 8 on M1)
- **3-5x faster** data generation
- Efficient for demo and full modes

**How It Works:**
1. Splits normal user generation across CPU cores
2. Each core generates a chunk independently
3. Results are merged at the end

**Example Performance:**
```
Without Parallel:
- Demo mode: ~5-6 minutes
- Full mode: ~15-20 minutes

With Parallel (M1 Max):
- Demo mode: ~2-3 minutes ⚡
- Full mode: ~5-8 minutes ⚡
```

---

## 🚀 **Quick Start**

### **For Demos (Recommended):**
```bash
# Edit docker-compose.yml
DATA_MODE=demo
USE_PARALLEL=true

# Run
docker-compose down -v
docker-compose up -d
docker-compose logs -f data-init
```

Expected output:
```
[  0.0%] Waiting for Neo4j... (elapsed: 2.0s)
[ 20.0%] [1/5] Generating synthetic security events... (elapsed: 5.0s)
  Data mode: demo, Parallel processing: true
  Using 8 parallel processes...
  Generated 620 normal events
  Generating demo anomaly patterns...
  Generated 850 events across 18 anomaly types
[ 20.0%] ✓ Step 1 complete (elapsed: 15.2s)
[ 40.0%] [2/5] Loading data into Neo4j... (elapsed: 15.5s)
  Loading with batch_size=200...
  Detected 15 suspicious device clusters
[ 40.0%] ✓ Step 2 complete: Graph loaded (elapsed: 45.3s)
...
[100.0%] ✓ Data initialization complete! (Total time: 120.5s)
```

### **For Quick Testing:**
```bash
DATA_MODE=lightweight
USE_PARALLEL=false

docker-compose up -d
```

### **For Load Testing:**
```bash
DATA_MODE=full
USE_PARALLEL=true

docker-compose up -d
```

---

## 📊 **Comparison Table**

| Feature | Lightweight | Demo ⭐ | Full |
|---------|-------------|---------|------|
| **Events** | ~200 | ~850 | ~3500 |
| **Users** | 20 | 80 | 100 |
| **Anomaly Types** | 6 | 18 | 26 |
| **Fraud Clusters** | 3-5 | 10-15 | 30-40 |
| **Init Time (M1)** | ~1 min | ~2-3 min | ~5-8 min |
| **Memory Usage** | <500MB | <1.5GB | <3GB |
| **UI Populated?** | ⚠️ Sparse | ✅ Full | ✅ Full |
| **Use Case** | Testing | **Demos** | Production |

---

## 🎨 **UI Impact**

### **Lightweight Mode:**
- Graph tab: ⚠️ Sparse clusters
- Drift tab: ⚠️ Insufficient data for trends
- Audit tab: ⚠️ Few traces to compare

### **Demo Mode ⭐:**
- Graph tab: ✅ **Clear fraud clusters visible**
- Drift tab: ✅ **Meaningful trend lines**
- Audit tab: ✅ **Multiple traces to compare**
- Query tab: ✅ **Realistic reasoning depth**

### **Full Mode:**
- Graph tab: ✅ Complex network structures
- Drift tab: ✅ Long-term trends
- Audit tab: ✅ Extensive trace history
- Query tab: ✅ Production-realistic scenarios

---

## 🔧 **Customization**

### **Create Custom Mode:**

Edit `reason_agent/ingest/synthetic_generator.py`:

```python
elif mode == "custom":
    num_normal_users = 50
    sessions_range = (8, 15)
    anomaly_config = {
        'multi_email': 8,
        'shared_device': 12,
        'ip_rotation': 10,
        'velocity': 30,
        'signup_storm': 15,
        'stub_types': 10,
        'stub_events_per_type': (4, 6)
    }
```

Then use:
```bash
DATA_MODE=custom docker-compose up -d
```

---

## 💡 **Recommendations**

**Use Demo Mode if:**
- ✅ Presenting to stakeholders
- ✅ Recording demos
- ✅ Need populated UI graphs
- ✅ Want realistic fraud scenarios
- ✅ Have 5-10 minutes for init

**Use Lightweight Mode if:**
- ✅ Quick iteration during development
- ✅ CI/CD pipeline testing
- ✅ Limited time/memory
- ✅ Just testing core functionality

**Use Full Mode if:**
- ✅ Load testing
- ✅ Research experiments
- ✅ Benchmarking performance
- ✅ Simulating production scale

---

## 🐛 **Troubleshooting**

### **"Initialization takes >10 minutes"**
- Check `USE_PARALLEL=true` is set
- Verify you're on demo mode, not full
- Check CPU usage (should see 8 Python processes)

### **"Out of memory"**
- Switch to lightweight mode
- Or keep demo mode but reduce Neo4j memory:
  ```yaml
  NEO4J_dbms_memory_heap_max__size=256m
  ```

### **"UI shows empty graphs"**
- Confirm you're using `demo` or `full` mode
- Check logs: `docker-compose logs data-init`
- Verify init completed successfully

---

**Default Mode:** Demo (optimal for showcasing)
**Recommended for M1 Mac:** Demo + Parallel Processing
**Init Time Target:** <3 minutes
