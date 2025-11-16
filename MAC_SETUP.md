# Mac Setup Guide (Apple Silicon)

Optimized instructions for running SecGraph-RL Agent on Mac M1/M2/M3 machines.

---

## 🍎 Two Options for Mac Users

### Option 1: Docker (Easy, CPU-only)
**Pros:** No setup, works immediately
**Cons:** Slower training (CPU only)
**Speed:** ~100-200 tokens/sec

### Option 2: Native (Faster, Metal GPU)
**Pros:** 2-3x faster with Metal GPU acceleration
**Cons:** Requires local Python setup
**Speed:** ~300-600 tokens/sec with MPS

---

## Option 1: Docker on Mac (Easiest)

Docker on Mac doesn't support GPU passthrough, so it runs on CPU only. Still works perfectly, just slower.

### Requirements
- Docker Desktop for Mac (Apple Silicon version)
- 8GB RAM minimum
- 20GB free disk space

### Quick Start
```bash
# Clone the repo
git clone https://github.com/your-org/secgraph-rl-agent
cd secgraph-rl-agent

# Start everything (first time: ~10 minutes)
./start.sh

# Access UI at http://localhost:8501
```

**What you'll see:**
```
⚠️  No GPU detected - using CPU
For faster training:
  - Mac M1/M2/M3: Run natively (not Docker) to use Metal GPU
```

This is normal! Training will work but be slower.

---

## Option 2: Native Mac Setup (Faster with Metal GPU)

Run directly on macOS to use Apple's Metal GPU acceleration for 2-3x faster training.

### Step 1: Install Dependencies

**Install Homebrew** (if not already installed):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Install Python 3.10:**
```bash
brew install python@3.10
```

**Install Poetry:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Step 2: Install Neo4j (for graph database)

```bash
# Install Neo4j via Homebrew
brew install neo4j

# Start Neo4j
brew services start neo4j

# Set password (first time only)
# Navigate to http://localhost:7474
# Default user: neo4j / password: neo4j
# Change password to: secgraph123
```

### Step 3: Install Python Dependencies

```bash
cd SecGraph-RL-Agent

# Install all dependencies (this will install PyTorch with MPS support)
poetry install

# Activate virtual environment
poetry shell
```

### Step 4: Download the Model

```bash
# Download TinyLlama model (~2GB)
python scripts/download_model.py
```

**You'll see:**
```
ℹ️  Downloading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
✓ Tokenizer downloaded (32000 tokens)
ℹ️  Downloading model weights...
✓ Model downloaded (1.10B parameters)
✓ Model cached in: models/
```

### Step 5: Generate Data & Start Training

```bash
# Generate synthetic security data
make ingest

# Build vector index
make index

# Option A: Launch Streamlit UI
make ui
# Access at http://localhost:8501

# Option B: Train directly
python -m reason_agent.rl.ppo
```

**What you'll see:**
```
🍎 Apple Silicon GPU detected (Metal Performance Shaders)
MPS acceleration ENABLED (2-3x faster than CPU on M1/M2/M3)
Mixed precision not supported on MPS, using float32
Device: mps
```

---

## Performance Comparison

| Setup | Device | Speed (tokens/sec) | Time for 100 episodes |
|-------|--------|-------------------|----------------------|
| **Docker on Mac** | CPU (8 cores) | ~150 | 2-3 hours |
| **Native Mac** | Metal GPU (M1/M2/M3) | ~450 | 40-60 minutes |
| **Linux NVIDIA** | CUDA + FP16 | ~1,600 | 10-15 minutes |

---

## Troubleshooting

### MPS Not Detected

If you see "No GPU detected" when running natively:

```bash
# Check PyTorch MPS support
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

If it says `False`, reinstall PyTorch:
```bash
pip uninstall torch
pip install torch torchvision torchaudio
```

### Out of Memory on Mac

If training crashes with OOM:

1. **Reduce batch size** in `configs/rl/ppo.yaml`:
   ```yaml
   training:
     batch_size: 2  # Reduce from 4
     mini_batch_size: 1
   ```

2. **Enable gradient checkpointing** (already enabled by default):
   ```yaml
   advanced:
     enable_gradient_checkpointing: true
   ```

3. **Close other applications** to free up memory

### Neo4j Connection Issues

```bash
# Check if Neo4j is running
brew services list | grep neo4j

# Restart if needed
brew services restart neo4j

# Check logs
tail -f $(brew --prefix)/var/log/neo4j/neo4j.log
```

---

## Optimizations for Mac

### 1. Use Native Instead of Docker
Running natively gives you Metal GPU access - 2-3x faster!

### 2. Optimize for Your Mac
Edit `configs/rl/ppo.yaml` based on your Mac:

**M1 (8GB RAM):**
```yaml
training:
  batch_size: 2
  mini_batch_size: 1
```

**M1 Pro/Max (16GB+ RAM):**
```yaml
training:
  batch_size: 4
  mini_batch_size: 2
```

**M2/M3 (16GB+ RAM):**
```yaml
training:
  batch_size: 8
  mini_batch_size: 2
```

### 3. Monitor GPU Usage
```bash
# In separate terminal, monitor GPU
sudo powermetrics --samplers gpu_power -i1000
```

---

## Switching Between Docker and Native

**From Docker to Native:**
```bash
# Stop Docker containers
docker-compose down

# Use native commands
poetry shell
make ui
```

**From Native to Docker:**
```bash
# Stop native services
# Press Ctrl+C in Streamlit terminal

# Start Docker
./start.sh
```

---

## Common Commands

```bash
# Native Mac Setup
poetry shell                    # Activate environment
python scripts/download_model.py # Download model
make ingest                     # Generate data
make index                      # Build vector index
make ui                         # Start Streamlit UI
python -m reason_agent.rl.ppo   # Train PPO directly

# Docker Setup
./start.sh                      # Start everything
docker-compose logs -f ui       # View UI logs
docker-compose down             # Stop everything
```

---

## Recommended Workflow

**For Development (Fastest):**
Use native Mac setup with Metal GPU.

**For Production/Deployment:**
Use Docker for consistency across platforms.

**For Quick Testing:**
Use Docker if you just want to try it out.

---

## Getting Help

- Check logs: `docker-compose logs ui` (Docker) or terminal output (native)
- GPU detection: Look for 🍎 Apple Silicon message
- Performance: Native should be 2-3x faster than Docker
- Issues: Open GitHub issue with output from `python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"`

---

Built for Mac. Optimized for Apple Silicon. Ready to train! 🍎
