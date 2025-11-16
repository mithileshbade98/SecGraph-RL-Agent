# LoRA Implementation Guide

This document describes the complete LoRA (Low-Rank Adaptation) implementation for parameter-efficient fine-tuning in the SecGraph-RL-Agent project.

## Overview

LoRA enables efficient fine-tuning of large language models by training only small adapter matrices (typically 1-5% of total parameters) instead of the entire model. This provides:

- **10x faster training** compared to full fine-tuning
- **80% less memory usage**
- **Easy adapter switching** for different tasks
- **No degradation in model quality**

## Architecture

### Components

1. **LoRA Utilities** (`reason_agent/rl/lora_utils.py`)
   - Model initialization with PEFT
   - Adapter saving and loading
   - Preprocessing (tokenization)
   - Postprocessing (decoding)
   - Generation utilities

2. **PPO Training** (`reason_agent/rl/ppo.py`)
   - Trains LoRA adapters using Proximal Policy Optimization
   - Optimizes for verifiable rewards
   - Saves adapters to `artifacts/models/ppo_adapter/`

3. **DPO Training** (`reason_agent/rl/dpo.py`)
   - Trains LoRA adapters using Direct Preference Optimization
   - Learns from preference pairs (chosen vs. rejected)
   - Saves adapters to `artifacts/models/dpo_adapter/`

4. **Inference** (`reason_agent/reasoning/planner.py`)
   - Loads trained LoRA adapters
   - Uses fine-tuned models for reasoning and planning
   - Falls back to heuristic planning if no adapter

## Configuration

LoRA parameters are configured in YAML files:

### PPO Configuration (`configs/rl/ppo.yaml`)

```yaml
peft_config:
  method: lora
  r: 16                    # LoRA rank (lower = fewer params)
  lora_alpha: 32           # Scaling factor
  lora_dropout: 0.1        # Dropout probability
  target_modules:          # Modules to apply LoRA
    - q_proj
    - v_proj
  bias: none               # Bias handling
  task_type: CAUSAL_LM     # Task type
```

### DPO Configuration (`configs/rl/dpo.yaml`)

```yaml
peft_config:
  method: lora
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules:          # More comprehensive for DPO
    - q_proj
    - v_proj
    - k_proj
    - o_proj
  bias: none
  task_type: CAUSAL_LM
```

### Key Parameters

- **r (rank)**: Controls adapter size. Higher = more capacity but more params
  - Typical range: 8-64
  - Default: 16

- **lora_alpha**: Scaling factor for LoRA updates
  - Rule of thumb: 2x the rank
  - Default: 32

- **lora_dropout**: Regularization dropout
  - Typical range: 0.0-0.1
  - Default: 0.1

- **target_modules**: Which linear layers to adapt
  - Attention projections: q_proj, k_proj, v_proj, o_proj
  - FFN: up_proj, down_proj, gate_proj (optional)

## Usage

### 1. Training a PPO Adapter

```python
from pathlib import Path
from reason_agent.rl.ppo import PPOTrainer

# Initialize trainer with base model
trainer = PPOTrainer(
    config_path=Path("configs/rl/ppo.yaml"),
    base_model="meta-llama/Llama-2-7b-hf",  # Or local path
    initialize_model=True,
)

# Train
metrics = trainer.train(
    num_episodes=100,
    save_path=Path("artifacts/models/ppo_adapter"),
)
```

### 2. Training a DPO Adapter

```python
from pathlib import Path
from reason_agent.rl.dpo import DPOTrainer

# Initialize trainer
trainer = DPOTrainer(
    config_path=Path("configs/rl/dpo.yaml"),
    base_model="meta-llama/Llama-2-7b-hf",
    initialize_model=True,
)

# Train on preference pairs
metrics = trainer.train(
    data_path=Path("data/audits/pairs.jsonl"),
    num_epochs=50,
    save_path=Path("artifacts/models/dpo_adapter"),
)
```

### 3. Using Trained Adapters for Inference

```python
from pathlib import Path
from reason_agent.reasoning.planner import ReasoningPlanner
from reason_agent.tools.registry import ToolRegistry
from reason_agent.tools.router import ToolRouter
from reason_agent.tools.executors import ExecutorRegistry

# Set up components
registry = ToolRegistry()
router = ToolRouter(registry, method="rule")
executors = ExecutorRegistry()

# Initialize planner with LoRA adapter
planner = ReasoningPlanner(
    tool_registry=registry,
    tool_router=router,
    executor_registry=executors,
    use_llm=True,
    lora_adapter_path=Path("artifacts/models/ppo_adapter"),
    base_model="meta-llama/Llama-2-7b-hf",
)

# Use for reasoning
trace = planner.plan_and_execute("Detect multi-account abuse")
```

### 4. Preprocessing and Postprocessing

```python
from reason_agent.rl.lora_utils import (
    preprocess_text,
    preprocess_batch,
    postprocess_output,
)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Preprocess single text
inputs = preprocess_text(
    text="Your query here",
    tokenizer=tokenizer,
    max_length=512,
)

# Preprocess batch
batch_inputs = preprocess_batch(
    texts=["Query 1", "Query 2", "Query 3"],
    tokenizer=tokenizer,
    max_length=512,
)

# Postprocess model output
decoded_text = postprocess_output(
    output_ids=model_output,
    tokenizer=tokenizer,
    skip_special_tokens=True,
)
```

### 5. Direct Generation with LoRA Model

```python
from reason_agent.rl.lora_utils import load_lora_adapter, generate_with_lora

# Load adapter
model, tokenizer = load_lora_adapter(
    adapter_path="artifacts/models/ppo_adapter",
    base_model_name="meta-llama/Llama-2-7b-hf",
)

# Generate
response = generate_with_lora(
    model=model,
    tokenizer=tokenizer,
    prompt="Your prompt here",
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
)
```

## Adapter Structure

When saved, adapters have the following structure:

```
artifacts/models/ppo_adapter/
├── adapter_config.json       # PEFT configuration
├── adapter_model.bin          # LoRA weights
├── adapter_info.json          # Custom metadata
├── tokenizer_config.json      # Tokenizer configuration
├── tokenizer.json             # Tokenizer vocabulary
└── special_tokens_map.json    # Special tokens
```

## CLI Usage

### Train PPO Adapter

```bash
python -m reason_agent.rl.ppo \
    --config configs/rl/ppo.yaml \
    --episodes 100 \
    --save-path artifacts/models/ppo_adapter
```

### Train DPO Adapter

```bash
python -m reason_agent.rl.dpo \
    --config configs/rl/dpo.yaml \
    --data data/audits/pairs.jsonl \
    --epochs 50 \
    --save-path artifacts/models/dpo_adapter
```

### Test Planner with LoRA

```bash
python -m reason_agent.reasoning.planner \
    --query "Detect multi-account abuse" \
    --lora-adapter artifacts/models/ppo_adapter \
    --base-model meta-llama/Llama-2-7b-hf
```

## Model Selection

### Recommended Base Models

1. **For Development/Testing:**
   - `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B params, fast)
   - `google/flan-t5-base` (250M params, very fast)

2. **For Production:**
   - `meta-llama/Llama-2-7b-hf` (7B params, good balance)
   - `meta-llama/Llama-2-13b-hf` (13B params, better quality)
   - `mistralai/Mistral-7B-v0.1` (7B params, excellent)

3. **For Resource-Constrained:**
   - Any model with 8-bit quantization: `load_in_8bit=True`

## Memory Requirements

Approximate GPU memory requirements:

| Model Size | Full Fine-tuning | LoRA (r=16) | LoRA + 8-bit |
|-----------|------------------|-------------|--------------|
| 1B params | ~8 GB            | ~2 GB       | ~1.5 GB      |
| 7B params | ~28 GB           | ~8 GB       | ~5 GB        |
| 13B params| ~52 GB           | ~14 GB      | ~9 GB        |

## Performance Tuning

### Increase Capacity

```yaml
peft_config:
  r: 32                    # Increase rank
  lora_alpha: 64           # Increase alpha proportionally
  target_modules:          # Add more modules
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - up_proj              # Add FFN layers
    - down_proj
```

### Reduce Memory

```yaml
training:
  load_in_8bit: true       # Enable 8-bit quantization
  gradient_checkpointing: true  # Trade compute for memory

peft_config:
  r: 8                     # Reduce rank
  target_modules:          # Fewer modules
    - q_proj
    - v_proj
```

### Speed Up Training

```yaml
training:
  batch_size: 8            # Larger batches (if memory allows)
  gradient_accumulation: 4 # Simulate larger batches
  mixed_precision: fp16    # Use FP16 training
```

## Troubleshooting

### Issue: Out of memory during training

**Solutions:**
1. Enable 8-bit quantization: `load_in_8bit=True`
2. Reduce LoRA rank: `r: 8`
3. Reduce target modules
4. Use gradient checkpointing
5. Reduce batch size
6. Use smaller base model

### Issue: Poor adapter quality

**Solutions:**
1. Increase LoRA rank: `r: 32` or `r: 64`
2. Add more target modules (include FFN layers)
3. Train for more episodes/epochs
4. Adjust learning rate
5. Use larger base model

### Issue: Adapter not loading

**Solutions:**
1. Ensure base model name matches training
2. Check adapter files exist in path
3. Verify PEFT library version compatibility
4. Check device compatibility (CUDA vs CPU)

### Issue: Slow inference

**Solutions:**
1. Merge adapter into base: `merge_adapter_to_base()`
2. Use smaller base model
3. Reduce `max_new_tokens`
4. Enable batch processing
5. Use GPU if available

## Best Practices

1. **Start Small:** Begin with TinyLlama or smaller models for testing
2. **Match Models:** Always use the same base model for training and inference
3. **Save Often:** Training can be interrupted, save checkpoints regularly
4. **Monitor Metrics:** Track training curves to detect issues early
5. **Version Adapters:** Save metadata with each adapter (date, config, metrics)
6. **Test First:** Validate adapter quality before deploying to production
7. **Document Changes:** Keep notes on what worked and what didn't

## Advanced Topics

### Custom Target Modules

```python
from reason_agent.rl.lora_utils import create_lora_config

# Create custom config
custom_config = {
    'r': 32,
    'lora_alpha': 64,
    'target_modules': [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',  # Attention
        'gate_proj', 'up_proj', 'down_proj',     # FFN
    ],
}

lora_config = create_lora_config(custom_config)
```

### Merging Adapters

```python
from reason_agent.rl.lora_utils import load_lora_adapter, merge_adapter_to_base

# Load adapter
model, tokenizer = load_lora_adapter("artifacts/models/ppo_adapter")

# Merge for faster inference
merged_model = merge_adapter_to_base(
    peft_model=model,
    save_merged_path="artifacts/models/ppo_merged",
)
```

### Multi-Adapter Inference

```python
# Load base model once
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Switch between adapters
ppo_model = PeftModel.from_pretrained(base_model, "artifacts/models/ppo_adapter")
dpo_model = PeftModel.from_pretrained(base_model, "artifacts/models/dpo_adapter")
```

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [PEFT Library](https://github.com/huggingface/peft)
- [Transformers Library](https://github.com/huggingface/transformers)
- [PPO Paper](https://arxiv.org/abs/2410.15246) - RL with Verifiable Rewards
- [DPO Paper](https://arxiv.org/abs/2305.18290) - Direct Preference Optimization

## Support

For issues or questions:
1. Check this documentation
2. Review example scripts in `examples/lora_training_example.py`
3. Check configuration files in `configs/rl/`
4. Open an issue on GitHub
