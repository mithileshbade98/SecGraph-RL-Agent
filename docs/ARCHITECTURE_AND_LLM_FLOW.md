# SecGraph-RL-Agent: Complete Architecture & LLM Integration

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [LLM Call Points](#llm-call-points)
3. [Data Flow: Preprocessing → LLM → Postprocessing](#data-flow)
4. [What's Implemented vs Missing](#whats-implemented-vs-missing)
5. [Complete Pipeline](#complete-pipeline)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SecGraph-RL-Agent Architecture                │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Training   │      │  Inference   │      │  Embeddings  │
│              │      │              │      │              │
│  - PPO       │      │  - Planner   │      │  - Router    │
│  - DPO       │      │  - API       │      │  - Retrieval │
└──────┬───────┘      └──────┬───────┘      └──────┬───────┘
       │                     │                     │
       │                     │                     │
       └─────────────────────┴─────────────────────┘
                             │
                   ┌─────────▼──────────┐
                   │   LoRA Utilities   │
                   │                    │
                   │  - Preprocessing   │
                   │  - Model Init      │
                   │  - Generation      │
                   │  - Postprocessing  │
                   │  - Save/Load       │
                   └─────────┬──────────┘
                             │
                   ┌─────────▼──────────┐
                   │   Base LLM Model   │
                   │                    │
                   │  - Llama-2-7b      │
                   │  - Mistral-7B      │
                   │  - TinyLlama       │
                   └────────────────────┘
```

---

## LLM Call Points

### 🎯 Where the LLM is Actually Hit

#### 1. **Primary LLM Generation** (Production)
**File:** `reason_agent/rl/lora_utils.py:386`
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
    do_sample=do_sample,
    pad_token_id=tokenizer.pad_token_id,
)
```
- **Called by:** `generate_with_lora()` function
- **Used in:** Planner for reasoning, inference endpoints
- **Frequency:** Every planning/inference request

#### 2. **Planner LLM Planning** (Production)
**File:** `reason_agent/reasoning/planner.py:219`
```python
response = generate_with_lora(
    model=self.lora_model,
    tokenizer=self.tokenizer,
    prompt=prompt,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
)
```
- **Called by:** `_lora_plan()` method in ReasoningPlanner
- **Trigger:** When `use_llm=True` and LoRA adapter is loaded
- **Input:** Query + tool descriptions formatted as prompt
- **Output:** Step-by-step reasoning plan

#### 3. **Training Model Initialization** (Setup Only)
**Files:**
- `reason_agent/rl/ppo.py:76` - PPO model init
- `reason_agent/rl/dpo.py:80` - DPO policy model init
- `reason_agent/rl/dpo.py:97` - DPO reference model init

```python
self.model, self.tokenizer = initialize_peft_model(
    base_model_name=self.base_model_name,
    peft_config=self.peft_config,
    device=None,
    load_in_8bit=self.training_config.get('load_in_8bit', False),
)
```
- **Called by:** Trainer `__init__` with `initialize_model=True`
- **Purpose:** Load base model + attach LoRA adapters
- **Note:** ⚠️ **NOT used in actual training loop currently** (see Missing section)

#### 4. **Embedding Model** (Semantic Similarity)
**File:** `reason_agent/embeddings/text_embedder.py:72`
```python
embeddings = self.model.encode(
    texts,
    batch_size=batch_size,
    show_progress_bar=False,
    convert_to_numpy=True,
)
```
- **Model:** Sentence-Transformers (MiniLM/GTE)
- **Used in:** Tool routing, semantic search
- **Not LoRA:** Different model type (bi-encoder)

---

## Data Flow: Preprocessing → LLM → Postprocessing

### Complete Pipeline

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. PREPROCESSING                                            │
│    Location: reason_agent/rl/lora_utils.py:262-323         │
└─────────────────────────────────────────────────────────────┘
    │
    │  preprocess_text() or preprocess_batch()
    │  ┌────────────────────────────────────────┐
    │  │ • Load tokenizer                       │
    │  │ • Tokenize text to IDs                 │
    │  │ • Add special tokens ([BOS], [EOS])    │
    │  │ • Pad to max_length                    │
    │  │ • Create attention mask                │
    │  │ • Convert to tensors (PyTorch)         │
    │  └────────────────────────────────────────┘
    │
    ▼
input_ids: [1, 450, 338, 263, 1243, ...]  (token IDs)
attention_mask: [1, 1, 1, 1, ...]          (valid positions)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. LLM INFERENCE                                            │
│    Location: reason_agent/rl/lora_utils.py:353-400         │
└─────────────────────────────────────────────────────────────┘
    │
    │  generate_with_lora(model, tokenizer, prompt, ...)
    │  ┌────────────────────────────────────────┐
    │  │ • Move inputs to GPU/CPU               │
    │  │ • Call model.generate()                │
    │  │   - Forward pass through base model    │
    │  │   - Apply LoRA adapters (if trained)   │
    │  │   - Sampling/decoding (temperature,    │
    │  │     top-p, top-k)                      │
    │  │   - Generate new tokens autoregressively│
    │  └────────────────────────────────────────┘
    │
    ▼
output_ids: [1, 450, 338, ..., 29871, 13]  (generated token IDs)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. POSTPROCESSING                                           │
│    Location: reason_agent/rl/lora_utils.py:326-350         │
└─────────────────────────────────────────────────────────────┘
    │
    │  postprocess_output(output_ids, tokenizer)
    │  ┌────────────────────────────────────────┐
    │  │ • Extract generated tokens only        │
    │  │   (skip input prompt)                  │
    │  │ • Decode token IDs to text             │
    │  │ • Remove special tokens ([BOS], [EOS]) │
    │  │ • Clean whitespace                     │
    │  └────────────────────────────────────────┘
    │
    ▼
Generated Text: "Step 1: Check policy violations..."
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. RESPONSE PARSING (Planner Only)                          │
│    Location: reason_agent/reasoning/planner.py:205-255     │
└─────────────────────────────────────────────────────────────┘
    │
    │  _parse_llm_response(response)
    │  ┌────────────────────────────────────────┐
    │  │ • Extract steps from text              │
    │  │ • Parse tool names                     │
    │  │ • Parse JSON parameters                │
    │  │ • Create Step objects                  │
    │  └────────────────────────────────────────┘
    │
    ▼
Structured Steps: [Step(1, "Check policy", "policy_checker", {...}), ...]
    │
    ▼
Reasoning Trace / API Response
```

### Detailed Preprocessing Example

```python
# Input
text = "Detect multi-account abuse from the same device"

# Step 1: Tokenization (lora_utils.py:282)
inputs = tokenizer(
    text,
    max_length=512,
    padding="max_length",
    truncation=True,
    add_special_tokens=True,
    return_tensors="pt",
)

# Output:
{
    'input_ids': tensor([[    1,  5953,   522,  2763, 29899,  5646,  633, ...  0,  0]]),
    'attention_mask': tensor([[    1,     1,     1,     1,     1, ...  0,  0]]),
}

# Step 2: Move to device
inputs = {k: v.to(device) for k, v in inputs.items()}  # GPU if available

# Step 3: Generate (lora_utils.py:386)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

# Step 4: Postprocess (lora_utils.py:396-398)
input_length = inputs['input_ids'].shape[1]
generated_ids = outputs[0][input_length:]  # Skip input tokens
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
```

---

## What's Implemented vs Missing

### ✅ Fully Implemented

1. **LoRA Utilities** (`reason_agent/rl/lora_utils.py`)
   - ✅ Model initialization with PEFT
   - ✅ Adapter saving/loading
   - ✅ Preprocessing (tokenization)
   - ✅ Postprocessing (decoding)
   - ✅ Generation utilities
   - ✅ Adapter merging

2. **Inference Pipeline** (`reason_agent/reasoning/planner.py`)
   - ✅ Load LoRA adapters
   - ✅ LLM-based planning with fine-tuned models
   - ✅ Prompt formatting
   - ✅ Response parsing
   - ✅ Fallback to heuristic planning

3. **API & CLI**
   - ✅ FastAPI endpoints (`reason_agent/serving/api.py`)
   - ✅ CLI tools (`reason_agent/cli/`)
   - ✅ Streamlit UI (`ui/streamlit_app.py`)

4. **Embeddings**
   - ✅ Sentence-Transformers integration
   - ✅ Semantic similarity
   - ✅ Tool routing

### ⚠️ Partially Implemented (Simulation Mode)

1. **PPO Training** (`reason_agent/rl/ppo.py`)
   - ✅ Model initialization
   - ✅ Configuration loading
   - ✅ Adapter saving
   - ✅ Metrics tracking
   - ❌ **Actual training loop** (generates mock metrics)
   - ❌ Forward passes through model
   - ❌ Reward computation from model outputs
   - ❌ Policy gradient updates

2. **DPO Training** (`reason_agent/rl/dpo.py`)
   - ✅ Policy & reference model initialization
   - ✅ Preference pair loading
   - ✅ Adapter saving
   - ✅ Metrics tracking
   - ❌ **Actual training loop** (generates mock metrics)
   - ❌ Log probability computation
   - ❌ DPO loss calculation
   - ❌ Gradient updates

### ❌ Not Implemented

1. **Actual RL Training Loops**
   - ❌ PPO rollout collection
   - ❌ Advantage estimation (GAE)
   - ❌ PPO clipped objective
   - ❌ Value function training
   - ❌ DPO preference loss
   - ❌ Gradient computation & backprop
   - ❌ Optimizer steps

2. **Data Processing for Training**
   - ❌ Data collator for batching
   - ❌ Experience replay buffer
   - ❌ Trajectory collection
   - ❌ Reward shaping utilities

3. **Reward Computation**
   - ❌ Process reward model (PRM)
   - ❌ Outcome reward model (ORM)
   - ❌ Verifiable reward computation
   - ❌ Reward aggregation

4. **Advanced Features**
   - ❌ Multi-adapter switching
   - ❌ Adapter composition
   - ❌ Quantized inference (4-bit)
   - ❌ vLLM integration for fast inference

---

## Complete Pipeline

### Training Pipeline (What Should Happen)

```
1. Data Collection
   ├─ Generate reasoning traces
   ├─ Collect user feedback (DPO) or verify outcomes (PPO)
   └─ Format as training examples

2. Preprocessing
   ├─ Tokenize prompts and completions
   ├─ Create batches with data collator
   └─ Compute reference model logits (DPO)

3. Training Loop
   ├─ Forward pass through policy model
   ├─ Compute loss (PPO or DPO)
   ├─ Backward pass
   ├─ Gradient clipping
   ├─ Optimizer step
   └─ Log metrics

4. Adapter Saving
   ├─ Save LoRA weights
   ├─ Save training config
   └─ Save metrics history

5. Evaluation
   ├─ Load adapter
   ├─ Run on test queries
   └─ Measure performance
```

### Inference Pipeline (Currently Working)

```
1. Load Adapter
   └─ planner.py:107 - _load_lora_adapter()

2. Format Query
   └─ planner.py:203-215 - Create prompt with tools

3. Preprocess
   └─ lora_utils.py:370 - preprocess_text()

4. Generate
   └─ lora_utils.py:386 - model.generate()

5. Postprocess
   └─ lora_utils.py:396-398 - Extract & decode

6. Parse Response
   └─ planner.py:229 - _parse_llm_response()

7. Execute Tools
   └─ planner.py:154-171 - Execute each step

8. Return Trace
   └─ planner.py:174-182 - Compile results
```

---

## How to Use Current Implementation

### 1. Inference Only (Works Now)

```python
from reason_agent.reasoning.planner import ReasoningPlanner
from reason_agent.tools.registry import ToolRegistry
from reason_agent.tools.router import ToolRouter
from reason_agent.tools.executors import ExecutorRegistry

# Setup
registry = ToolRegistry()
router = ToolRouter(registry, method="rule")
executors = ExecutorRegistry()

# Load trained adapter (if you have one)
planner = ReasoningPlanner(
    tool_registry=registry,
    tool_router=router,
    executor_registry=executors,
    use_llm=True,
    lora_adapter_path="artifacts/models/ppo_adapter",
    base_model="meta-llama/Llama-2-7b-hf",
)

# Query
trace = planner.plan_and_execute("Detect multi-account abuse")
print(trace)
```

### 2. Training (Simulation Mode - No Real Model Updates)

```python
from reason_agent.rl.ppo import PPOTrainer

# Initialize (model loaded but not used for training)
trainer = PPOTrainer(
    base_model="meta-llama/Llama-2-7b-hf",
    initialize_model=True,
)

# "Train" (generates mock metrics, doesn't update model)
metrics = trainer.train(
    num_episodes=100,
    save_path="artifacts/models/ppo_adapter",
)
```

---

## Summary

### LLM is Hit Here:
1. **Inference:** `lora_utils.py:386` via `model.generate()`
2. **Planning:** `planner.py:219` via `generate_with_lora()`
3. **Embeddings:** `text_embedder.py:72` via `model.encode()` (different model)

### Preprocessing Happens:
- `lora_utils.py:282` - Single text tokenization
- `lora_utils.py:314` - Batch text tokenization

### Postprocessing Happens:
- `lora_utils.py:345` - Single output decoding
- `lora_utils.py:349` - Batch output decoding
- `planner.py:229` - Response parsing to steps

### Missing for Full Training:
1. Actual forward/backward passes in training loops
2. Loss computation (PPO clipped objective, DPO preference loss)
3. Optimizer steps and gradient updates
4. Data collators and batching utilities
5. Reward models (PRM/ORM)

The architecture is **complete for inference** but **incomplete for training**. The training code initializes models but doesn't use them - it just generates mock metrics for demonstration purposes.
