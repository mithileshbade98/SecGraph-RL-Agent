"""
Example: Training and using LoRA adapters for SecGraph-RL-Agent

This example demonstrates:
1. Training a PPO adapter with LoRA
2. Training a DPO adapter with LoRA
3. Loading and using trained adapters for inference
4. Using preprocessing and postprocessing utilities

Prerequisites:
- Base model available (e.g., "meta-llama/Llama-2-7b-hf" from HuggingFace)
- GPU recommended for training
- CUDA installed (optional, will fallback to CPU)
"""

from pathlib import Path
from reason_agent.rl.ppo import PPOTrainer
from reason_agent.rl.dpo import DPOTrainer
from reason_agent.reasoning.planner import ReasoningPlanner
from reason_agent.tools.registry import ToolRegistry
from reason_agent.tools.router import ToolRouter
from reason_agent.tools.executors import ExecutorRegistry
from loguru import logger


def example_ppo_training():
    """Example 1: Train a PPO adapter with LoRA."""
    logger.info("=" * 60)
    logger.info("Example 1: PPO Training with LoRA")
    logger.info("=" * 60)

    # Initialize trainer with base model
    # Option 1: Use a local model path
    # base_model = "/path/to/local/llama-2-7b"

    # Option 2: Use HuggingFace model (will download on first run)
    base_model = "meta-llama/Llama-2-7b-hf"  # Requires HF token for Llama
    # Or use an open model like: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Option 3: Run without actual model (simulation mode)
    # base_model = None

    trainer = PPOTrainer(
        config_path=Path("configs/rl/ppo.yaml"),
        base_model=base_model,
        initialize_model=True if base_model else False,
    )

    # Train
    logger.info("Starting PPO training...")
    metrics = trainer.train(
        num_episodes=100,
        save_path=Path("artifacts/models/ppo_adapter"),
    )

    logger.info(f"Training completed!")
    logger.info(f"Final average reward: {metrics['final_avg_reward']:.3f}")
    logger.info(f"Adapter saved to: artifacts/models/ppo_adapter")


def example_dpo_training():
    """Example 2: Train a DPO adapter with LoRA."""
    logger.info("=" * 60)
    logger.info("Example 2: DPO Training with LoRA")
    logger.info("=" * 60)

    # Initialize trainer with base model
    base_model = "meta-llama/Llama-2-7b-hf"  # Or use TinyLlama for testing
    # base_model = None  # Simulation mode

    trainer = DPOTrainer(
        config_path=Path("configs/rl/dpo.yaml"),
        base_model=base_model,
        initialize_model=True if base_model else False,
    )

    # Train
    logger.info("Starting DPO training...")
    metrics = trainer.train(
        data_path=Path("data/audits/pairs.jsonl"),  # Will create mock data if missing
        num_epochs=50,
        save_path=Path("artifacts/models/dpo_adapter"),
    )

    logger.info(f"Training completed!")
    logger.info(f"Final accuracy: {metrics['final_accuracy']:.3f}")
    logger.info(f"Adapter saved to: artifacts/models/dpo_adapter")


def example_inference_with_lora():
    """Example 3: Use trained LoRA adapter for inference."""
    logger.info("=" * 60)
    logger.info("Example 3: Inference with LoRA Adapter")
    logger.info("=" * 60)

    # Set up planner components
    registry = ToolRegistry()
    router = ToolRouter(registry, method="rule")
    executors = ExecutorRegistry()

    # Initialize planner with LoRA adapter
    # Use the PPO-trained adapter
    adapter_path = Path("artifacts/models/ppo_adapter")

    if not adapter_path.exists():
        logger.warning(f"Adapter not found at {adapter_path}")
        logger.info("Run PPO training first or set adapter_path to None for heuristic planning")
        adapter_path = None

    planner = ReasoningPlanner(
        tool_registry=registry,
        tool_router=router,
        executor_registry=executors,
        use_llm=True,
        lora_adapter_path=adapter_path,
        base_model="meta-llama/Llama-2-7b-hf",  # Must match training base model
    )

    # Test queries
    queries = [
        "Detect multi-account abuse from the same device",
        "Check for velocity violations in user logins",
        "Identify burst patterns in account signups",
    ]

    for query in queries:
        logger.info(f"\nQuery: {query}")
        trace = planner.plan_and_execute(query)

        logger.info(f"Success: {trace['success']}")
        logger.info(f"Number of steps: {trace['num_steps']}")

        for step in trace['steps']:
            logger.info(f"  Step {step['step_id']}: {step['thought']}")
            logger.info(f"    Tool: {step['tool']}")


def example_preprocessing_postprocessing():
    """Example 4: Using preprocessing and postprocessing utilities."""
    logger.info("=" * 60)
    logger.info("Example 4: Preprocessing and Postprocessing")
    logger.info("=" * 60)

    from reason_agent.rl.lora_utils import (
        preprocess_text,
        preprocess_batch,
        postprocess_output,
    )
    from transformers import AutoTokenizer
    import torch

    # Load tokenizer (can use any model's tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Example 1: Preprocess single text
    text = "Detect anomalies in user behavior patterns"
    inputs = preprocess_text(
        text=text,
        tokenizer=tokenizer,
        max_length=128,
    )

    logger.info(f"Input text: {text}")
    logger.info(f"Input shape: {inputs['input_ids'].shape}")
    logger.info(f"Input IDs (first 10): {inputs['input_ids'][0][:10].tolist()}")

    # Example 2: Preprocess batch
    texts = [
        "Check for multi-account abuse",
        "Verify rate limiting policies",
        "Detect temporal bursts",
    ]

    batch_inputs = preprocess_batch(
        texts=texts,
        tokenizer=tokenizer,
        max_length=128,
    )

    logger.info(f"\nBatch size: {len(texts)}")
    logger.info(f"Batch input shape: {batch_inputs['input_ids'].shape}")

    # Example 3: Postprocess output
    # Simulate model output
    output_ids = torch.tensor([[1, 2, 3, 4, 5]])
    decoded = postprocess_output(
        output_ids=output_ids,
        tokenizer=tokenizer,
        skip_special_tokens=True,
    )

    logger.info(f"\nDecoded output: {decoded}")


def example_adapter_info():
    """Example 5: Inspect saved adapter information."""
    logger.info("=" * 60)
    logger.info("Example 5: Inspect Adapter Information")
    logger.info("=" * 60)

    import json

    adapters = [
        Path("artifacts/models/ppo_adapter"),
        Path("artifacts/models/dpo_adapter"),
    ]

    for adapter_path in adapters:
        info_file = adapter_path / "adapter_info.json"

        if info_file.exists():
            with open(info_file) as f:
                info = json.load(f)

            logger.info(f"\nAdapter: {adapter_path.name}")
            logger.info(f"  Base model: {info.get('base_model', 'N/A')}")
            logger.info(f"  LoRA rank: {info.get('r', 'N/A')}")
            logger.info(f"  LoRA alpha: {info.get('lora_alpha', 'N/A')}")
            logger.info(f"  Target modules: {info.get('target_modules', 'N/A')}")
        else:
            logger.warning(f"No adapter info found at {adapter_path}")


def main():
    """Run all examples."""
    logger.info("SecGraph-RL-Agent: LoRA Training and Inference Examples")
    logger.info("=" * 60)

    # Example 1: PPO Training
    # Uncomment to run:
    # example_ppo_training()

    # Example 2: DPO Training
    # Uncomment to run:
    # example_dpo_training()

    # Example 3: Inference with LoRA
    # Requires trained adapter from examples 1 or 2
    # Uncomment to run:
    # example_inference_with_lora()

    # Example 4: Preprocessing and Postprocessing
    example_preprocessing_postprocessing()

    # Example 5: Inspect Adapters
    example_adapter_info()

    logger.info("\n" + "=" * 60)
    logger.info("Examples completed!")
    logger.info("=" * 60)
    logger.info("\nTo run training examples:")
    logger.info("1. Set a base_model in the functions above")
    logger.info("2. Uncomment the example you want to run in main()")
    logger.info("3. Run: python examples/lora_training_example.py")


if __name__ == "__main__":
    main()
