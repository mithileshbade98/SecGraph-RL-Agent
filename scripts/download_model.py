#!/usr/bin/env python3
"""
Download and cache the TinyLlama model for local use.
This ensures the model is available before training starts.
"""

import os
import sys
from pathlib import Path


class SimpleLogger:
    """Simple logger for console output."""

    @staticmethod
    def info(msg):
        print(f"ℹ️  {msg}")

    @staticmethod
    def success(msg):
        print(f"✓ {msg}")

    @staticmethod
    def error(msg):
        print(f"❌ {msg}", file=sys.stderr)


logger = SimpleLogger()


def download_model(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", cache_dir: str = "models"):
    """
    Download and cache model from HuggingFace.

    Args:
        model_name: HuggingFace model identifier
        cache_dir: Local directory to cache the model
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Downloading model: {model_name}")
        logger.info(f"Cache directory: {cache_dir}")

        # Create cache directory
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        logger.success(f"✓ Tokenizer downloaded ({len(tokenizer)} tokens)")

        # Download model
        logger.info("Downloading model weights (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            device_map="auto",
        )

        # Get model size
        param_count = sum(p.numel() for p in model.parameters())
        param_count_b = param_count / 1e9

        logger.success(f"✓ Model downloaded ({param_count_b:.2f}B parameters)")
        logger.success(f"✓ Model cached in: {cache_path.absolute()}")

        # Save model info
        info_file = cache_path / "model_info.txt"
        with open(info_file, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Parameters: {param_count_b:.2f}B\n")
            f.write(f"Cache directory: {cache_path.absolute()}\n")

        logger.info(f"Model info saved to: {info_file}")

        return True

    except ImportError:
        logger.error("transformers library not installed!")
        logger.error("Run: pip install transformers torch")
        return False
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False


def check_model_exists(cache_dir: str = "models") -> bool:
    """Check if model is already cached."""
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        return False

    # Check for huggingface cache
    has_cache = any(cache_path.rglob("*.bin")) or any(cache_path.rglob("*.safetensors"))

    return has_cache


def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("SecGraph-RL Agent - Model Download Script")
    logger.info("="*60)
    logger.info("")

    # Model configuration
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    cache_dir = "models"

    logger.info(f"Model: {model_name}")
    logger.info(f"  - Size: 1.1B parameters")
    logger.info(f"  - Type: Instruction-tuned chat model")
    logger.info(f"  - HuggingFace: https://huggingface.co/{model_name}")
    logger.info("")

    # Check if already cached
    if check_model_exists(cache_dir):
        logger.info(f"✓ Model already cached in: {Path(cache_dir).absolute()}")
        logger.info("Skipping download.")
        return 0

    # Download model
    logger.info("Model not found locally. Starting download...")
    logger.info("")

    success = download_model(model_name, cache_dir)

    if success:
        logger.info("")
        logger.success("="*60)
        logger.success("Model download complete!")
        logger.success("="*60)
        return 0
    else:
        logger.error("")
        logger.error("="*60)
        logger.error("Model download failed!")
        logger.error("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
