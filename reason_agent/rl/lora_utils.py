"""
LoRA utilities for parameter-efficient fine-tuning.

Provides functions for:
- Initializing PEFT models with LoRA configuration
- Saving and loading LoRA adapters
- Merging adapters into base models
- Tokenization and decoding utilities
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)
from loguru import logger


def create_lora_config(peft_config: Dict[str, Any]) -> LoraConfig:
    """
    Create LoRA configuration from config dict.

    Args:
        peft_config: Dictionary with LoRA parameters
            - r: LoRA rank
            - lora_alpha: Scaling factor
            - lora_dropout: Dropout probability
            - target_modules: List of modules to apply LoRA
            - bias: Bias handling ("none", "all", "lora_only")
            - task_type: Task type (e.g., "CAUSAL_LM")

    Returns:
        LoraConfig object
    """
    config = LoraConfig(
        r=peft_config.get('r', 16),
        lora_alpha=peft_config.get('lora_alpha', 32),
        lora_dropout=peft_config.get('lora_dropout', 0.1),
        target_modules=peft_config.get('target_modules', ["q_proj", "v_proj"]),
        bias=peft_config.get('bias', "none"),
        task_type=peft_config.get('task_type', "CAUSAL_LM"),
    )

    logger.info(f"Created LoRA config: rank={config.r}, alpha={config.lora_alpha}, "
                f"dropout={config.lora_dropout}, targets={config.target_modules}")

    return config


def initialize_peft_model(
    base_model_name: str,
    peft_config: Dict[str, Any],
    device: Optional[str] = None,
    load_in_8bit: bool = False,
) -> tuple[PeftModel, PreTrainedTokenizer]:
    """
    Initialize a PEFT model with LoRA adapters.

    Args:
        base_model_name: HuggingFace model name or path
        peft_config: LoRA configuration dictionary
        device: Device to load model on ("cuda", "cpu", or None for auto)
        load_in_8bit: Whether to load in 8-bit precision

    Returns:
        Tuple of (PEFT model, tokenizer)
    """
    logger.info(f"Initializing PEFT model from {base_model_name}")

    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model_kwargs = {"device_map": "auto" if device == "cuda" else None}
    if load_in_8bit and device == "cuda":
        model_kwargs["load_in_8bit"] = True

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        **model_kwargs,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # Prepare for training if using quantization
    if load_in_8bit:
        base_model = prepare_model_for_kbit_training(base_model)

    # Create LoRA config
    lora_config = create_lora_config(peft_config)

    # Apply PEFT
    peft_model = get_peft_model(base_model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_percentage = 100 * trainable_params / total_params

    logger.info(f"PEFT model initialized:")
    logger.info(f"  Trainable params: {trainable_params:,} ({trainable_percentage:.2f}%)")
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Device: {device}")

    return peft_model, tokenizer


def save_lora_adapter(
    model: PeftModel,
    save_path: Union[str, Path],
    tokenizer: Optional[PreTrainedTokenizer] = None,
) -> None:
    """
    Save LoRA adapter weights.

    Args:
        model: PEFT model with LoRA adapters
        save_path: Path to save adapter
        tokenizer: Optional tokenizer to save alongside
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save adapter weights
    model.save_pretrained(save_path)
    logger.info(f"LoRA adapter saved to {save_path}")

    # Save tokenizer if provided
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
        logger.info(f"Tokenizer saved to {save_path}")

    # Save adapter config for reference
    adapter_config = {
        'base_model': model.peft_config['default'].base_model_name_or_path,
        'r': model.peft_config['default'].r,
        'lora_alpha': model.peft_config['default'].lora_alpha,
        'target_modules': model.peft_config['default'].target_modules,
    }

    import json
    config_file = save_path / "adapter_info.json"
    with open(config_file, 'w') as f:
        json.dump(adapter_config, f, indent=2)

    logger.success(f"Adapter configuration saved to {config_file}")


def load_lora_adapter(
    adapter_path: Union[str, Path],
    base_model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> tuple[PeftModel, PreTrainedTokenizer]:
    """
    Load a LoRA adapter and apply it to base model.

    Args:
        adapter_path: Path to saved adapter
        base_model_name: Base model name (if None, read from adapter config)
        device: Device to load on

    Returns:
        Tuple of (PEFT model with adapter, tokenizer)
    """
    adapter_path = Path(adapter_path)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    logger.info(f"Loading LoRA adapter from {adapter_path}")

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load adapter config to get base model if not provided
    if base_model_name is None:
        import json
        config_file = adapter_path / "adapter_config.json"
        if config_file.exists():
            with open(config_file) as f:
                adapter_config = json.load(f)
                base_model_name = adapter_config.get('base_model_name_or_path')

        if base_model_name is None:
            raise ValueError("base_model_name must be provided or found in adapter config")

    logger.info(f"Loading base model: {base_model_name}")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # Load PEFT model with adapter
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.success(f"LoRA adapter loaded successfully")

    return peft_model, tokenizer


def merge_adapter_to_base(
    peft_model: PeftModel,
    save_merged_path: Optional[Union[str, Path]] = None,
) -> PreTrainedModel:
    """
    Merge LoRA adapter weights into base model.

    This creates a single merged model for faster inference.

    Args:
        peft_model: PEFT model with LoRA adapter
        save_merged_path: Optional path to save merged model

    Returns:
        Merged base model
    """
    logger.info("Merging LoRA adapter into base model...")

    # Merge adapter weights
    merged_model = peft_model.merge_and_unload()

    logger.success("Adapter merged successfully")

    # Save if path provided
    if save_merged_path:
        save_merged_path = Path(save_merged_path)
        save_merged_path.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(save_merged_path)
        logger.info(f"Merged model saved to {save_merged_path}")

    return merged_model


# Preprocessing and Postprocessing Utilities

def preprocess_text(
    text: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    add_special_tokens: bool = True,
    return_tensors: str = "pt",
) -> Dict[str, torch.Tensor]:
    """
    Preprocess text for model input.

    Args:
        text: Input text
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        add_special_tokens: Whether to add special tokens
        return_tensors: Format for return tensors ("pt" for PyTorch)

    Returns:
        Dictionary with input_ids, attention_mask, etc.
    """
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        add_special_tokens=add_special_tokens,
        return_tensors=return_tensors,
    )

    return inputs


def preprocess_batch(
    texts: list[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    add_special_tokens: bool = True,
    return_tensors: str = "pt",
) -> Dict[str, torch.Tensor]:
    """
    Preprocess a batch of texts.

    Args:
        texts: List of input texts
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        add_special_tokens: Whether to add special tokens
        return_tensors: Format for return tensors

    Returns:
        Dictionary with batched input_ids, attention_mask, etc.
    """
    inputs = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        add_special_tokens=add_special_tokens,
        return_tensors=return_tensors,
    )

    return inputs


def postprocess_output(
    output_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    skip_special_tokens: bool = True,
) -> Union[str, list[str]]:
    """
    Postprocess model output tokens to text.

    Args:
        output_ids: Output token IDs from model
        tokenizer: Tokenizer instance
        skip_special_tokens: Whether to skip special tokens in decoding

    Returns:
        Decoded text (string or list of strings for batch)
    """
    # Handle both single outputs and batches
    if output_ids.dim() == 1:
        # Single output
        text = tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens)
        return text
    else:
        # Batch output
        texts = tokenizer.batch_decode(output_ids, skip_special_tokens=skip_special_tokens)
        return texts


def generate_with_lora(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """
    Generate text using LoRA model.

    Args:
        model: PEFT model with LoRA
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling

    Returns:
        Generated text
    """
    # Preprocess input
    inputs = preprocess_text(prompt, tokenizer, return_tensors="pt")

    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Postprocess - extract only the generated part (skip input prompt)
    input_length = inputs['input_ids'].shape[1]
    generated_ids = outputs[0][input_length:]
    generated_text = postprocess_output(generated_ids, tokenizer)

    return generated_text
