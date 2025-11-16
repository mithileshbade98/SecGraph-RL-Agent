"""
Post-training utilities for model optimization and deployment.

Includes:
- Model quantization (4-bit, 8-bit)
- Adapter merging
- Model pruning
- Export to different formats
- Deployment optimization
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import json


class PostTrainingOptimizer:
    """
    Optimize trained models for deployment.

    Supports:
    - Quantization (4-bit, 8-bit)
    - Adapter merging
    - Model compression
    - Export to different formats
    """

    def __init__(self, model_path: Path, adapter_path: Optional[Path] = None):
        """
        Initialize post-training optimizer.

        Args:
            model_path: Path to base model or merged model
            adapter_path: Optional path to LoRA adapter
        """
        self.model_path = Path(model_path)
        self.adapter_path = Path(adapter_path) if adapter_path else None

        logger.info(f"PostTrainingOptimizer initialized with model: {model_path}")

    def merge_adapter(
        self,
        output_path: Path,
        save_merged: bool = True,
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
    ) -> Any:
        """
        Merge LoRA adapter with base model.

        Args:
            output_path: Path to save merged model
            save_merged: Whether to save the merged model
            push_to_hub: Whether to push to HuggingFace Hub
            hub_model_id: Model ID for Hub (e.g., "username/model-name")

        Returns:
            Merged model
        """
        if self.adapter_path is None:
            logger.error("No adapter path provided, cannot merge")
            return None

        logger.info("Loading base model and adapter for merging...")

        # Load adapter config
        peft_config = PeftConfig.from_pretrained(self.adapter_path)

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.float16,
            device_map='auto',
        )

        # Load PEFT model
        model = PeftModel.from_pretrained(base_model, self.adapter_path)

        logger.info("Merging adapter with base model...")

        # Merge adapter weights into base model
        merged_model = model.merge_and_unload()

        logger.success("Adapter merged successfully")

        # Save merged model
        if save_merged:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving merged model to {output_path}")

            merged_model.save_pretrained(output_path)

            # Also save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
            tokenizer.save_pretrained(output_path)

            logger.success(f"Merged model saved to {output_path}")

        # Push to Hub if requested
        if push_to_hub and hub_model_id:
            logger.info(f"Pushing merged model to Hub: {hub_model_id}")
            merged_model.push_to_hub(hub_model_id)
            tokenizer.push_to_hub(hub_model_id)
            logger.success(f"Model pushed to {hub_model_id}")

        return merged_model

    def quantize_4bit(
        self,
        output_path: Path,
        compute_dtype: str = 'bfloat16',
        quant_type: str = 'nf4',
        double_quant: bool = True,
    ) -> Any:
        """
        Quantize model to 4-bit (QLoRA).

        Args:
            output_path: Path to save quantized model
            compute_dtype: Computation dtype ('bfloat16' or 'float16')
            quant_type: Quantization type ('nf4' or 'fp4')
            double_quant: Whether to use double quantization

        Returns:
            Quantized model
        """
        logger.info("Loading model for 4-bit quantization...")

        # Determine compute dtype
        if compute_dtype == 'bfloat16':
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=double_quant,
        )

        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map='auto',
        )

        logger.success("Model quantized to 4-bit")

        # Save quantized model
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(output_path)

        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.save_pretrained(output_path)

        logger.success(f"Quantized model saved to {output_path}")

        return model

    def quantize_8bit(
        self,
        output_path: Path,
    ) -> Any:
        """
        Quantize model to 8-bit.

        Args:
            output_path: Path to save quantized model

        Returns:
            Quantized model
        """
        logger.info("Loading model for 8-bit quantization...")

        # Load model with 8-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            load_in_8bit=True,
            device_map='auto',
        )

        logger.success("Model quantized to 8-bit")

        # Save quantized model
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(output_path)

        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.save_pretrained(output_path)

        logger.success(f"Quantized model saved to {output_path}")

        return model

    def export_to_onnx(
        self,
        output_path: Path,
        example_input_length: int = 128,
        opset_version: int = 14,
    ):
        """
        Export model to ONNX format.

        Args:
            output_path: Path to save ONNX model
            example_input_length: Length of example input for tracing
            opset_version: ONNX opset version
        """
        try:
            import torch.onnx

            logger.info("Loading model for ONNX export...")

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,  # ONNX requires float32
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            model.eval()

            # Create example input
            example_text = "This is an example input for ONNX export."
            inputs = tokenizer(
                example_text,
                max_length=example_input_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )

            # Export to ONNX
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Exporting model to ONNX: {output_path}")

            torch.onnx.export(
                model,
                (inputs['input_ids'], inputs['attention_mask']),
                output_path,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size', 1: 'sequence_length'},
                },
                opset_version=opset_version,
            )

            logger.success(f"Model exported to ONNX: {output_path}")

        except ImportError:
            logger.error("ONNX export requires torch.onnx. Install with: pip install onnx")

    def optimize_for_inference(
        self,
        output_path: Path,
        use_flash_attention: bool = True,
        use_bettertransformer: bool = True,
    ):
        """
        Optimize model for faster inference.

        Args:
            output_path: Path to save optimized model
            use_flash_attention: Whether to enable FlashAttention
            use_bettertransformer: Whether to use BetterTransformer
        """
        logger.info("Loading model for inference optimization...")

        # Load model
        model_kwargs = {
            'torch_dtype': torch.float16,
            'device_map': 'auto',
        }

        if use_flash_attention:
            model_kwargs['use_flash_attention_2'] = True
            logger.info("Enabling FlashAttention 2")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs,
        )

        # Apply BetterTransformer if requested
        if use_bettertransformer:
            try:
                from optimum.bettertransformer import BetterTransformer

                logger.info("Applying BetterTransformer optimization")
                model = BetterTransformer.transform(model)
            except ImportError:
                logger.warning("BetterTransformer not available. Install with: pip install optimum")

        # Save optimized model
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(output_path)

        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.save_pretrained(output_path)

        logger.success(f"Optimized model saved to {output_path}")

    def create_deployment_package(
        self,
        output_dir: Path,
        include_config: bool = True,
        include_examples: bool = True,
    ):
        """
        Create a complete deployment package.

        Args:
            output_dir: Directory to save deployment package
            include_config: Whether to include config files
            include_examples: Whether to include example usage
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating deployment package in {output_dir}")

        # Copy model files (if not already there)
        model_dir = output_dir / "model"
        if not model_dir.exists():
            logger.info("Copying model files...")
            # Here you would copy the model files
            # For now, just create the directory
            model_dir.mkdir(parents=True, exist_ok=True)

        # Create config
        if include_config:
            config = {
                'model_type': 'causal_lm',
                'model_path': str(self.model_path),
                'adapter_path': str(self.adapter_path) if self.adapter_path else None,
                'deployment_config': {
                    'max_length': 512,
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 50,
                },
            }

            config_path = output_dir / "deployment_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Config saved to {config_path}")

        # Create example usage script
        if include_examples:
            example_script = '''"""
Example usage of deployed model.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("model/")
tokenizer = AutoTokenizer.from_pretrained("model/")

# Example query
query = "Detect multi-account abuse from the same device"

# Tokenize
inputs = tokenizer(query, return_tensors="pt")

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )

# Decode
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
'''

            example_path = output_dir / "example_usage.py"
            with open(example_path, 'w') as f:
                f.write(example_script)

            logger.info(f"Example script saved to {example_path}")

        # Create README
        readme_content = f"""# Model Deployment Package

## Contents

- `model/` - Model files
- `deployment_config.json` - Deployment configuration
- `example_usage.py` - Example usage script

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("model/")
tokenizer = AutoTokenizer.from_pretrained("model/")

# Your code here
```

## Model Information

- Base Model: {self.model_path}
- Adapter: {self.adapter_path if self.adapter_path else 'None'}

## Requirements

```
torch>=2.0.0
transformers>=4.36.0
peft>=0.7.0
```
"""

        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)

        logger.success(f"Deployment package created in {output_dir}")


def merge_and_quantize(
    base_model: str,
    adapter_path: Path,
    output_path: Path,
    quantization: str = '4bit',
):
    """
    Convenience function to merge adapter and quantize in one step.

    Args:
        base_model: Base model name or path
        adapter_path: Path to LoRA adapter
        output_path: Path to save merged and quantized model
        quantization: Quantization type ('4bit', '8bit', or 'none')
    """
    # First merge
    optimizer = PostTrainingOptimizer(base_model, adapter_path)

    merged_path = output_path / "merged"
    merged_model = optimizer.merge_adapter(merged_path, save_merged=True)

    # Then quantize
    if quantization == '4bit':
        quantized_path = output_path / "quantized_4bit"
        optimizer.model_path = merged_path
        optimizer.quantize_4bit(quantized_path)
    elif quantization == '8bit':
        quantized_path = output_path / "quantized_8bit"
        optimizer.model_path = merged_path
        optimizer.quantize_8bit(quantized_path)

    logger.success(f"Merged and quantized model saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Example: Merge adapter
    optimizer = PostTrainingOptimizer(
        model_path="meta-llama/Llama-2-7b-hf",
        adapter_path="artifacts/models/ppo_adapter",
    )

    # Merge adapter with base model
    merged_model = optimizer.merge_adapter(
        output_path="artifacts/models/merged_model",
        save_merged=True,
    )

    # Quantize to 4-bit
    optimizer.model_path = "artifacts/models/merged_model"
    optimizer.quantize_4bit(
        output_path="artifacts/models/merged_model_4bit",
        compute_dtype='bfloat16',
    )

    # Create deployment package
    optimizer.create_deployment_package(
        output_dir="artifacts/deployment",
        include_config=True,
        include_examples=True,
    )

    print("Post-training optimization complete!")
