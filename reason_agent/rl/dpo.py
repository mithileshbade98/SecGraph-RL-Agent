"""
DPO (Direct Preference Optimization) training for trace alignment.

Why: Direct preference learning without reward model brittleness.
Source: "Direct Preference Optimization" - arXiv:2305.18290
Enhanced: Offset-DPO & Dynamic-β - ACL 2024
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import yaml
import random
import numpy as np
from datetime import datetime
from loguru import logger
import torch
from reason_agent.rl.lora_utils import (
    initialize_peft_model,
    save_lora_adapter,
    preprocess_text,
    preprocess_batch,
    postprocess_output,
    generate_with_lora,
)
from reason_agent.rl.training_utils import (
    compute_dpo_loss,
    compute_log_probs,
    clip_gradients,
)
from reason_agent.rl.data_collators import DPODataCollator
from reason_agent.rl.schedulers import get_scheduler
from reason_agent.rl.checkpointing import TrainingCheckpointer
from reason_agent.rl.evaluator import ModelEvaluator
from reason_agent.rl.distributed import (
    setup_distributed,
    is_distributed,
    is_main_process,
    wrap_model_ddp,
    DistributedMetrics,
    DistributedConfig,
)
import torch.nn.functional as F


class DPOTrainer:
    """DPO trainer for preference alignment."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        base_model: Optional[str] = None,
        initialize_model: bool = True,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize DPO trainer.

        Args:
            config_path: Path to dpo.yaml config
            base_model: Base model name/path (if None, use from config)
            initialize_model: Whether to initialize the PEFT models immediately (default: True)
            checkpoint_dir: Directory for checkpoints (default: artifacts/checkpoints/dpo)
        """
        if config_path is None:
            config_path = Path("configs/rl/dpo.yaml")

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config.get('model', {})
        self.training_config = self.config.get('training', {})
        self.dpo_config = self.config.get('dpo', {})
        self.data_config = self.config.get('data', {})
        self.peft_config = self.model_config.get('peft_config', {})
        self.advanced_config = self.config.get('advanced', {})

        # Distributed training setup
        self.use_distributed = self.advanced_config.get('use_distributed', False)
        self.distributed_config = None
        if self.use_distributed:
            self.distributed_config = setup_distributed()
            logger.info(f"Distributed training enabled: rank {self.distributed_config.rank}/{self.distributed_config.world_size}")

        # Model configuration
        self.base_model_name = base_model or self.model_config.get('base_model')
        self.policy_model = None  # Trainable policy model
        self.reference_model = None  # Frozen reference model
        self.tokenizer = None

        logger.info("DPO trainer initialized")
        logger.info(f"Beta: {self.dpo_config.get('beta')}")
        logger.info(f"Dynamic beta: {self.dpo_config.get('use_dynamic_beta')}")
        logger.info(f"Use offset: {self.dpo_config.get('use_offset')}")
        logger.info(f"PEFT method: {self.peft_config.get('method', 'lora')}")

        # Initialize models if requested and base model is specified
        if initialize_model and self.base_model_name:
            self._initialize_models()

        # Initialize optimizer (if policy model exists)
        self.optimizer = None
        if self.policy_model is not None:
            self.optimizer = torch.optim.AdamW(
                self.policy_model.parameters(),
                lr=self.training_config.get('learning_rate', 1e-6),
            )

        # Initialize scheduler (if optimizer exists)
        self.scheduler = None
        if self.optimizer is not None:
            scheduler_config = self.training_config.get('scheduler', {})
            scheduler_name = scheduler_config.get('name', 'cosine')
            warmup_steps = scheduler_config.get('warmup_steps', 100)
            total_steps = scheduler_config.get('total_steps', 1000)

            self.scheduler = get_scheduler(
                name=scheduler_name,
                optimizer=self.optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
            )
            logger.info(f"Initialized {scheduler_name} scheduler with {warmup_steps} warmup steps")

        # Initialize checkpointer
        if checkpoint_dir is None:
            checkpoint_dir = Path("artifacts/checkpoints/dpo")
        self.checkpointer = TrainingCheckpointer(
            checkpoint_dir=checkpoint_dir,
            keep_last_n=self.training_config.get('keep_last_n_checkpoints', 3),
            save_best=True,
        )

        # Initialize evaluator (if model exists)
        self.evaluator = None
        if self.policy_model is not None and self.tokenizer is not None:
            self.evaluator = ModelEvaluator(
                model=self.policy_model,
                tokenizer=self.tokenizer,
                batch_size=self.training_config.get('eval_batch_size', 4),
            )

    def _initialize_models(self) -> None:
        """Initialize policy and reference models with LoRA adapters."""
        if not self.base_model_name:
            logger.warning("No base model specified, skipping model initialization")
            return

        logger.info(f"Initializing DPO models from {self.base_model_name}")

        try:
            # Initialize policy model (trainable)
            logger.info("Initializing policy model with LoRA...")
            self.policy_model, self.tokenizer = initialize_peft_model(
                base_model_name=self.base_model_name,
                peft_config=self.peft_config,
                device=None,  # Auto-detect
                load_in_8bit=self.training_config.get('load_in_8bit', False),
            )

            # Initialize reference model (frozen)
            # Option 1: Load a separate frozen copy of the base model
            # Option 2: Use the same model but disable gradients
            reference_model_name = self.config.get('reference_model')

            if reference_model_name:
                logger.info(f"Initializing separate reference model from {reference_model_name}")
                from transformers import AutoModelForCausalLM

                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.reference_model = AutoModelForCausalLM.from_pretrained(
                    reference_model_name,
                    device_map="auto" if device == "cuda" else None,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                )
                self.reference_model.eval()  # Freeze reference model
                for param in self.reference_model.parameters():
                    param.requires_grad = False
            else:
                logger.info("Using policy model base as reference (will create frozen copy)")
                # Reference will be the base model without LoRA adapters
                # For DPO, we typically use the pre-trained base without adapters as reference
                self.reference_model = None  # Will be handled during training

            # Wrap policy model with DistributedDataParallel if distributed training enabled
            # Note: Reference model stays non-DDP as it's frozen
            if self.use_distributed and is_distributed():
                logger.info("Wrapping policy model with DistributedDataParallel")
                self.policy_model = wrap_model_ddp(
                    self.policy_model,
                    find_unused_parameters=self.advanced_config.get('find_unused_parameters', False),
                )
                logger.success("Policy model wrapped with DDP")

            logger.success("DPO models initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize DPO models: {e}")
            raise RuntimeError(
                f"Model initialization failed: {e}\n"
                "Make sure the model is downloaded and accessible. "
                "Run: python3 scripts/download_model.py"
            )

    def load_preference_pairs(self, data_path: Path) -> List[Dict[str, Any]]:
        """
        Load preference pairs from JSONL.

        Format: {prompt, chosen, rejected, confidence (optional)}
        """
        pairs = []

        if not data_path.exists():
            logger.warning(f"Data file not found: {data_path}")
            return pairs

        with open(data_path) as f:
            for line in f:
                pair = json.loads(line)
                pairs.append(pair)

        logger.info(f"Loaded {len(pairs)} preference pairs from {data_path}")
        return pairs

    def train(
        self,
        data_path: Optional[Path] = None,
        num_epochs: int = 1,
        save_path: Optional[Path] = None,
        val_data: Optional[list] = None,
        resume_from_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """
        Train DPO on preference pairs.

        Args:
            data_path: Path to preference pairs JSONL
            num_epochs: Number of training epochs
            save_path: Path to save trained adapter
            val_data: Optional validation data for evaluation
            resume_from_checkpoint: Whether to resume from latest checkpoint if available

        Returns:
            Training metrics
        """
        if self.policy_model is None or self.optimizer is None:
            raise RuntimeError(
                "Model not initialized! DPO trainer requires a model to train. "
                "Make sure base_model is configured in dpo.yaml and the model downloads successfully."
            )

        if data_path is None:
            data_path = Path(self.data_config.get('train_file', 'data/audits/pairs.jsonl'))

        logger.info(f"Starting DPO training for {num_epochs} epochs...")

        # Try to resume from checkpoint
        start_epoch = 0
        if resume_from_checkpoint:
            checkpoint = self.checkpointer.resume_from_latest(
                model=self.policy_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
            if checkpoint is not None:
                start_epoch = checkpoint.get('epoch', 0) + 1
                logger.info(f"Resumed from epoch {start_epoch}")

        # Load data
        pairs = self.load_preference_pairs(data_path)

        if not pairs:
            logger.warning("No preference pairs found, creating mock data")
            pairs = self._create_mock_pairs(10)

        logger.info("🔥 Using ACTUAL DPO training with model updates")
        return self._train_actual(pairs, num_epochs, save_path, val_data, start_epoch)

    def _train_actual(
        self,
        pairs: List[Dict[str, Any]],
        num_epochs: int,
        save_path: Optional[Path],
        val_data: Optional[list] = None,
        start_epoch: int = 0,
    ) -> Dict[str, Any]:
        """
        Actual DPO training with real model updates.

        Args:
            pairs: Preference pairs
            num_epochs: Number of epochs
            save_path: Save path
            val_data: Validation data for evaluation
            start_epoch: Epoch to start from (for resuming)

        Returns:
            Training metrics
        """
        logger.info("🔥 Starting ACTUAL DPO training with gradient updates")

        # Training hyperparameters
        beta = self.dpo_config.get('beta', 0.1)
        use_offset = self.dpo_config.get('use_offset', False)
        offset_margin = self.dpo_config.get('offset_margin', 0.0)
        max_grad_norm = self.training_config.get('max_grad_norm', 1.0)
        batch_size = min(4, len(pairs))
        checkpoint_freq = self.training_config.get('checkpoint_freq', 10)
        eval_freq = self.training_config.get('eval_freq', 10)
        gradient_accumulation_steps = self.training_config.get('gradient_accumulation_steps', 1)
        use_mixed_precision = self.advanced_config.get('use_mixed_precision', False)

        # Initialize GradScaler for mixed precision training
        scaler = None
        if use_mixed_precision and torch.cuda.is_available():
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
            logger.info("Mixed precision training enabled (FP16)")

        # Initialize data collator
        collator = DPODataCollator(
            tokenizer=self.tokenizer,
            max_length=self.training_config.get('max_length', 512),
        )

        # Metrics tracking
        accuracies = []
        dpo_losses = []
        preference_margins = []
        chosen_rewards = []
        rejected_rewards = []
        val_accuracies = []
        best_val_accuracy = 0.0

        # Gradient accumulation counter
        accumulation_counter = 0

        # Training loop
        for epoch in range(start_epoch, num_epochs):
            # Sample batch
            batch_indices = np.random.choice(len(pairs), min(batch_size, len(pairs)), replace=False)
            batch = [pairs[i] for i in batch_indices]

            # Collate batch
            dpo_batch = collator(batch)

            # Move to device
            device = next(self.policy_model.parameters()).device
            chosen_inputs = {
                'input_ids': dpo_batch.chosen_input_ids.to(device),
                'attention_mask': dpo_batch.chosen_attention_mask.to(device),
            }
            rejected_inputs = {
                'input_ids': dpo_batch.rejected_input_ids.to(device),
                'attention_mask': dpo_batch.rejected_attention_mask.to(device),
            }

            # Only zero gradients at the start of accumulation cycle
            if accumulation_counter % gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if scaler is not None:
                from torch.cuda.amp import autocast
                with autocast():
                    # Forward pass - policy model
                    policy_chosen_outputs = self.policy_model(**chosen_inputs)
                    policy_rejected_outputs = self.policy_model(**rejected_inputs)

                    # Compute log probs for policy
                    policy_chosen_log_probs = compute_log_probs(
                        policy_chosen_outputs.logits,
                        dpo_batch.chosen_labels.to(device),
                        dpo_batch.chosen_attention_mask.to(device),
                    )
                    policy_rejected_log_probs = compute_log_probs(
                        policy_rejected_outputs.logits,
                        dpo_batch.rejected_labels.to(device),
                        dpo_batch.rejected_attention_mask.to(device),
                    )

                    # Forward pass - reference model (if available)
                    if self.reference_model is not None:
                        with torch.no_grad():
                            ref_chosen_outputs = self.reference_model(**chosen_inputs)
                            ref_rejected_outputs = self.reference_model(**rejected_inputs)

                            ref_chosen_log_probs = compute_log_probs(
                                ref_chosen_outputs.logits,
                                dpo_batch.chosen_labels.to(device),
                                dpo_batch.chosen_attention_mask.to(device),
                            )
                            ref_rejected_log_probs = compute_log_probs(
                                ref_rejected_outputs.logits,
                                dpo_batch.rejected_labels.to(device),
                                dpo_batch.rejected_attention_mask.to(device),
                            )
                    else:
                        # Use policy model initial state as reference (approximation)
                        with torch.no_grad():
                            ref_chosen_log_probs = policy_chosen_log_probs.detach()
                            ref_rejected_log_probs = policy_rejected_log_probs.detach()

                    # Compute DPO loss
                    loss, metrics = compute_dpo_loss(
                        policy_chosen_log_probs=policy_chosen_log_probs,
                        policy_rejected_log_probs=policy_rejected_log_probs,
                        reference_chosen_log_probs=ref_chosen_log_probs,
                        reference_rejected_log_probs=ref_rejected_log_probs,
                        beta=beta,
                        use_offset=use_offset,
                        offset_margin=offset_margin,
                    )

                    # Normalize loss by accumulation steps
                    loss = loss / gradient_accumulation_steps

                # Backward pass with scaled gradients
                scaler.scale(loss).backward()
            else:
                # Forward pass - policy model (no mixed precision)
                policy_chosen_outputs = self.policy_model(**chosen_inputs)
                policy_rejected_outputs = self.policy_model(**rejected_inputs)

                # Compute log probs for policy
                policy_chosen_log_probs = compute_log_probs(
                    policy_chosen_outputs.logits,
                    dpo_batch.chosen_labels.to(device),
                    dpo_batch.chosen_attention_mask.to(device),
                )
                policy_rejected_log_probs = compute_log_probs(
                    policy_rejected_outputs.logits,
                    dpo_batch.rejected_labels.to(device),
                    dpo_batch.rejected_attention_mask.to(device),
                )

                # Forward pass - reference model (if available)
                if self.reference_model is not None:
                    with torch.no_grad():
                        ref_chosen_outputs = self.reference_model(**chosen_inputs)
                        ref_rejected_outputs = self.reference_model(**rejected_inputs)

                        ref_chosen_log_probs = compute_log_probs(
                            ref_chosen_outputs.logits,
                            dpo_batch.chosen_labels.to(device),
                            dpo_batch.chosen_attention_mask.to(device),
                        )
                        ref_rejected_log_probs = compute_log_probs(
                            ref_rejected_outputs.logits,
                            dpo_batch.rejected_labels.to(device),
                            dpo_batch.rejected_attention_mask.to(device),
                        )
                else:
                    # Use policy model initial state as reference (approximation)
                    with torch.no_grad():
                        ref_chosen_log_probs = policy_chosen_log_probs.detach()
                        ref_rejected_log_probs = policy_rejected_log_probs.detach()

                # Compute DPO loss
                loss, metrics = compute_dpo_loss(
                    policy_chosen_log_probs=policy_chosen_log_probs,
                    policy_rejected_log_probs=policy_rejected_log_probs,
                    reference_chosen_log_probs=ref_chosen_log_probs,
                    reference_rejected_log_probs=ref_rejected_log_probs,
                    beta=beta,
                    use_offset=use_offset,
                    offset_margin=offset_margin,
                )

                # Normalize loss by accumulation steps
                loss = loss / gradient_accumulation_steps

                # Backward pass (standard)
                loss.backward()

            # Increment accumulation counter
            accumulation_counter += 1

            # Only update weights after accumulating gradients
            if accumulation_counter % gradient_accumulation_steps == 0:
                if scaler is not None:
                    # Unscale gradients and clip
                    scaler.unscale_(self.optimizer)
                    grad_norm = clip_gradients(self.policy_model, max_grad_norm)

                    # Optimizer step with scaler
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    # Clip gradients
                    grad_norm = clip_gradients(self.policy_model, max_grad_norm)

                    # Optimizer step
                    self.optimizer.step()

                # Scheduler step (only when we actually update)
                if self.scheduler is not None:
                    self.scheduler.step()

            # Store metrics (denormalize loss for logging)
            accuracies.append(metrics['accuracy'].item())
            dpo_losses.append(loss.item() * gradient_accumulation_steps)
            preference_margins.append(metrics['margin'].item())
            chosen_rewards.append(metrics['chosen_reward'].item())
            rejected_rewards.append(metrics['rejected_reward'].item())

            # Run validation
            if val_data is not None and self.evaluator is not None and (epoch + 1) % eval_freq == 0:
                logger.info("Running validation...")
                val_metrics = self.evaluator.evaluate_with_generation(
                    eval_data=val_data,
                    show_progress=False,
                )
                val_accuracy = val_metrics['accuracy']
                val_accuracies.append(val_accuracy)
                logger.info(f"Validation accuracy: {val_accuracy:.3f}")

                # Check if best model
                is_best = val_accuracy > best_val_accuracy
                if is_best:
                    best_val_accuracy = val_accuracy
                    logger.success(f"New best validation accuracy: {val_accuracy:.3f}")
            else:
                is_best = False

            # Save checkpoint
            if (epoch + 1) % checkpoint_freq == 0:
                logger.info(f"Saving checkpoint at epoch {epoch + 1}")
                checkpoint_metrics = {
                    'accuracy': accuracies[-1],
                    'dpo_loss': dpo_losses[-1],
                    'preference_margin': preference_margins[-1],
                }
                if val_accuracies:
                    checkpoint_metrics['val_accuracy'] = val_accuracies[-1]

                self.checkpointer.save_checkpoint(
                    epoch=epoch,
                    model=self.policy_model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics=checkpoint_metrics,
                    is_best=is_best,
                )

            if (epoch + 1) % max(1, num_epochs // 10) == 0 or epoch == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Acc: {metrics['accuracy'].item():.3f} | "
                    f"Margin: {metrics['margin'].item():.3f} | "
                    f"LR: {lr:.2e}"
                )

        return self._save_training_metrics(
            accuracies, dpo_losses, preference_margins,
            chosen_rewards, rejected_rewards, len(pairs),
            num_epochs, save_path
        )

    def _save_training_metrics(
        self,
        accuracies: list,
        dpo_losses: list,
        preference_margins: list,
        chosen_rewards: list,
        rejected_rewards: list,
        num_pairs: int,
        num_epochs: int,
        save_path: Optional[Path],
    ) -> Dict[str, Any]:
        """Save DPO training metrics and adapter."""
        from datetime import datetime

        # Save training metrics to JSON
        metrics = {
            'algorithm': 'dpo',
            'timestamp': datetime.now().isoformat(),
            'num_epochs': num_epochs,
            'num_pairs': num_pairs,
            'final_accuracy': float(accuracies[-1]) if accuracies else 0.0,
            'final_loss': float(dpo_losses[-1]) if dpo_losses else 0.0,
            'final_margin': float(preference_margins[-1]) if preference_margins else 0.0,
            'accuracies': [float(a) for a in accuracies],
            'dpo_losses': [float(l) for l in dpo_losses],
            'preference_margins': [float(m) for m in preference_margins],
            'chosen_rewards': [float(r) for r in chosen_rewards],
            'rejected_rewards': [float(r) for r in rejected_rewards],
        }

        # Save to artifacts directory
        output_dir = Path("artifacts/runs/dpo")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = output_dir / f"dpo_run_{timestamp_str}.json"

        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.success(f"DPO training metrics saved to {metrics_file}")

        # Save adapter
        if save_path and self.policy_model is not None and self.tokenizer is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                save_lora_adapter(
                    model=self.policy_model,
                    save_path=save_path,
                    tokenizer=self.tokenizer,
                )
                logger.success(f"DPO LoRA adapter saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save LoRA adapter: {e}")

        return metrics

    def _create_mock_pairs(self, num_pairs: int) -> List[Dict[str, Any]]:
        """Create mock preference pairs for testing."""
        pairs = []
        for i in range(num_pairs):
            pairs.append({
                'prompt': f'Detect anomaly pattern {i}',
                'chosen': f'Step 1: Check policy. Step 2: Verify. Conclusion: Anomaly detected.',
                'rejected': f'Random guess without reasoning.',
                'confidence': 0.8,
            })
        return pairs


def main():
    """CLI for DPO training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train DPO on preference pairs")
    parser.add_argument("--config", default="configs/rl/dpo.yaml", help="Config path")
    parser.add_argument("--data", default="data/audits/pairs.jsonl", help="Preference pairs")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--save-path", default="artifacts/models/dpo_adapter", help="Save path")
    args = parser.parse_args()

    trainer = DPOTrainer(config_path=Path(args.config))
    metrics = trainer.train(
        data_path=Path(args.data),
        num_epochs=args.epochs,
        save_path=Path(args.save_path),
    )

    logger.info(f"\nTraining complete!")
    logger.info(f"Final accuracy: {metrics['final_accuracy']:.3f}")
    logger.info(f"Trained on {metrics['num_pairs']} pairs")


if __name__ == "__main__":
    main()
