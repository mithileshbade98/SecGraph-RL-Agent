"""
PPO (Proximal Policy Optimization) training for reasoning.

Why: Stable policy gradients with verifiable rewards.
Source: "RL with Verifiable Rewards" - arXiv:2410.15246
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import yaml
import json
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
import torch.nn.functional as F
from reason_agent.rl.training_utils import (
    compute_gae,
    compute_ppo_loss,
    compute_value_loss,
    compute_log_probs,
    compute_kl_divergence,
    clip_gradients,
    whiten_advantages,
    RewardComputer,
)
from reason_agent.rl.data_collators import PPODataCollator
from reason_agent.rl.value_head import ModelWithValueHead
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


class PPOTrainer:
    """PPO trainer for reasoning agents."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        base_model: Optional[str] = None,
        initialize_model: bool = True,
        checkpoint_dir: Optional[Path] = None,
        use_value_head: bool = True,
    ):
        """
        Initialize PPO trainer.

        Args:
            config_path: Path to ppo.yaml config
            base_model: Base model name/path (if None, use from config)
            initialize_model: Whether to initialize the PEFT model immediately (default: True)
            checkpoint_dir: Directory for checkpoints (default: artifacts/checkpoints/ppo)
            use_value_head: Whether to use actual value head
        """
        if config_path is None:
            config_path = Path("configs/rl/ppo.yaml")

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config.get('model', {})
        self.training_config = self.config.get('training', {})
        self.reward_config = self.config.get('rewards', {})
        self.peft_config = self.model_config.get('peft_config', {})
        self.advanced_config = self.config.get('advanced', {})

        # Distributed training setup
        self.use_distributed = self.advanced_config.get('use_distributed', False)
        self.distributed_config = None
        if self.use_distributed:
            self.distributed_config = setup_distributed()
            logger.info(f"Distributed training enabled: rank {self.distributed_config.rank}/{self.distributed_config.world_size}")

        # Device detection and configuration (CUDA, MPS, or CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.use_gpu = True
            self.backend = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.success(f"🚀 NVIDIA GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            logger.info(f"Mixed precision: {'ENABLED (2x faster)' if self.advanced_config.get('use_mixed_precision') else 'disabled'}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.use_gpu = True
            self.backend = "mps"
            logger.success(f"🍎 Apple Silicon GPU detected (Metal Performance Shaders)")
            logger.info("MPS acceleration ENABLED (2-3x faster than CPU on M1/M2/M3)")
            logger.info("Mixed precision not supported on MPS, using float32")
        else:
            self.device = torch.device("cpu")
            self.use_gpu = False
            self.backend = "cpu"
            logger.warning("⚠️  No GPU detected - using CPU")
            logger.info("For faster training:")
            logger.info("  - Mac M1/M2/M3: Run natively (not Docker) to use Metal GPU")
            logger.info("  - Linux/Windows: Install NVIDIA GPU + nvidia-docker")

        # Model configuration
        self.base_model_name = base_model or self.model_config.get('base_model')
        self.model = None
        self.tokenizer = None
        self.use_value_head = use_value_head

        logger.info("PPO trainer initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Learning rate: {self.training_config.get('learning_rate')}")
        logger.info(f"Batch size: {self.training_config.get('batch_size')}")
        logger.info(f"PPO epochs: {self.training_config.get('ppo_epochs')}")
        logger.info(f"PEFT method: {self.peft_config.get('method', 'lora')}")
        logger.info(f"Value head: {'enabled' if use_value_head else 'disabled (mock values)'}")

        # Initialize model if requested and base model is specified
        if initialize_model and self.base_model_name:
            self._initialize_model()

        # Initialize optimizer (if model exists)
        self.optimizer = None
        if self.model is not None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.training_config.get('learning_rate', 1e-5),
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
            checkpoint_dir = Path("artifacts/checkpoints/ppo")
        self.checkpointer = TrainingCheckpointer(
            checkpoint_dir=checkpoint_dir,
            keep_last_n=self.training_config.get('keep_last_n_checkpoints', 3),
            save_best=True,
        )

        # Initialize evaluator (if model exists)
        self.evaluator = None
        if self.model is not None and self.tokenizer is not None:
            self.evaluator = ModelEvaluator(
                model=self.model,
                tokenizer=self.tokenizer,
                batch_size=self.training_config.get('eval_batch_size', 4),
            )

    def _initialize_model(self) -> None:
        """Initialize PEFT model with LoRA adapters and optional value head."""
        if not self.base_model_name:
            logger.warning("No base model specified, skipping model initialization")
            return

        logger.info(f"Initializing PEFT model from {self.base_model_name}")

        try:
            base_model, self.tokenizer = initialize_peft_model(
                base_model_name=self.base_model_name,
                peft_config=self.peft_config,
                device=None,  # Auto-detect
                load_in_8bit=self.training_config.get('load_in_8bit', False),
            )

            # Wrap with value head if enabled
            if self.use_value_head:
                logger.info("Wrapping model with value head")
                hidden_size = base_model.config.hidden_size
                self.model = ModelWithValueHead(
                    model=base_model,
                    hidden_size=hidden_size,
                    value_head_dropout=self.training_config.get('value_head_dropout', 0.1),
                    value_head_layers=self.training_config.get('value_head_layers', 2),
                )
                logger.success("Model with value head initialized")
            else:
                self.model = base_model
                logger.success("PEFT model initialized (no value head)")

            # Wrap with DistributedDataParallel if distributed training enabled
            if self.use_distributed and is_distributed():
                logger.info("Wrapping model with DistributedDataParallel")
                self.model = wrap_model_ddp(
                    self.model,
                    find_unused_parameters=self.advanced_config.get('find_unused_parameters', False),
                )
                logger.success("Model wrapped with DDP")

        except Exception as e:
            logger.error(f"Failed to initialize PEFT model: {e}")
            raise RuntimeError(
                f"Model initialization failed: {e}\n"
                "Make sure the model is downloaded and accessible. "
                "Run: python3 scripts/download_model.py"
            )

    def train(
        self,
        num_episodes: int = 100,
        save_path: Optional[Path] = None,
        training_data: Optional[list] = None,
        val_data: Optional[list] = None,
        resume_from_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """
        Train PPO on verifiable math tasks.

        Args:
            num_episodes: Number of training episodes
            save_path: Path to save trained adapter
            training_data: Optional training data (list of dicts with query/response/reward)
            val_data: Optional validation data for evaluation
            resume_from_checkpoint: Whether to resume from latest checkpoint if available

        Returns:
            Training metrics
        """
        if self.model is None or self.optimizer is None:
            raise RuntimeError(
                "Model not initialized! PPO trainer requires a model to train. "
                "Make sure base_model is configured in ppo.yaml and the model downloads successfully."
            )

        logger.info(f"Starting PPO training for {num_episodes} episodes...")

        # Try to resume from checkpoint
        start_episode = 0
        if resume_from_checkpoint:
            checkpoint = self.checkpointer.resume_from_latest(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
            if checkpoint is not None:
                start_episode = checkpoint.get('epoch', 0) + 1
                logger.info(f"Resumed from episode {start_episode}")

        logger.info("🔥 Using ACTUAL PPO training with model updates")
        return self._train_actual(num_episodes, save_path, training_data, val_data, start_episode)

    def _train_actual(
        self,
        num_episodes: int,
        save_path: Optional[Path],
        training_data: Optional[list] = None,
        val_data: Optional[list] = None,
        start_episode: int = 0,
    ) -> Dict[str, Any]:
        """
        Actual PPO training with real model updates.

        Args:
            num_episodes: Number of training episodes
            save_path: Path to save adapter
            training_data: Training data (queries and expected outputs)
            val_data: Validation data for evaluation
            start_episode: Episode to start from (for resuming)

        Returns:
            Training metrics
        """
        logger.info("🔥 Starting ACTUAL PPO training with gradient updates")

        # Training hyperparameters
        ppo_epochs = self.training_config.get('ppo_epochs', 4)
        clip_range = self.training_config.get('clip_range', 0.2)
        value_coef = self.training_config.get('value_coef', 0.5)
        entropy_coef = self.training_config.get('entropy_coef', 0.01)
        gamma = self.reward_config.get('gamma', 0.99)
        gae_lambda = self.reward_config.get('gae_lambda', 0.95)
        max_grad_norm = self.training_config.get('max_grad_norm', 1.0)
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

        # Initialize reward computer
        reward_computer = RewardComputer()

        # Metrics tracking
        mean_rewards = []
        policy_losses = []
        value_losses = []
        kl_divs = []
        val_accuracies = []
        best_val_accuracy = 0.0

        # Create mock training data if none provided
        if training_data is None:
            training_data = self._create_mock_training_data(num_samples=50)

        # Gradient accumulation counter
        accumulation_counter = 0

        # Training loop
        for episode in range(start_episode, num_episodes):
            # Sample a batch from training data
            batch_size = min(4, len(training_data))
            batch_indices = np.random.choice(len(training_data), batch_size, replace=False)
            batch = [training_data[i] for i in batch_indices]

            # Generate rollouts
            rollouts = self._generate_rollouts(batch)

            # Store old log probs for PPO
            with torch.no_grad():
                old_log_probs = self._compute_rollout_log_probs(rollouts)

            # Compute advantages using GAE
            advantages, returns = self._compute_advantages_gae(
                rollouts, gamma=gamma, gae_lambda=gae_lambda
            )

            # Whiten advantages
            advantages = whiten_advantages(advantages)

            # PPO update epochs
            epoch_policy_losses = []
            epoch_value_losses = []
            epoch_kl_divs = []

            for ppo_epoch in range(ppo_epochs):
                # Only zero gradients at the start of accumulation cycle
                if accumulation_counter % gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad()

                # Forward pass with mixed precision
                if scaler is not None:
                    from torch.cuda.amp import autocast
                    with autocast():
                        # Forward pass
                        outputs = self._forward_rollouts(rollouts)

                        # Compute new log probs
                        new_log_probs = outputs['log_probs']
                        values = outputs['values']
                        logits = outputs['logits']

                        # Compute PPO loss
                        policy_loss = compute_ppo_loss(
                            log_probs=new_log_probs,
                            old_log_probs=old_log_probs,
                            advantages=advantages,
                            clip_range=clip_range,
                        )

                        # Compute value loss
                        value_loss = compute_value_loss(values, returns)

                        # Compute entropy for exploration
                        from reason_agent.rl.training_utils import compute_entropy
                        entropy = compute_entropy(logits)

                        # Total loss
                        loss = (
                            policy_loss
                            + value_coef * value_loss
                            - entropy_coef * entropy
                        )

                        # Normalize loss by accumulation steps
                        loss = loss / gradient_accumulation_steps

                    # Backward pass with scaled gradients
                    scaler.scale(loss).backward()
                else:
                    # Forward pass (no mixed precision)
                    outputs = self._forward_rollouts(rollouts)

                    # Compute new log probs
                    new_log_probs = outputs['log_probs']
                    values = outputs['values']
                    logits = outputs['logits']

                    # Compute PPO loss
                    policy_loss = compute_ppo_loss(
                        log_probs=new_log_probs,
                        old_log_probs=old_log_probs,
                        advantages=advantages,
                        clip_range=clip_range,
                    )

                    # Compute value loss
                    value_loss = compute_value_loss(values, returns)

                    # Compute entropy for exploration
                    from reason_agent.rl.training_utils import compute_entropy
                    entropy = compute_entropy(logits)

                    # Total loss
                    loss = (
                        policy_loss
                        + value_coef * value_loss
                        - entropy_coef * entropy
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
                        grad_norm = clip_gradients(self.model, max_grad_norm)

                        # Optimizer step with scaler
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        # Clip gradients
                        grad_norm = clip_gradients(self.model, max_grad_norm)

                        # Optimizer step
                        self.optimizer.step()

                    # Scheduler step (only when we actually update)
                    if self.scheduler is not None:
                        self.scheduler.step()

                # Compute KL for monitoring
                with torch.no_grad():
                    kl = torch.abs(new_log_probs - old_log_probs).mean()

                # Store metrics (denormalize loss for logging)
                epoch_policy_losses.append(policy_loss.item() * gradient_accumulation_steps)
                epoch_value_losses.append(value_loss.item() * gradient_accumulation_steps)
                epoch_kl_divs.append(kl.item())

            # Compute episode rewards
            episode_rewards = [r['reward'] for r in rollouts]
            mean_reward = np.mean(episode_rewards)

            # Store metrics
            mean_rewards.append(mean_reward)
            policy_losses.append(np.mean(epoch_policy_losses))
            value_losses.append(np.mean(epoch_value_losses))
            kl_divs.append(np.mean(epoch_kl_divs))

            # Run validation
            if val_data is not None and self.evaluator is not None and (episode + 1) % eval_freq == 0:
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
            if (episode + 1) % checkpoint_freq == 0:
                logger.info(f"Saving checkpoint at episode {episode + 1}")
                checkpoint_metrics = {
                    'mean_reward': mean_reward,
                    'policy_loss': policy_losses[-1],
                    'value_loss': value_losses[-1],
                    'kl_divergence': kl_divs[-1],
                }
                if val_accuracies:
                    checkpoint_metrics['val_accuracy'] = val_accuracies[-1]

                self.checkpointer.save_checkpoint(
                    epoch=episode,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics=checkpoint_metrics,
                    is_best=is_best,
                )

            if (episode + 1) % 10 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Reward: {mean_reward:.3f} | "
                    f"Policy Loss: {policy_losses[-1]:.3f} | "
                    f"Value Loss: {value_losses[-1]:.3f} | "
                    f"KL: {kl_divs[-1]:.4f} | "
                    f"LR: {lr:.2e}"
                )

        return self._save_training_metrics(
            mean_rewards, policy_losses, value_losses, kl_divs,
            num_episodes, save_path
        )

    def _save_training_metrics(
        self,
        mean_rewards: list,
        policy_losses: list,
        value_losses: list,
        kl_divs: list,
        num_episodes: int,
        save_path: Optional[Path],
    ) -> Dict[str, Any]:
        """Save training metrics and adapter."""

        # Save metrics for UI to display
        metrics_dir = Path("artifacts/runs/ppo")
        metrics_dir.mkdir(parents=True, exist_ok=True)

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'num_episodes': num_episodes,
            'final_avg_reward': float(np.mean(mean_rewards[-10:])),  # Last 10 episodes
            'initial_reward': float(mean_rewards[0]),
            'max_reward': float(max(mean_rewards)),
            'mean_rewards': [float(r) for r in mean_rewards],
            'policy_losses': [float(l) for l in policy_losses],
            'value_losses': [float(l) for l in value_losses],
            'kl_divergences': [float(k) for k in kl_divs],
            'config': {
                'learning_rate': self.training_config.get('learning_rate'),
                'clip_range': self.training_config.get('clip_range'),
                'ppo_epochs': self.training_config.get('ppo_epochs'),
            }
        }

        # Save metrics
        metrics_file = metrics_dir / f"ppo_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved to {metrics_file}")

        # Save adapter
        if save_path and self.model is not None and self.tokenizer is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                save_lora_adapter(
                    model=self.model,
                    save_path=save_path,
                    tokenizer=self.tokenizer,
                )
                logger.success(f"LoRA adapter saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save LoRA adapter: {e}")

        return metrics

    def _create_mock_training_data(self, num_samples: int = 50) -> list:
        """Create mock training data for PPO."""
        data = []
        queries = [
            "Detect multi-account abuse from the same device",
            "Check for velocity violations in login attempts",
            "Identify burst patterns in account creation",
            "Analyze suspicious fund transfer patterns",
            "Detect coordinated fraud behavior across accounts",
        ]

        for i in range(num_samples):
            query = queries[i % len(queries)]
            data.append({
                'query': query,
                'expected_output': f"Step 1: Analyze pattern. Step 2: Check policy. Step 3: Verify anomaly.",
                'is_correct': random.random() > 0.3,  # 70% correct
            })

        return data

    def _generate_rollouts(self, batch: list) -> list:
        """
        Generate rollouts by running model on queries.

        Args:
            batch: List of training examples

        Returns:
            List of rollouts with responses and rewards
        """
        rollouts = []

        for example in batch:
            query = example['query']

            # Generate response
            try:
                response = generate_with_lora(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=query,
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.9,
                )
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                response = "Step 1: Error occurred"

            # Compute reward
            reward_computer = RewardComputer()
            reward = reward_computer.compute_reward(
                output=response,
                expected_output=example.get('expected_output'),
                is_correct=example.get('is_correct'),
            )

            rollouts.append({
                'query': query,
                'response': response,
                'reward': reward,
                'expected_output': example.get('expected_output'),
            })

        return rollouts

    def _compute_rollout_log_probs(self, rollouts: list) -> torch.Tensor:
        """Compute log probabilities for rollouts."""
        log_probs_list = []

        for rollout in rollouts:
            query = rollout['query']
            response = rollout['response']

            # Tokenize
            full_text = query + " " + response
            inputs = preprocess_text(full_text, self.tokenizer, max_length=512)

            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Compute log probs for response tokens
            response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
            query_len = len(self.tokenizer.encode(query, add_special_tokens=False))

            # Extract log probs for response
            response_logits = logits[0, query_len:query_len + len(response_tokens), :]
            log_probs = F.log_softmax(response_logits, dim=-1)

            # Get log probs for actual tokens
            token_log_probs = []
            for i, token_id in enumerate(response_tokens):
                if i < log_probs.shape[0]:
                    token_log_probs.append(log_probs[i, token_id].item())

            # Pad to fixed length
            max_len = 128
            while len(token_log_probs) < max_len:
                token_log_probs.append(0.0)
            token_log_probs = token_log_probs[:max_len]

            log_probs_list.append(token_log_probs)

        return torch.tensor(log_probs_list, dtype=torch.float32)

    def _compute_advantages_gae(
        self,
        rollouts: list,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages using Generalized Advantage Estimation."""
        batch_size = len(rollouts)
        seq_len = 128

        # Extract rewards
        rewards = torch.tensor([r['reward'] for r in rollouts], dtype=torch.float32)

        # Expand to sequence
        rewards_seq = rewards.unsqueeze(-1).expand(-1, seq_len)

        # Get value estimates
        if self.use_value_head and isinstance(self.model, ModelWithValueHead):
            # Use actual value head
            texts = [r['query'] + " " + r['response'] for r in rollouts]
            inputs = preprocess_batch(texts, self.tokenizer, max_length=512)

            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass to get values
            with torch.no_grad():
                outputs = self.model(**inputs)
                values = outputs['values']  # (batch_size, seq_len)

                # Pad or truncate to seq_len
                if values.shape[1] < seq_len:
                    padding = torch.zeros(
                        batch_size, seq_len - values.shape[1],
                        device=device
                    )
                    values = torch.cat([values, padding], dim=1)
                else:
                    values = values[:, :seq_len]

                # Move to CPU for GAE computation
                values = values.cpu()

            next_values = torch.roll(values, -1, dims=1)
        else:
            # Mock value estimates (fallback)
            values = torch.randn(batch_size, seq_len) * 0.1 + rewards.unsqueeze(-1)
            next_values = torch.roll(values, -1, dims=1)

        # Done flags (episode ends)
        dones = torch.zeros(batch_size, seq_len)
        dones[:, -1] = 1.0

        # Compute GAE
        advantages, returns = compute_gae(
            rewards=rewards_seq,
            values=values,
            next_values=next_values,
            dones=dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        return advantages, returns

    def _forward_rollouts(self, rollouts: list) -> Dict[str, torch.Tensor]:
        """Forward pass through model for rollouts."""
        batch_size = len(rollouts)
        seq_len = 128

        # Tokenize all rollouts
        texts = [r['query'] + " " + r['response'] for r in rollouts]
        inputs = preprocess_batch(texts, self.tokenizer, max_length=512)

        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        if self.use_value_head and isinstance(self.model, ModelWithValueHead):
            # Use model with value head
            outputs = self.model(**inputs)
            logits = outputs['logits']
            values = outputs['values']
        else:
            # Regular model
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Mock value estimates
            values = torch.randn(batch_size, logits.shape[1], device=device) * 0.1

        # Compute log probs (simplified)
        log_probs = F.log_softmax(logits, dim=-1)

        # For this simplified version, use mean log prob
        mean_log_probs = log_probs.mean(dim=-1)[:, :seq_len]

        # Pad to seq_len if needed
        if mean_log_probs.shape[1] < seq_len:
            padding = torch.zeros(
                batch_size, seq_len - mean_log_probs.shape[1],
                device=device
            )
            mean_log_probs = torch.cat([mean_log_probs, padding], dim=1)
        else:
            mean_log_probs = mean_log_probs[:, :seq_len]

        # Pad values to seq_len if needed
        if values.shape[1] < seq_len:
            padding = torch.zeros(
                batch_size, seq_len - values.shape[1],
                device=device
            )
            values = torch.cat([values, padding], dim=1)
        else:
            values = values[:, :seq_len]

        return {
            'log_probs': mean_log_probs,
            'values': values,
            'logits': logits,
        }


def main():
    """CLI for PPO training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO on verifiable tasks")
    parser.add_argument("--config", default="configs/rl/ppo.yaml", help="Config path")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--save-path", default="artifacts/models/ppo_adapter", help="Save path")
    args = parser.parse_args()

    trainer = PPOTrainer(config_path=Path(args.config))
    metrics = trainer.train(num_episodes=args.episodes, save_path=Path(args.save_path))

    logger.info(f"\nTraining complete!")
    logger.info(f"Final average reward: {metrics['final_avg_reward']:.3f}")


if __name__ == "__main__":
    main()
