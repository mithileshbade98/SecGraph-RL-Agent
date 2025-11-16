"""
Checkpointing utilities for training.

Save and resume training from checkpoints.
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
from datetime import datetime


class TrainingCheckpointer:
    """
    Manages training checkpoints.

    Saves model, optimizer, scheduler, and training state.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        keep_last_n: int = 3,
        save_best: bool = True,
    ):
        """
        Initialize checkpointer.

        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
            save_best: Whether to save best checkpoint separately
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last_n = keep_last_n
        self.save_best = save_best

        self.best_metric = None
        self.checkpoints = []

        logger.info(f"Checkpointer initialized at {checkpoint_dir}")

    def save_checkpoint(
        self,
        epoch: int,
        model,
        optimizer,
        scheduler=None,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            epoch: Current epoch
            model: Model (can be PEFT model or ModelWithValueHead)
            optimizer: Optimizer
            scheduler: Optional LR scheduler
            metrics: Optional metrics dict
            extra_state: Optional extra state to save
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch_{epoch}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Prepare checkpoint dict
        checkpoint = {
            'epoch': epoch,
            'timestamp': timestamp,
            'metrics': metrics or {},
        }

        # Save model state
        if hasattr(model, 'state_dict'):
            checkpoint['model_state_dict'] = model.state_dict()
        else:
            logger.warning("Model has no state_dict(), skipping model state")

        # Save optimizer state
        if hasattr(optimizer, 'state_dict'):
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Save scheduler state
        if scheduler is not None and hasattr(scheduler, 'state_dict'):
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Save extra state
        if extra_state is not None:
            checkpoint['extra_state'] = extra_state

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Save metadata JSON
        metadata = {
            'epoch': epoch,
            'timestamp': timestamp,
            'checkpoint_path': str(checkpoint_path),
            'metrics': metrics or {},
        }

        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Track checkpoint
        self.checkpoints.append(checkpoint_path)

        # Save best if applicable
        if is_best and self.save_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            logger.success(f"Saved best checkpoint to {best_path}")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model,
        optimizer=None,
        scheduler=None,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore training state.

        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state

        Returns:
            Checkpoint dict with metadata
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load model state
        if 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.success("Loaded model state")
            except Exception as e:
                logger.error(f"Failed to load model state: {e}")

        # Load optimizer state
        if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.success("Loaded optimizer state")
            except Exception as e:
                logger.error(f"Failed to load optimizer state: {e}")

        # Load scheduler state
        if load_scheduler and scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.success("Loaded scheduler state")
            except Exception as e:
                logger.error(f"Failed to load scheduler state: {e}")

        logger.success(f"Restored training from epoch {checkpoint.get('epoch', 'unknown')}")

        return checkpoint

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only last N."""
        if self.keep_last_n <= 0:
            return

        # Sort by modification time
        all_checkpoints = sorted(
            [p for p in self.checkpoint_dir.glob("checkpoint_epoch_*.pt")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Keep last N, delete rest
        to_delete = all_checkpoints[self.keep_last_n:]

        for checkpoint_path in to_delete:
            try:
                checkpoint_path.unlink()
                # Also delete metadata
                metadata_path = checkpoint_path.with_suffix('.json')
                if metadata_path.exists():
                    metadata_path.unlink()

                logger.debug(f"Deleted old checkpoint: {checkpoint_path.name}")
            except Exception as e:
                logger.warning(f"Failed to delete {checkpoint_path}: {e}")

    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint."""
        all_checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))

        if not all_checkpoints:
            return None

        # Sort by modification time
        latest = max(all_checkpoints, key=lambda p: p.stat().st_mtime)

        logger.info(f"Found latest checkpoint: {latest}")

        return latest

    def find_best_checkpoint(self) -> Optional[Path]:
        """Find the best checkpoint."""
        best_path = self.checkpoint_dir / "checkpoint_best.pt"

        if best_path.exists():
            logger.info(f"Found best checkpoint: {best_path}")
            return best_path

        return None

    def resume_from_latest(
        self,
        model,
        optimizer=None,
        scheduler=None,
    ) -> Optional[Dict[str, Any]]:
        """
        Resume training from latest checkpoint if available.

        Args:
            model: Model
            optimizer: Optimizer
            scheduler: Scheduler

        Returns:
            Checkpoint dict if found, None otherwise
        """
        latest = self.find_latest_checkpoint()

        if latest is None:
            logger.info("No checkpoint found, starting from scratch")
            return None

        return self.load_checkpoint(
            latest,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )


def auto_resume(
    checkpoint_dir: Path,
    model,
    optimizer=None,
    scheduler=None,
) -> tuple[int, Optional[Dict]]:
    """
    Automatically resume from checkpoint if available.

    Args:
        checkpoint_dir: Checkpoint directory
        model: Model
        optimizer: Optimizer
        scheduler: Scheduler

    Returns:
        Tuple of (start_epoch, checkpoint_dict)
    """
    checkpointer = TrainingCheckpointer(checkpoint_dir)
    checkpoint = checkpointer.resume_from_latest(model, optimizer, scheduler)

    if checkpoint is not None:
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.success(f"Resumed from epoch {checkpoint.get('epoch')}, starting at {start_epoch}")
        return start_epoch, checkpoint
    else:
        logger.info("No checkpoint found, starting from epoch 0")
        return 0, None


# Example usage
if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim

    # Mock model and optimizer
    model = nn.Linear(10, 1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Create checkpointer
    checkpointer = TrainingCheckpointer(
        checkpoint_dir=Path("checkpoints/test"),
        keep_last_n=3,
    )

    # Save checkpoint
    checkpointer.save_checkpoint(
        epoch=5,
        model=model,
        optimizer=optimizer,
        metrics={'loss': 0.5, 'accuracy': 0.85},
    )

    # Resume
    checkpoint = checkpointer.resume_from_latest(model, optimizer)

    if checkpoint:
        print(f"Resumed from epoch {checkpoint['epoch']}")
        print(f"Metrics: {checkpoint['metrics']}")
