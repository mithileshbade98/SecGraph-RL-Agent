"""
Learning rate schedulers for training.

Provides various LR schedules: cosine annealing, linear warmup, polynomial decay, etc.
"""

import math
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from loguru import logger


class CosineAnnealingWithWarmup(_LRScheduler):
    """
    Cosine annealing schedule with linear warmup.

    LR increases linearly during warmup, then decreases following cosine curve.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        Initialize scheduler.

        Args:
            optimizer: Optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate
            last_epoch: Last epoch (for resuming)
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

        logger.info(
            f"Cosine annealing with warmup: {warmup_steps} warmup steps, "
            f"{total_steps} total steps, min_lr={min_lr}"
        )

    def get_lr(self):
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class LinearWarmupLinearDecay(_LRScheduler):
    """
    Linear warmup followed by linear decay.

    Simple but effective schedule.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        Initialize scheduler.

        Args:
            optimizer: Optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate
            last_epoch: Last epoch
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

        logger.info(
            f"Linear warmup + decay: {warmup_steps} warmup, "
            f"{total_steps} total steps, min_lr={min_lr}"
        )

    def get_lr(self):
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return [
                base_lr * (1 - progress) + self.min_lr * progress
                for base_lr in self.base_lrs
            ]


class PolynomialDecayWithWarmup(_LRScheduler):
    """
    Polynomial decay with linear warmup.

    LR = (base_lr - min_lr) * (1 - progress)^power + min_lr
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        power: float = 1.0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        Initialize scheduler.

        Args:
            optimizer: Optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            power: Polynomial power (1.0 = linear, 2.0 = quadratic)
            min_lr: Minimum learning rate
            last_epoch: Last epoch
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.power = power
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

        logger.info(
            f"Polynomial decay (power={power}) with warmup: "
            f"{warmup_steps} warmup, {total_steps} total"
        )

    def get_lr(self):
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Polynomial decay
            progress = (self.last_epoch - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return [
                (base_lr - self.min_lr) * (1 - progress) ** self.power + self.min_lr
                for base_lr in self.base_lrs
            ]


class ConstantWithWarmup(_LRScheduler):
    """
    Constant LR with linear warmup.

    Useful when you don't want decay but still want warmup.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        """
        Initialize scheduler.

        Args:
            optimizer: Optimizer
            warmup_steps: Number of warmup steps
            last_epoch: Last epoch
        """
        self.warmup_steps = warmup_steps

        super().__init__(optimizer, last_epoch)

        logger.info(f"Constant LR with {warmup_steps} warmup steps")

    def get_lr(self):
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Constant
            return self.base_lrs


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    warmup_steps: int = 0,
    total_steps: Optional[int] = None,
    **kwargs,
) -> _LRScheduler:
    """
    Get scheduler by name.

    Args:
        name: Scheduler name
            - 'cosine': Cosine annealing with warmup
            - 'linear': Linear warmup + linear decay
            - 'polynomial': Polynomial decay with warmup
            - 'constant': Constant with warmup
        optimizer: Optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps (required for some schedulers)
        **kwargs: Additional arguments for scheduler

    Returns:
        LR scheduler

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = get_scheduler(
        ...     'cosine',
        ...     optimizer,
        ...     warmup_steps=100,
        ...     total_steps=1000,
        ...     min_lr=1e-7
        ... )
    """
    name = name.lower()

    if name == 'cosine':
        if total_steps is None:
            raise ValueError("total_steps required for cosine scheduler")
        return CosineAnnealingWithWarmup(
            optimizer, warmup_steps, total_steps, **kwargs
        )

    elif name == 'linear':
        if total_steps is None:
            raise ValueError("total_steps required for linear scheduler")
        return LinearWarmupLinearDecay(
            optimizer, warmup_steps, total_steps, **kwargs
        )

    elif name == 'polynomial':
        if total_steps is None:
            raise ValueError("total_steps required for polynomial scheduler")
        return PolynomialDecayWithWarmup(
            optimizer, warmup_steps, total_steps, **kwargs
        )

    elif name == 'constant':
        return ConstantWithWarmup(optimizer, warmup_steps, **kwargs)

    else:
        raise ValueError(
            f"Unknown scheduler: {name}. "
            f"Choose from: cosine, linear, polynomial, constant"
        )


# Example usage
if __name__ == "__main__":
    import torch

    # Mock model and optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create scheduler
    scheduler = get_scheduler(
        'cosine',
        optimizer,
        warmup_steps=100,
        total_steps=1000,
        min_lr=1e-6
    )

    # Simulate training
    lrs = []
    for step in range(1000):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    print(f"Initial LR: {lrs[0]:.2e}")
    print(f"LR after warmup (step 100): {lrs[100]:.2e}")
    print(f"LR at middle (step 500): {lrs[500]:.2e}")
    print(f"Final LR (step 999): {lrs[999]:.2e}")
