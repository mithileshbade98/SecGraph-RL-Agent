"""
Distributed training utilities for multi-GPU and multi-node training.

Provides:
- Distributed initialization and cleanup
- DistributedDataParallel wrapper
- Gradient synchronization
- Rank and world size management
- Multi-node communication utilities
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Any
from loguru import logger
from pathlib import Path


class DistributedConfig:
    """Configuration for distributed training."""

    def __init__(
        self,
        backend: str = 'nccl',  # 'nccl' for GPU, 'gloo' for CPU
        init_method: str = 'env://',
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        master_addr: str = 'localhost',
        master_port: str = '29500',
    ):
        """
        Initialize distributed training configuration.

        Args:
            backend: Communication backend ('nccl', 'gloo', 'mpi')
            init_method: Initialization method ('env://', 'tcp://', 'file://')
            world_size: Total number of processes
            rank: Global rank of current process
            local_rank: Local rank on current node
            master_addr: Address of master node
            master_port: Port on master node
        """
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.master_addr = master_addr
        self.master_port = master_port

    @classmethod
    def from_env(cls) -> 'DistributedConfig':
        """
        Create config from environment variables.

        Reads standard distributed training env vars:
        - WORLD_SIZE: Total number of processes
        - RANK: Global rank of current process
        - LOCAL_RANK: Local rank on current node
        - MASTER_ADDR: Master node address
        - MASTER_PORT: Master node port
        """
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')

        # Determine backend based on device availability
        if torch.cuda.is_available():
            backend = 'nccl'
        else:
            backend = 'gloo'

        return cls(
            backend=backend,
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            master_addr=master_addr,
            master_port=master_port,
        )


def setup_distributed(config: Optional[DistributedConfig] = None) -> DistributedConfig:
    """
    Initialize distributed training environment.

    Args:
        config: Distributed configuration (if None, read from environment)

    Returns:
        DistributedConfig object
    """
    if config is None:
        config = DistributedConfig.from_env()

    # Set environment variables
    os.environ['MASTER_ADDR'] = config.master_addr
    os.environ['MASTER_PORT'] = config.master_port

    # Initialize process group
    if not dist.is_initialized():
        if config.world_size > 1:
            logger.info(
                f"Initializing distributed training: "
                f"backend={config.backend}, "
                f"world_size={config.world_size}, "
                f"rank={config.rank}, "
                f"local_rank={config.local_rank}"
            )

            dist.init_process_group(
                backend=config.backend,
                init_method=config.init_method,
                world_size=config.world_size,
                rank=config.rank,
            )

            # Set device for current process
            if torch.cuda.is_available():
                torch.cuda.set_device(config.local_rank)

            logger.success(
                f"Distributed training initialized on rank {config.rank}/{config.world_size}"
            )
        else:
            logger.info("Single process training (no distributed)")
    else:
        logger.warning("Distributed already initialized")

    return config


def cleanup_distributed():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed training cleaned up")


def is_distributed() -> bool:
    """Check if distributed training is active."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get global rank of current process."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    All-reduce operation across all processes.

    Args:
        tensor: Tensor to reduce
        op: Reduction operation (SUM, PRODUCT, MIN, MAX)

    Returns:
        Reduced tensor
    """
    if is_distributed():
        dist.all_reduce(tensor, op=op)
    return tensor


def all_gather(tensor: torch.Tensor) -> list:
    """
    Gather tensors from all processes.

    Args:
        tensor: Tensor to gather

    Returns:
        List of tensors from all processes
    """
    if not is_distributed():
        return [tensor]

    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast tensor from source process to all processes.

    Args:
        tensor: Tensor to broadcast
        src: Source rank

    Returns:
        Broadcasted tensor
    """
    if is_distributed():
        dist.broadcast(tensor, src=src)
    return tensor


def wrap_model_ddp(
    model: torch.nn.Module,
    device_ids: Optional[list] = None,
    output_device: Optional[int] = None,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True,
) -> torch.nn.Module:
    """
    Wrap model with DistributedDataParallel.

    Args:
        model: Model to wrap
        device_ids: List of device IDs (if None, uses local_rank)
        output_device: Output device (if None, uses local_rank)
        find_unused_parameters: Whether to find unused parameters
        gradient_as_bucket_view: Use gradient as bucket view for memory efficiency

    Returns:
        DDP-wrapped model (or original if not distributed)
    """
    if not is_distributed():
        logger.info("Not using DDP (single process)")
        return model

    # Get local rank
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # Set device IDs
    if device_ids is None and torch.cuda.is_available():
        device_ids = [local_rank]

    if output_device is None and torch.cuda.is_available():
        output_device = local_rank

    logger.info(
        f"Wrapping model with DDP: "
        f"device_ids={device_ids}, "
        f"output_device={output_device}, "
        f"find_unused_parameters={find_unused_parameters}"
    )

    # Wrap model
    ddp_model = DDP(
        model,
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
    )

    logger.success("Model wrapped with DDP")
    return ddp_model


def save_checkpoint_distributed(
    checkpoint: dict,
    checkpoint_path: Path,
    is_best: bool = False,
    only_main_process: bool = True,
):
    """
    Save checkpoint in distributed environment.

    Args:
        checkpoint: Checkpoint dictionary
        checkpoint_path: Path to save checkpoint
        is_best: Whether this is the best checkpoint
        only_main_process: Only save from main process (rank 0)
    """
    if only_main_process and not is_main_process():
        return

    # Ensure directory exists
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Save best checkpoint
    if is_best:
        best_path = checkpoint_path.parent / "best_checkpoint.pt"
        torch.save(checkpoint, best_path)
        logger.success(f"Best checkpoint saved: {best_path}")


def load_checkpoint_distributed(
    checkpoint_path: Path,
    map_location: Optional[str] = None,
) -> dict:
    """
    Load checkpoint in distributed environment.

    Args:
        checkpoint_path: Path to checkpoint
        map_location: Device to map checkpoint to

    Returns:
        Checkpoint dictionary
    """
    if map_location is None:
        if torch.cuda.is_available():
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            map_location = f'cuda:{local_rank}'
        else:
            map_location = 'cpu'

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    logger.info(f"Checkpoint loaded: {checkpoint_path}")

    return checkpoint


class DistributedMetrics:
    """Utilities for aggregating metrics across processes."""

    @staticmethod
    def aggregate_scalar(value: float, reduction: str = 'mean') -> float:
        """
        Aggregate scalar value across all processes.

        Args:
            value: Scalar value
            reduction: Reduction type ('mean', 'sum', 'max', 'min')

        Returns:
            Aggregated value
        """
        if not is_distributed():
            return value

        # Convert to tensor
        tensor = torch.tensor(value, dtype=torch.float32)
        if torch.cuda.is_available():
            tensor = tensor.cuda()

        # Reduce across processes
        if reduction == 'mean':
            all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor = tensor / get_world_size()
        elif reduction == 'sum':
            all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif reduction == 'max':
            all_reduce(tensor, op=dist.ReduceOp.MAX)
        elif reduction == 'min':
            all_reduce(tensor, op=dist.ReduceOp.MIN)
        else:
            raise ValueError(f"Unknown reduction type: {reduction}")

        return tensor.item()

    @staticmethod
    def aggregate_dict(
        metrics: dict, reduction: str = 'mean'
    ) -> dict:
        """
        Aggregate dictionary of metrics across all processes.

        Args:
            metrics: Dictionary of scalar metrics
            reduction: Reduction type ('mean', 'sum', 'max', 'min')

        Returns:
            Aggregated metrics
        """
        aggregated = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                aggregated[key] = DistributedMetrics.aggregate_scalar(value, reduction)
            else:
                # Keep non-scalar values unchanged
                aggregated[key] = value
        return aggregated


# Context manager for distributed training
class DistributedContext:
    """Context manager for distributed training."""

    def __init__(self, config: Optional[DistributedConfig] = None):
        """
        Initialize distributed context.

        Args:
            config: Distributed configuration
        """
        self.config = config

    def __enter__(self):
        """Enter context (setup distributed)."""
        self.config = setup_distributed(self.config)
        return self.config

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context (cleanup distributed)."""
        cleanup_distributed()


# Example usage
if __name__ == "__main__":
    # Example: Single-node multi-GPU training
    # Run with: torchrun --nproc_per_node=4 distributed.py

    with DistributedContext() as config:
        logger.info(f"Rank {get_rank()}/{get_world_size()}")

        # Create model
        model = torch.nn.Linear(10, 5)
        if torch.cuda.is_available():
            model = model.cuda()

        # Wrap with DDP
        model = wrap_model_ddp(model)

        # Example: Aggregate metrics
        local_loss = torch.rand(1).item()
        global_loss = DistributedMetrics.aggregate_scalar(local_loss, 'mean')

        if is_main_process():
            logger.info(f"Global average loss: {global_loss}")

        # Synchronize all processes
        barrier()

    logger.info("Distributed training complete")
