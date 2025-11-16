"""
Experiment tracking utilities for ML training.

Provides:
- Weights & Biases integration
- Metric logging and visualization
- Hyperparameter tracking
- Model artifact management
- Run comparison and analysis
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from loguru import logger
import json


class ExperimentTracker:
    """
    Unified experiment tracking interface.

    Supports Weights & Biases (wandb) with fallback to local logging.
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        enabled: bool = True,
        wandb_enabled: bool = True,
        log_dir: Optional[Path] = None,
    ):
        """
        Initialize experiment tracker.

        Args:
            project: Project name
            name: Run name (if None, auto-generated)
            config: Configuration/hyperparameters to log
            tags: Tags for run organization
            notes: Run description/notes
            enabled: Whether tracking is enabled
            wandb_enabled: Whether to use W&B (if False, local logging only)
            log_dir: Directory for local logs (if None, use artifacts/runs)
        """
        self.enabled = enabled
        self.wandb_enabled = wandb_enabled
        self.wandb_run = None

        if not self.enabled:
            logger.info("Experiment tracking disabled")
            return

        # Set up local logging directory
        if log_dir is None:
            log_dir = Path("artifacts/runs") / project
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Try to initialize W&B if enabled
        if self.wandb_enabled:
            try:
                import wandb

                self.wandb = wandb

                # Initialize run
                self.wandb_run = wandb.init(
                    project=project,
                    name=name,
                    config=config,
                    tags=tags,
                    notes=notes,
                    reinit=True,  # Allow multiple runs in same process
                )

                logger.success(f"W&B tracking initialized: {self.wandb_run.url}")

            except ImportError:
                logger.warning(
                    "Weights & Biases not installed. Install with: pip install wandb"
                )
                self.wandb_enabled = False
                self.wandb = None
            except Exception as e:
                logger.error(f"Failed to initialize W&B: {e}")
                logger.warning("Falling back to local logging only")
                self.wandb_enabled = False
                self.wandb = None
        else:
            logger.info("W&B tracking disabled, using local logging only")
            self.wandb = None

        # Local metrics storage
        self.metrics_file = self.log_dir / f"{name or 'run'}_metrics.jsonl"
        logger.info(f"Local metrics will be saved to {self.metrics_file}")

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics to log
            step: Step/epoch number (if None, auto-incremented)
            commit: Whether to commit immediately (W&B)
        """
        if not self.enabled:
            return

        # Log to W&B
        if self.wandb_enabled and self.wandb_run is not None:
            try:
                self.wandb.log(metrics, step=step, commit=commit)
            except Exception as e:
                logger.error(f"Failed to log to W&B: {e}")

        # Log locally
        try:
            log_entry = {"step": step, **metrics}
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write local metrics: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log scalar metrics.

        Args:
            metrics: Dictionary of scalar metrics
            step: Step number
        """
        self.log(metrics, step=step)

    def log_hyperparameters(self, params: Dict[str, Any]):
        """
        Log hyperparameters.

        Args:
            params: Dictionary of hyperparameters
        """
        if not self.enabled:
            return

        # Log to W&B config
        if self.wandb_enabled and self.wandb_run is not None:
            try:
                self.wandb_run.config.update(params)
            except Exception as e:
                logger.error(f"Failed to log hyperparameters to W&B: {e}")

        # Save locally
        try:
            params_file = self.log_dir / "hyperparameters.json"
            with open(params_file, 'w') as f:
                json.dump(params, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save hyperparameters locally: {e}")

    def log_artifact(
        self,
        artifact_path: Path,
        artifact_name: str,
        artifact_type: str = "model",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log artifact (model, dataset, etc.).

        Args:
            artifact_path: Path to artifact file/directory
            artifact_name: Name of artifact
            artifact_type: Type of artifact (model, dataset, etc.)
            metadata: Optional metadata
        """
        if not self.enabled:
            return

        # Log to W&B
        if self.wandb_enabled and self.wandb_run is not None:
            try:
                artifact = self.wandb.Artifact(
                    name=artifact_name,
                    type=artifact_type,
                    metadata=metadata or {},
                )

                if artifact_path.is_file():
                    artifact.add_file(str(artifact_path))
                else:
                    artifact.add_dir(str(artifact_path))

                self.wandb_run.log_artifact(artifact)
                logger.success(f"Artifact logged to W&B: {artifact_name}")

            except Exception as e:
                logger.error(f"Failed to log artifact to W&B: {e}")

        # Record locally
        try:
            artifacts_file = self.log_dir / "artifacts.jsonl"
            artifact_entry = {
                "name": artifact_name,
                "type": artifact_type,
                "path": str(artifact_path),
                "metadata": metadata or {},
            }
            with open(artifacts_file, 'a') as f:
                f.write(json.dumps(artifact_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to record artifact locally: {e}")

    def log_model(
        self,
        model_path: Path,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log model artifact.

        Args:
            model_path: Path to model file/directory
            model_name: Name of model
            metadata: Optional metadata (metrics, hyperparameters, etc.)
        """
        self.log_artifact(
            artifact_path=model_path,
            artifact_name=model_name,
            artifact_type="model",
            metadata=metadata,
        )

    def log_checkpoint(
        self,
        checkpoint_path: Path,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ):
        """
        Log training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            epoch: Epoch number
            metrics: Optional metrics at checkpoint
            is_best: Whether this is the best checkpoint
        """
        metadata = {
            "epoch": epoch,
            "is_best": is_best,
        }
        if metrics:
            metadata.update(metrics)

        checkpoint_name = f"checkpoint_epoch_{epoch}" + ("_best" if is_best else "")
        self.log_artifact(
            artifact_path=checkpoint_path,
            artifact_name=checkpoint_name,
            artifact_type="checkpoint",
            metadata=metadata,
        )

    def log_table(
        self,
        table_name: str,
        data: List[List[Any]],
        columns: Optional[List[str]] = None,
    ):
        """
        Log table data.

        Args:
            table_name: Name of table
            data: Table data (list of rows)
            columns: Column names
        """
        if not self.enabled:
            return

        # Log to W&B
        if self.wandb_enabled and self.wandb_run is not None:
            try:
                table = self.wandb.Table(data=data, columns=columns)
                self.wandb.log({table_name: table})
            except Exception as e:
                logger.error(f"Failed to log table to W&B: {e}")

        # Save locally as CSV
        try:
            import csv

            table_file = self.log_dir / f"{table_name}.csv"
            with open(table_file, 'w', newline='') as f:
                writer = csv.writer(f)
                if columns:
                    writer.writerow(columns)
                writer.writerows(data)
        except Exception as e:
            logger.error(f"Failed to save table locally: {e}")

    def log_histogram(
        self,
        name: str,
        values: List[float],
        step: Optional[int] = None,
    ):
        """
        Log histogram.

        Args:
            name: Histogram name
            values: Values to histogram
            step: Step number
        """
        if not self.enabled:
            return

        # Log to W&B
        if self.wandb_enabled and self.wandb_run is not None:
            try:
                import numpy as np

                self.wandb.log({name: self.wandb.Histogram(np.array(values))}, step=step)
            except Exception as e:
                logger.error(f"Failed to log histogram to W&B: {e}")

    def log_text(
        self,
        name: str,
        text: str,
        step: Optional[int] = None,
    ):
        """
        Log text.

        Args:
            name: Text label
            text: Text content
            step: Step number
        """
        if not self.enabled:
            return

        # Log to W&B
        if self.wandb_enabled and self.wandb_run is not None:
            try:
                self.wandb.log({name: self.wandb.Html(text)}, step=step)
            except Exception as e:
                logger.error(f"Failed to log text to W&B: {e}")

        # Save locally
        try:
            text_file = self.log_dir / f"{name}.txt"
            with open(text_file, 'a') as f:
                f.write(f"\n--- Step {step} ---\n{text}\n")
        except Exception as e:
            logger.error(f"Failed to save text locally: {e}")

    def watch_model(
        self,
        model,
        log_freq: int = 100,
        log_gradients: bool = True,
        log_parameters: bool = True,
    ):
        """
        Watch model for gradient and parameter tracking.

        Args:
            model: PyTorch model to watch
            log_freq: Logging frequency
            log_gradients: Whether to log gradients
            log_parameters: Whether to log parameters
        """
        if not self.enabled:
            return

        if self.wandb_enabled and self.wandb_run is not None:
            try:
                log_type = ""
                if log_gradients and log_parameters:
                    log_type = "all"
                elif log_gradients:
                    log_type = "gradients"
                elif log_parameters:
                    log_type = "parameters"

                self.wandb.watch(model, log=log_type, log_freq=log_freq)
                logger.success(f"Model watched by W&B (log={log_type})")
            except Exception as e:
                logger.error(f"Failed to watch model: {e}")

    def finish(self):
        """Finish tracking run."""
        if not self.enabled:
            return

        if self.wandb_enabled and self.wandb_run is not None:
            try:
                self.wandb_run.finish()
                logger.success("W&B run finished")
            except Exception as e:
                logger.error(f"Failed to finish W&B run: {e}")

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager (finish run)."""
        self.finish()


# Example usage
if __name__ == "__main__":
    # Example: Track experiment with W&B
    with ExperimentTracker(
        project="secgraph-rl",
        name="ppo_training_v1",
        config={
            "learning_rate": 1e-5,
            "batch_size": 4,
            "ppo_epochs": 4,
        },
        tags=["ppo", "reasoning", "experiment"],
        notes="First PPO training run with value head",
    ) as tracker:
        # Log metrics
        for step in range(100):
            tracker.log_metrics(
                {
                    "loss": 0.5 - step * 0.001,
                    "reward": step * 0.01,
                    "accuracy": min(0.9, step * 0.005),
                },
                step=step,
            )

        # Log checkpoint
        tracker.log_checkpoint(
            checkpoint_path=Path("artifacts/checkpoints/checkpoint_epoch_10.pt"),
            epoch=10,
            metrics={"loss": 0.3, "accuracy": 0.85},
            is_best=True,
        )

        # Log final model
        tracker.log_model(
            model_path=Path("artifacts/models/final_model"),
            model_name="ppo_final",
            metadata={"final_accuracy": 0.92},
        )

    logger.info("Experiment tracking complete")
