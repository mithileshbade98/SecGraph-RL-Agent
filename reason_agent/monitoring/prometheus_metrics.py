"""
Production monitoring with Prometheus metrics.

Provides:
- Request/response metrics
- Model performance metrics
- System resource metrics
- Custom business metrics
- Health check endpoints
"""

import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
from loguru import logger
import psutil
import os


class PrometheusMetrics:
    """
    Prometheus metrics for production monitoring.

    Exports metrics for Prometheus scraping with fallback to local logging.
    """

    def __init__(
        self,
        enabled: bool = True,
        port: int = 8000,
        namespace: str = "secgraph_rl",
    ):
        """
        Initialize Prometheus metrics.

        Args:
            enabled: Whether metrics collection is enabled
            port: Port for metrics endpoint
            namespace: Metrics namespace prefix
        """
        self.enabled = enabled
        self.port = port
        self.namespace = namespace
        self.prometheus_available = False

        if not self.enabled:
            logger.info("Prometheus metrics disabled")
            return

        # Try to import prometheus_client
        try:
            from prometheus_client import (
                Counter,
                Histogram,
                Gauge,
                Summary,
                start_http_server,
            )

            self.Counter = Counter
            self.Histogram = Histogram
            self.Gauge = Gauge
            self.Summary = Summary
            self.prometheus_available = True

            # Initialize metrics
            self._init_metrics()

            # Start metrics server
            try:
                start_http_server(port)
                logger.success(f"Prometheus metrics server started on port {port}")
            except OSError as e:
                logger.warning(f"Metrics server port {port} already in use: {e}")

        except ImportError:
            logger.warning(
                "prometheus_client not installed. Install with: pip install prometheus-client"
            )
            self.prometheus_available = False

    def _init_metrics(self):
        """Initialize all Prometheus metrics."""
        if not self.prometheus_available:
            return

        # Request metrics
        self.request_count = self.Counter(
            f"{self.namespace}_requests_total",
            "Total number of requests",
            ["method", "endpoint", "status"],
        )

        self.request_duration = self.Histogram(
            f"{self.namespace}_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        self.request_in_progress = self.Gauge(
            f"{self.namespace}_requests_in_progress",
            "Number of requests currently in progress",
            ["method", "endpoint"],
        )

        # Model inference metrics
        self.inference_count = self.Counter(
            f"{self.namespace}_inference_total",
            "Total number of model inferences",
            ["model", "status"],
        )

        self.inference_duration = self.Histogram(
            f"{self.namespace}_inference_duration_seconds",
            "Model inference duration in seconds",
            ["model"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
        )

        self.inference_tokens = self.Histogram(
            f"{self.namespace}_inference_tokens",
            "Number of tokens in inference",
            ["model", "type"],  # type: input/output
            buckets=(10, 50, 100, 256, 512, 1024, 2048),
        )

        # Training metrics
        self.training_steps = self.Counter(
            f"{self.namespace}_training_steps_total",
            "Total number of training steps",
            ["trainer"],
        )

        self.training_loss = self.Gauge(
            f"{self.namespace}_training_loss",
            "Current training loss",
            ["trainer", "loss_type"],
        )

        self.training_accuracy = self.Gauge(
            f"{self.namespace}_training_accuracy",
            "Current training accuracy",
            ["trainer"],
        )

        self.training_learning_rate = self.Gauge(
            f"{self.namespace}_training_learning_rate",
            "Current learning rate",
            ["trainer"],
        )

        # Model quality metrics
        self.model_accuracy = self.Gauge(
            f"{self.namespace}_model_accuracy",
            "Model accuracy on validation set",
            ["model", "dataset"],
        )

        self.model_latency_p50 = self.Gauge(
            f"{self.namespace}_model_latency_p50_seconds",
            "Model latency p50 (median)",
            ["model"],
        )

        self.model_latency_p99 = self.Gauge(
            f"{self.namespace}_model_latency_p99_seconds",
            "Model latency p99",
            ["model"],
        )

        # System resource metrics
        self.cpu_usage = self.Gauge(
            f"{self.namespace}_cpu_usage_percent",
            "CPU usage percentage",
        )

        self.memory_usage = self.Gauge(
            f"{self.namespace}_memory_usage_bytes",
            "Memory usage in bytes",
        )

        self.gpu_memory_usage = self.Gauge(
            f"{self.namespace}_gpu_memory_usage_bytes",
            "GPU memory usage in bytes",
            ["gpu_id"],
        )

        self.gpu_utilization = self.Gauge(
            f"{self.namespace}_gpu_utilization_percent",
            "GPU utilization percentage",
            ["gpu_id"],
        )

        # Error metrics
        self.error_count = self.Counter(
            f"{self.namespace}_errors_total",
            "Total number of errors",
            ["error_type", "component"],
        )

        # Cache metrics
        self.cache_hits = self.Counter(
            f"{self.namespace}_cache_hits_total",
            "Total cache hits",
            ["cache_name"],
        )

        self.cache_misses = self.Counter(
            f"{self.namespace}_cache_misses_total",
            "Total cache misses",
            ["cache_name"],
        )

        logger.success("Prometheus metrics initialized")

    # Request tracking
    def track_request(
        self,
        method: str,
        endpoint: str,
        status: str = "success",
        duration: Optional[float] = None,
    ):
        """Track HTTP/API request."""
        if not self.enabled or not self.prometheus_available:
            return

        self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        if duration is not None:
            self.request_duration.labels(method=method, endpoint=endpoint).observe(
                duration
            )

    def request_in_progress_tracker(self, method: str, endpoint: str):
        """Context manager to track requests in progress."""
        if not self.enabled or not self.prometheus_available:
            return _NoOpContextManager()

        return self.request_in_progress.labels(method=method, endpoint=endpoint).track_inprogress()

    # Inference tracking
    def track_inference(
        self,
        model: str,
        status: str = "success",
        duration: Optional[float] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ):
        """Track model inference."""
        if not self.enabled or not self.prometheus_available:
            return

        self.inference_count.labels(model=model, status=status).inc()

        if duration is not None:
            self.inference_duration.labels(model=model).observe(duration)

        if input_tokens is not None:
            self.inference_tokens.labels(model=model, type="input").observe(input_tokens)

        if output_tokens is not None:
            self.inference_tokens.labels(model=model, type="output").observe(
                output_tokens
            )

    # Training tracking
    def track_training_step(
        self,
        trainer: str,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ):
        """Track training step."""
        if not self.enabled or not self.prometheus_available:
            return

        self.training_steps.labels(trainer=trainer).inc()

        if loss is not None:
            self.training_loss.labels(trainer=trainer, loss_type="total").set(loss)

        if accuracy is not None:
            self.training_accuracy.labels(trainer=trainer).set(accuracy)

        if learning_rate is not None:
            self.training_learning_rate.labels(trainer=trainer).set(learning_rate)

    def set_training_loss(self, trainer: str, loss_type: str, value: float):
        """Set training loss gauge."""
        if not self.enabled or not self.prometheus_available:
            return

        self.training_loss.labels(trainer=trainer, loss_type=loss_type).set(value)

    # Model quality tracking
    def set_model_accuracy(self, model: str, dataset: str, accuracy: float):
        """Set model accuracy."""
        if not self.enabled or not self.prometheus_available:
            return

        self.model_accuracy.labels(model=model, dataset=dataset).set(accuracy)

    def set_model_latency(
        self,
        model: str,
        p50: Optional[float] = None,
        p99: Optional[float] = None,
    ):
        """Set model latency percentiles."""
        if not self.enabled or not self.prometheus_available:
            return

        if p50 is not None:
            self.model_latency_p50.labels(model=model).set(p50)

        if p99 is not None:
            self.model_latency_p99.labels(model=model).set(p99)

    # System resource tracking
    def update_system_metrics(self):
        """Update system resource metrics."""
        if not self.enabled or not self.prometheus_available:
            return

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_usage.set(cpu_percent)

        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_usage.set(memory.used)

        # GPU metrics (if available)
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    # GPU memory
                    memory_allocated = torch.cuda.memory_allocated(i)
                    self.gpu_memory_usage.labels(gpu_id=str(i)).set(memory_allocated)

                    # GPU utilization (approximate via memory usage)
                    memory_total = torch.cuda.get_device_properties(i).total_memory
                    utilization = (memory_allocated / memory_total) * 100
                    self.gpu_utilization.labels(gpu_id=str(i)).set(utilization)
        except ImportError:
            pass

    # Error tracking
    def track_error(self, error_type: str, component: str):
        """Track error occurrence."""
        if not self.enabled or not self.prometheus_available:
            return

        self.error_count.labels(error_type=error_type, component=component).inc()

    # Cache tracking
    def track_cache_hit(self, cache_name: str):
        """Track cache hit."""
        if not self.enabled or not self.prometheus_available:
            return

        self.cache_hits.labels(cache_name=cache_name).inc()

    def track_cache_miss(self, cache_name: str):
        """Track cache miss."""
        if not self.enabled or not self.prometheus_available:
            return

        self.cache_misses.labels(cache_name=cache_name).inc()

    # Decorators for automatic tracking
    def track_time(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Decorator to track function execution time."""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled or not self.prometheus_available:
                    return func(*args, **kwargs)

                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    status = "success"
                    return result
                except Exception as e:
                    status = "error"
                    raise
                finally:
                    duration = time.time() - start_time
                    # Track with appropriate metric based on context
                    if hasattr(self, metric_name):
                        metric = getattr(self, metric_name)
                        if labels:
                            metric.labels(**labels).observe(duration)
                        else:
                            metric.observe(duration)

            return wrapper

        return decorator


class _NoOpContextManager:
    """No-op context manager when metrics are disabled."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Global metrics instance
_global_metrics: Optional[PrometheusMetrics] = None


def get_metrics() -> PrometheusMetrics:
    """Get global metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = PrometheusMetrics()
    return _global_metrics


def init_metrics(
    enabled: bool = True,
    port: int = 8000,
    namespace: str = "secgraph_rl",
) -> PrometheusMetrics:
    """Initialize global metrics instance."""
    global _global_metrics
    _global_metrics = PrometheusMetrics(
        enabled=enabled,
        port=port,
        namespace=namespace,
    )
    return _global_metrics


# Example usage
if __name__ == "__main__":
    # Initialize metrics
    metrics = init_metrics(enabled=True, port=8000)

    # Track some metrics
    for i in range(10):
        # Track request
        metrics.track_request(
            method="POST",
            endpoint="/api/query",
            status="success",
            duration=0.5 + i * 0.1,
        )

        # Track inference
        metrics.track_inference(
            model="llama-2-7b",
            status="success",
            duration=1.2,
            input_tokens=128,
            output_tokens=256,
        )

        # Track training step
        metrics.track_training_step(
            trainer="ppo",
            loss=0.5 - i * 0.01,
            accuracy=0.7 + i * 0.02,
            learning_rate=1e-5,
        )

        # Update system metrics
        metrics.update_system_metrics()

        time.sleep(1)

    logger.info("Metrics available at http://localhost:8000/metrics")
