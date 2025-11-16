"""
Batch inference optimization for production deployment.

Provides:
- Dynamic batching
- Request queuing
- Latency optimization
- Throughput maximization
- Timeout handling
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Thread, Lock
from loguru import logger
import torch


@dataclass
class InferenceRequest:
    """Single inference request."""

    request_id: str
    input_text: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    timestamp: float = None
    timeout: float = 30.0  # seconds

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class InferenceResponse:
    """Single inference response."""

    request_id: str
    output_text: str
    success: bool = True
    error: Optional[str] = None
    latency: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class BatchOptimizer:
    """
    Optimizes inference throughput via dynamic batching.

    Collects requests into batches and processes them together
    to maximize GPU utilization while maintaining low latency.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size: int = 32,
        max_wait_ms: int = 100,
        max_queue_size: int = 1000,
        num_workers: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize batch optimizer.

        Args:
            model: Model for inference
            tokenizer: Tokenizer
            max_batch_size: Maximum batch size
            max_wait_ms: Maximum wait time before processing batch (milliseconds)
            max_queue_size: Maximum size of request queue
            num_workers: Number of worker threads
            device: Device for inference
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0  # Convert to seconds
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        self.device = device

        # Request queue
        self.request_queue = Queue(maxsize=max_queue_size)
        self.response_dict: Dict[str, InferenceResponse] = {}
        self.response_lock = Lock()

        # Worker threads
        self.workers: List[Thread] = []
        self.running = False

        # Statistics
        self.total_requests = 0
        self.total_batches = 0
        self.total_tokens = 0

        logger.info(
            f"BatchOptimizer initialized: "
            f"max_batch_size={max_batch_size}, "
            f"max_wait_ms={max_wait_ms:.0f}, "
            f"device={device}"
        )

    def start(self):
        """Start worker threads."""
        if self.running:
            logger.warning("BatchOptimizer already running")
            return

        self.running = True

        # Start workers
        for i in range(self.num_workers):
            worker = Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)

        logger.success(f"BatchOptimizer started with {self.num_workers} workers")

    def stop(self):
        """Stop worker threads."""
        if not self.running:
            return

        self.running = False

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)

        logger.info("BatchOptimizer stopped")

    def _worker_loop(self, worker_id: int):
        """Worker loop that processes batches."""
        logger.info(f"Worker {worker_id} started")

        while self.running:
            try:
                # Collect batch
                batch = self._collect_batch()

                if not batch:
                    time.sleep(0.001)  # Small sleep to avoid busy waiting
                    continue

                # Process batch
                self._process_batch(batch, worker_id)

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.info(f"Worker {worker_id} stopped")

    def _collect_batch(self) -> List[InferenceRequest]:
        """Collect requests into a batch."""
        batch = []
        deadline = time.time() + self.max_wait_ms

        while len(batch) < self.max_batch_size and time.time() < deadline:
            try:
                # Try to get request with short timeout
                timeout = max(0.001, deadline - time.time())
                request = self.request_queue.get(timeout=timeout)
                batch.append(request)
            except Empty:
                break

        # If we have at least one request, return immediately
        if batch:
            return batch

        # Otherwise try once more with longer timeout
        try:
            request = self.request_queue.get(timeout=self.max_wait_ms)
            batch.append(request)
        except Empty:
            pass

        return batch

    def _process_batch(self, batch: List[InferenceRequest], worker_id: int):
        """Process a batch of requests."""
        batch_start = time.time()
        batch_size = len(batch)

        logger.debug(f"Worker {worker_id} processing batch of {batch_size}")

        try:
            # Check for timeouts
            current_time = time.time()
            valid_requests = []
            for req in batch:
                if current_time - req.timestamp > req.timeout:
                    # Request timed out
                    self._set_response(
                        InferenceResponse(
                            request_id=req.request_id,
                            output_text="",
                            success=False,
                            error="Request timeout",
                            latency=current_time - req.timestamp,
                        )
                    )
                else:
                    valid_requests.append(req)

            if not valid_requests:
                return

            # Tokenize inputs
            input_texts = [req.input_text for req in valid_requests]
            inputs = self.tokenizer(
                input_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=2048,
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=valid_requests[0].max_tokens,  # Use first request's params
                    temperature=valid_requests[0].temperature,
                    top_p=valid_requests[0].top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode outputs
            generated_texts = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            # Create responses
            batch_end = time.time()
            for req, generated_text, output_ids in zip(
                valid_requests, generated_texts, outputs
            ):
                input_tokens = inputs["input_ids"].shape[1]
                output_tokens = output_ids.shape[0] - input_tokens

                response = InferenceResponse(
                    request_id=req.request_id,
                    output_text=generated_text,
                    success=True,
                    latency=batch_end - req.timestamp,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                self._set_response(response)

            # Update statistics
            self.total_batches += 1
            self.total_requests += len(valid_requests)
            self.total_tokens += sum(r.input_tokens + r.output_tokens for r in self._get_batch_responses(valid_requests))

            batch_latency = batch_end - batch_start
            logger.debug(
                f"Batch processed: size={len(valid_requests)}, "
                f"latency={batch_latency:.3f}s"
            )

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Mark all requests as failed
            for req in batch:
                self._set_response(
                    InferenceResponse(
                        request_id=req.request_id,
                        output_text="",
                        success=False,
                        error=str(e),
                        latency=time.time() - req.timestamp,
                    )
                )

    def _get_batch_responses(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Get responses for batch of requests."""
        with self.response_lock:
            return [
                self.response_dict.get(req.request_id, InferenceResponse(req.request_id, "", success=False))
                for req in requests
            ]

    def _set_response(self, response: InferenceResponse):
        """Store response."""
        with self.response_lock:
            self.response_dict[response.request_id] = response

    def _get_response(self, request_id: str) -> Optional[InferenceResponse]:
        """Get response if available."""
        with self.response_lock:
            return self.response_dict.pop(request_id, None)

    def infer(
        self,
        request: InferenceRequest,
        wait: bool = True,
        poll_interval: float = 0.01,
    ) -> Optional[InferenceResponse]:
        """
        Submit inference request.

        Args:
            request: Inference request
            wait: Whether to wait for response
            poll_interval: Polling interval in seconds

        Returns:
            Response if wait=True, None otherwise
        """
        # Add to queue
        try:
            self.request_queue.put(request, timeout=1.0)
        except Exception as e:
            logger.error(f"Failed to queue request: {e}")
            return InferenceResponse(
                request_id=request.request_id,
                output_text="",
                success=False,
                error="Queue full",
            )

        if not wait:
            return None

        # Wait for response
        deadline = time.time() + request.timeout
        while time.time() < deadline:
            response = self._get_response(request.request_id)
            if response is not None:
                return response
            time.sleep(poll_interval)

        # Timeout
        return InferenceResponse(
            request_id=request.request_id,
            output_text="",
            success=False,
            error="Response timeout",
            latency=request.timeout,
        )

    def infer_batch(
        self,
        requests: List[InferenceRequest],
        wait: bool = True,
    ) -> List[InferenceResponse]:
        """
        Submit batch of inference requests.

        Args:
            requests: List of requests
            wait: Whether to wait for all responses

        Returns:
            List of responses
        """
        # Submit all requests
        for request in requests:
            try:
                self.request_queue.put(request, timeout=1.0)
            except Exception as e:
                logger.error(f"Failed to queue request {request.request_id}: {e}")

        if not wait:
            return []

        # Wait for all responses
        responses = []
        for request in requests:
            response = self.infer(request, wait=True)
            responses.append(response)

        return responses

    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "total_tokens": self.total_tokens,
            "avg_batch_size": (
                self.total_requests / self.total_batches
                if self.total_batches > 0
                else 0
            ),
            "queue_size": self.request_queue.qsize(),
            "pending_responses": len(self.response_dict),
        }


# Example usage
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create optimizer
    optimizer = BatchOptimizer(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=4,
        max_wait_ms=100,
    )

    # Start workers
    optimizer.start()

    try:
        # Submit requests
        requests = [
            InferenceRequest(
                request_id=f"req_{i}",
                input_text=f"What is {i} + {i}?",
                max_tokens=50,
            )
            for i in range(10)
        ]

        # Process batch
        responses = optimizer.infer_batch(requests, wait=True)

        # Print results
        for response in responses:
            logger.info(
                f"Request {response.request_id}: "
                f"success={response.success}, "
                f"latency={response.latency:.3f}s"
            )
            if response.success:
                logger.info(f"Output: {response.output_text[:100]}")

        # Print statistics
        stats = optimizer.get_statistics()
        logger.info(f"Statistics: {stats}")

    finally:
        optimizer.stop()
