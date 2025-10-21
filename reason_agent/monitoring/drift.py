"""
Drift detection for monitoring agent performance degradation.

Tracks:
- Embedding distribution shifts
- Reward EWMA trends
- Tool success rates
"""

from typing import List, Dict, Any
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import mahalanobis
from loguru import logger


class DriftDetector:
    """Detect drift in agent behavior and performance."""

    def __init__(
        self,
        window_size: int = 100,
        ewma_alpha: float = 0.1,
        alert_threshold: float = 0.15,
    ):
        """
        Initialize drift detector.

        Args:
            window_size: Window size for statistics
            ewma_alpha: EWMA smoothing parameter
            alert_threshold: Threshold for drift alerts
        """
        self.window_size = window_size
        self.ewma_alpha = ewma_alpha
        self.alert_threshold = alert_threshold

    def detect_embedding_drift(
        self,
        current_embeddings: np.ndarray,
        baseline_embeddings: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Detect drift in embedding distributions.

        Uses centroid distance and covariance shift.
        """
        # Compute centroids
        current_centroid = np.mean(current_embeddings, axis=0)
        baseline_centroid = np.mean(baseline_embeddings, axis=0)

        # Euclidean distance between centroids
        centroid_dist = np.linalg.norm(current_centroid - baseline_centroid)

        # Covariance matrices
        current_cov = np.cov(current_embeddings.T)
        baseline_cov = np.cov(baseline_embeddings.T)

        # Frobenius norm of covariance difference
        cov_diff = np.linalg.norm(current_cov - baseline_cov, ord='fro')

        # Detect drift
        drift_detected = centroid_dist > self.alert_threshold

        return {
            'drift_detected': drift_detected,
            'centroid_distance': float(centroid_dist),
            'covariance_diff': float(cov_diff),
            'threshold': self.alert_threshold,
        }

    def compute_reward_ewma(
        self,
        rewards: List[float],
    ) -> Dict[str, Any]:
        """
        Compute EWMA of rewards and detect drops.

        Args:
            rewards: List of reward values (chronological)

        Returns:
            EWMA statistics and alert
        """
        if not rewards:
            return {'ewma': 0.0, 'drop_detected': False}

        # Compute EWMA
        ewma = rewards[0]
        ewma_values = [ewma]

        for reward in rewards[1:]:
            ewma = self.ewma_alpha * reward + (1 - self.ewma_alpha) * ewma
            ewma_values.append(ewma)

        # Detect significant drop
        recent_ewma = ewma_values[-min(10, len(ewma_values)):]
        drop_detected = (max(ewma_values) - min(recent_ewma)) > self.alert_threshold

        return {
            'ewma': float(ewma),
            'ewma_history': ewma_values,
            'drop_detected': drop_detected,
            'drop_magnitude': float(max(ewma_values) - min(recent_ewma)),
        }

    def analyze_tool_success_rate(
        self,
        traces: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze tool success rates over time.

        Args:
            traces: List of reasoning traces

        Returns:
            Tool success statistics
        """
        tool_calls = {}
        tool_successes = {}

        for trace in traces:
            for step in trace.get('steps', []):
                tool = step.get('tool')
                success = step.get('result', {}).get('success', False)

                if tool not in tool_calls:
                    tool_calls[tool] = 0
                    tool_successes[tool] = 0

                tool_calls[tool] += 1
                if success:
                    tool_successes[tool] += 1

        # Compute success rates
        success_rates = {}
        for tool, calls in tool_calls.items():
            success_rates[tool] = tool_successes[tool] / calls if calls > 0 else 0.0

        # Detect tools with low success rate
        low_success_tools = [
            tool for tool, rate in success_rates.items()
            if rate < 0.5  # Below 50% success
        ]

        return {
            'tool_success_rates': success_rates,
            'low_success_tools': low_success_tools,
            'alert': len(low_success_tools) > 0,
        }


def main():
    """CLI for drift monitoring."""
    import argparse

    parser = argparse.ArgumentParser(description="Drift monitoring")
    parser.add_argument("--runs-dir", default="artifacts/runs", help="Runs directory")
    args = parser.parse_args()

    detector = DriftDetector()

    # Load recent traces
    from reason_agent.reasoning.trace_recorder import TraceRecorder
    recorder = TraceRecorder(output_dir=args.runs_dir)
    traces = recorder.load_recent_traces(n=100)

    if not traces:
        logger.warning("No traces found")
        return

    logger.info(f"Analyzing {len(traces)} traces...")

    # Analyze tool success rates
    tool_stats = detector.analyze_tool_success_rate(traces)
    logger.info("\n=== Tool Success Rates ===")
    for tool, rate in tool_stats['tool_success_rates'].items():
        logger.info(f"{tool}: {rate:.2%}")

    if tool_stats['low_success_tools']:
        logger.warning(f"Low success tools: {tool_stats['low_success_tools']}")

    # Analyze reward trends
    rewards = [trace.get('rewards', {}).get('total_reward', 0.0) for trace in traces]
    reward_stats = detector.compute_reward_ewma(rewards)
    logger.info(f"\n=== Reward EWMA ===")
    logger.info(f"Current EWMA: {reward_stats['ewma']:.3f}")
    if reward_stats['drop_detected']:
        logger.warning(f"Reward drop detected! Magnitude: {reward_stats['drop_magnitude']:.3f}")


if __name__ == "__main__":
    main()
