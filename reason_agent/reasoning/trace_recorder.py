"""
Trace recorder - persists reasoning traces to JSONL for transparency and RL training.

Every agent run generates a trace with:
- Query
- Retrieval hits (tools, scores)
- Reasoning steps
- Tool calls and results
- Evidence (doc IDs, node IDs)
- Verifiable rewards
"""

from typing import Dict, Any, List, Optional
import json
from pathlib import Path
from datetime import datetime
import uuid
from loguru import logger


class TraceRecorder:
    """Record and persist reasoning traces."""

    def __init__(self, output_dir: str = "artifacts/runs"):
        """
        Initialize trace recorder.

        Args:
            output_dir: Directory to save trace files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_trace_id(self) -> str:
        """Generate unique trace ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"trace_{timestamp}_{unique_id}"

    def record_trace(
        self,
        query: str,
        retrieval_results: List[Dict[str, Any]],
        steps: List[Dict[str, Any]],
        final_result: Dict[str, Any],
        rewards: Optional[Dict[str, float]] = None,
        verification_results: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a complete reasoning trace.

        Args:
            query: User query
            retrieval_results: Retrieved tools/docs with scores
            steps: Reasoning steps (from planner)
            final_result: Final result
            rewards: Verifiable rewards (process, final, penalties, bonuses)
            verification_results: Results from verifiers
            metadata: Additional metadata

        Returns:
            trace_id
        """
        trace_id = self.create_trace_id()

        trace = {
            'trace_id': trace_id,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'retrieval': retrieval_results,
            'steps': steps,
            'final_result': final_result,
            'rewards': rewards or {},
            'verification': verification_results or [],
            'metadata': metadata or {},
        }

        # Save trace
        self.save_trace(trace_id, trace)

        return trace_id

    def save_trace(self, trace_id: str, trace: Dict[str, Any]):
        """Save trace to JSONL file."""
        # Save to daily JSONL file
        date_str = datetime.now().strftime("%Y%m%d")
        trace_file = self.output_dir / f"traces_{date_str}.jsonl"

        with open(trace_file, 'a') as f:
            f.write(json.dumps(trace) + '\n')

        logger.info(f"Saved trace {trace_id} to {trace_file}")

    def load_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Load a trace by ID."""
        # Search all JSONL files
        for trace_file in self.output_dir.glob("traces_*.jsonl"):
            with open(trace_file) as f:
                for line in f:
                    trace = json.loads(line)
                    if trace.get('trace_id') == trace_id:
                        return trace
        return None

    def load_recent_traces(self, n: int = 10) -> List[Dict[str, Any]]:
        """Load N most recent traces."""
        all_traces = []

        # Read all trace files (sorted by date, newest first)
        trace_files = sorted(self.output_dir.glob("traces_*.jsonl"), reverse=True)

        for trace_file in trace_files:
            with open(trace_file) as f:
                for line in f:
                    trace = json.loads(line)
                    all_traces.append(trace)

                    if len(all_traces) >= n:
                        return all_traces

        return all_traces

    def get_trace_stats(self) -> Dict[str, Any]:
        """Get statistics about recorded traces."""
        total_traces = 0
        total_steps = 0
        success_count = 0

        for trace_file in self.output_dir.glob("traces_*.jsonl"):
            with open(trace_file) as f:
                for line in f:
                    trace = json.loads(line)
                    total_traces += 1
                    total_steps += len(trace.get('steps', []))
                    if trace.get('final_result', {}).get('success'):
                        success_count += 1

        return {
            'total_traces': total_traces,
            'avg_steps_per_trace': total_steps / total_traces if total_traces > 0 else 0,
            'success_rate': success_count / total_traces if total_traces > 0 else 0,
        }


def main():
    """Test trace recorder."""
    recorder = TraceRecorder()

    # Example trace
    trace_id = recorder.record_trace(
        query="Detect multi-account abuse with shared devices",
        retrieval_results=[
            {'tool': 'policy_checker', 'score': 0.92},
            {'tool': 'cluster_detector', 'score': 0.88},
        ],
        steps=[
            {
                'step_id': 1,
                'thought': 'Check policy violations',
                'tool': 'policy_checker',
                'parameters': {'pattern': 'multi_account_same_device'},
                'result': {'success': True, 'violation': True},
            },
        ],
        final_result={'success': True, 'detected_abuse': True},
        rewards={'process': 0.8, 'final': 1.0, 'total': 1.8},
        verification_results=[
            {'verifier': 'policy_verifier', 'passed': True},
        ],
    )

    logger.info(f"Created trace: {trace_id}")

    # Load trace
    loaded = recorder.load_trace(trace_id)
    logger.info(f"Loaded trace: {loaded['trace_id']}")

    # Stats
    stats = recorder.get_trace_stats()
    logger.info(f"Trace stats: {stats}")


if __name__ == "__main__":
    main()
