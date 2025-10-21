#!/usr/bin/env python
"""CLI for managing expert audits."""

import click
import json
from pathlib import Path
from loguru import logger

from reason_agent.reasoning.trace_recorder import TraceRecorder


@click.command()
@click.option('--list', 'list_traces', is_flag=True, help='List recent traces')
@click.option('--compare', nargs=2, help='Compare two trace IDs')
@click.option('--output', default='data/audits/pairs.jsonl', help='Output file for pairs')
def main(list_traces, compare, output):
    """Manage expert audits and preference pairs."""
    recorder = TraceRecorder()

    if list_traces:
        traces = recorder.load_recent_traces(n=10)
        logger.info(f"\nRecent traces ({len(traces)}):")
        for trace in traces:
            logger.info(f"  {trace['trace_id']}: {trace['query']}")

    elif compare:
        trace_id_1, trace_id_2 = compare
        trace_1 = recorder.load_trace(trace_id_1)
        trace_2 = recorder.load_trace(trace_id_2)

        if not trace_1 or not trace_2:
            logger.error("Trace not found")
            return

        logger.info(f"\nTrace A: {trace_id_1}")
        logger.info(f"  Steps: {len(trace_1['steps'])}")
        logger.info(f"  Success: {trace_1.get('success', False)}")

        logger.info(f"\nTrace B: {trace_id_2}")
        logger.info(f"  Steps: {len(trace_2['steps'])}")
        logger.info(f"  Success: {trace_2.get('success', False)}")

        # Collect preference (interactive)
        choice = input("\nWhich is better? (A/B/Equal): ").strip().upper()
        rationale = input("Rationale: ").strip()

        if choice in ['A', 'B']:
            pair = {
                'prompt': trace_1['query'],
                'chosen': trace_1 if choice == 'A' else trace_2,
                'rejected': trace_2 if choice == 'A' else trace_1,
                'rationale': rationale,
            }

            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'a') as f:
                f.write(json.dumps(pair) + '\n')

            logger.info(f"✅ Preference pair saved to {output}")


if __name__ == '__main__':
    main()
