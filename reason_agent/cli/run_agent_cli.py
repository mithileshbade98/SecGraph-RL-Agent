#!/usr/bin/env python
"""CLI for running agent queries end-to-end."""

import click
import json
from loguru import logger

from reason_agent.tools.registry import ToolRegistry
from reason_agent.tools.router import ToolRouter
from reason_agent.tools.executors import ExecutorRegistry
from reason_agent.reasoning.planner import ReasoningPlanner
from reason_agent.reasoning.trace_recorder import TraceRecorder
from reason_agent.rl.rewards import RewardComputer


@click.command()
@click.option('--query', required=True, help='Query to execute')
@click.option('--config', default='configs/base.yaml', help='Config file')
@click.option('--save-trace', is_flag=True, default=True, help='Save trace to disk')
def main(query, config, save_trace):
    """Run agent query end-to-end."""
    logger.info(f"Query: {query}")

    # Initialize components
    registry = ToolRegistry()
    router = ToolRouter(registry, method="rule")
    executors = ExecutorRegistry()
    planner = ReasoningPlanner(registry, router, executors, use_llm=False)
    trace_recorder = TraceRecorder()
    reward_computer = RewardComputer({
        'process_weight': 0.3,
        'final_weight': 0.7,
    })

    # Execute
    trace = planner.plan_and_execute(query)

    # Compute rewards
    rewards = reward_computer.compute_reward(trace, [])

    # Print results
    logger.info(f"\n{'='*60}")
    logger.info(f"Success: {trace['success']}")
    logger.info(f"Steps: {trace['num_steps']}")
    logger.info(f"Total Reward: {rewards['total_reward']:.3f}")
    logger.info(f"{'='*60}\n")

    # Print steps
    for step in trace['steps']:
        logger.info(f"Step {step['step_id']}: {step['thought']}")
        logger.info(f"  Tool: {step['tool']}")
        logger.info(f"  Success: {step.get('result', {}).get('success', False)}")

    # Save trace
    if save_trace:
        trace_id = trace_recorder.record_trace(
            query=query,
            retrieval_results=[],
            steps=trace['steps'],
            final_result=trace['final_result'],
            rewards=rewards,
        )
        logger.info(f"\n✅ Trace saved: {trace_id}")


if __name__ == '__main__':
    main()
