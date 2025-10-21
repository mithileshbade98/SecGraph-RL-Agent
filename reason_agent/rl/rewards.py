"""
Verifiable reward computation for reasoning traces.

Combines process rewards (step-level) and final rewards (outcome-level)
with penalties and bonuses.

Why: Optimizing how the model reasons, not just outputs.
Source: "RL with Verifiable Rewards" - arXiv:2410.15246
"""

from typing import Dict, Any, List
from loguru import logger


class RewardComputer:
    """Compute verifiable rewards from reasoning traces."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward computer.

        Args:
            config: Reward configuration from base.yaml
        """
        self.process_weight = config.get('process_weight', 0.3)
        self.final_weight = config.get('final_weight', 0.7)
        self.penalty_long_trace = config.get('penalty_long_trace', -0.1)
        self.penalty_redundant_tool = config.get('penalty_redundant_tool', -0.05)
        self.bonus_early_success = config.get('bonus_early_success', 0.2)

    def compute_reward(
        self,
        trace: Dict[str, Any],
        verification_results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Compute total reward for a trace.

        Args:
            trace: Reasoning trace
            verification_results: Results from verifiers

        Returns:
            {
                'process_reward': float,
                'final_reward': float,
                'penalties': float,
                'bonuses': float,
                'total_reward': float,
            }
        """
        # Process rewards (step-level correctness)
        process_reward = self._compute_process_reward(trace)

        # Final rewards (outcome correctness)
        final_reward = self._compute_final_reward(trace, verification_results)

        # Penalties
        penalties = self._compute_penalties(trace)

        # Bonuses
        bonuses = self._compute_bonuses(trace)

        # Total weighted reward
        total_reward = (
            self.process_weight * process_reward +
            self.final_weight * final_reward +
            penalties +
            bonuses
        )

        return {
            'process_reward': process_reward,
            'final_reward': final_reward,
            'penalties': penalties,
            'bonuses': bonuses,
            'total_reward': total_reward,
        }

    def _compute_process_reward(self, trace: Dict[str, Any]) -> float:
        """Compute process reward (step-level correctness)."""
        steps = trace.get('steps', [])
        if not steps:
            return 0.0

        # Reward for successful tool calls
        successful_steps = sum(
            1 for step in steps
            if step.get('result', {}).get('success', False)
        )

        # Normalize by total steps
        return successful_steps / len(steps)

    def _compute_final_reward(
        self,
        trace: Dict[str, Any],
        verification_results: List[Dict[str, Any]],
    ) -> float:
        """Compute final reward (outcome correctness)."""
        # Check if final result succeeded
        final_result = trace.get('final_result', {})
        task_success = final_result.get('success', False)

        # Check verification results
        all_verified = all(
            vr.get('passed', False)
            for vr in verification_results
        )

        # Combined score
        if task_success and all_verified:
            return 1.0
        elif task_success:
            return 0.5
        else:
            return 0.0

    def _compute_penalties(self, trace: Dict[str, Any]) -> float:
        """Compute penalties for inefficiencies."""
        total_penalty = 0.0

        # Penalty for long traces
        num_steps = len(trace.get('steps', []))
        if num_steps > 8:  # Baseline
            total_penalty += (num_steps - 8) * self.penalty_long_trace

        # Penalty for redundant tool calls
        tools_used = [step.get('tool') for step in trace.get('steps', [])]
        unique_tools = set(tools_used)
        if len(tools_used) > len(unique_tools):
            redundancy = len(tools_used) - len(unique_tools)
            total_penalty += redundancy * self.penalty_redundant_tool

        return total_penalty

    def _compute_bonuses(self, trace: Dict[str, Any]) -> float:
        """Compute bonuses for good behavior."""
        total_bonus = 0.0

        # Bonus for early success (< 5 steps)
        num_steps = len(trace.get('steps', []))
        if num_steps < 5 and trace.get('final_result', {}).get('success'):
            total_bonus += self.bonus_early_success

        return total_bonus


def main():
    """Test reward computation."""
    config = {
        'process_weight': 0.3,
        'final_weight': 0.7,
        'penalty_long_trace': -0.1,
        'penalty_redundant_tool': -0.05,
        'bonus_early_success': 0.2,
    }

    computer = RewardComputer(config)

    # Example trace
    trace = {
        'steps': [
            {'tool': 'policy_checker', 'result': {'success': True}},
            {'tool': 'cluster_detector', 'result': {'success': True}},
        ],
        'final_result': {'success': True},
    }

    verification_results = [
        {'passed': True, 'score': 1.0},
    ]

    rewards = computer.compute_reward(trace, verification_results)
    logger.info(f"Rewards: {rewards}")


if __name__ == "__main__":
    main()
