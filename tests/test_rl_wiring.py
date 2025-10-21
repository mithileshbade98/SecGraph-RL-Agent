"""Tests for RL wiring and reward computation."""

import pytest

from reason_agent.rl.rewards import RewardComputer


def test_reward_computation():
    """Test reward computation."""
    config = {
        'process_weight': 0.3,
        'final_weight': 0.7,
        'penalty_long_trace': -0.1,
        'penalty_redundant_tool': -0.05,
        'bonus_early_success': 0.2,
    }

    computer = RewardComputer(config)

    # Successful trace
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

    assert 'process_reward' in rewards
    assert 'final_reward' in rewards
    assert 'total_reward' in rewards

    # Process reward should be 1.0 (all steps successful)
    assert rewards['process_reward'] == 1.0

    # Final reward should be 1.0 (success + verified)
    assert rewards['final_reward'] == 1.0

    # Total should be positive
    assert rewards['total_reward'] > 0


def test_reward_penalties():
    """Test reward penalties for inefficiency."""
    config = {
        'process_weight': 0.3,
        'final_weight': 0.7,
        'penalty_long_trace': -0.1,
        'penalty_redundant_tool': -0.05,
    }

    computer = RewardComputer(config)

    # Long redundant trace
    trace = {
        'steps': [
            {'tool': 'policy_checker', 'result': {'success': True}},
            {'tool': 'policy_checker', 'result': {'success': True}},  # Redundant
            {'tool': 'policy_checker', 'result': {'success': True}},  # Redundant
        ] * 4,  # 12 steps total
        'final_result': {'success': True},
    }

    rewards = computer.compute_reward(trace, [])

    # Should have penalties
    assert rewards['penalties'] < 0


def test_reward_bonuses():
    """Test reward bonuses for efficiency."""
    config = {
        'process_weight': 0.3,
        'final_weight': 0.7,
        'bonus_early_success': 0.2,
    }

    computer = RewardComputer(config)

    # Efficient trace (< 5 steps)
    trace = {
        'steps': [
            {'tool': 'policy_checker', 'result': {'success': True}},
            {'tool': 'cluster_detector', 'result': {'success': True}},
        ],
        'final_result': {'success': True},
    }

    rewards = computer.compute_reward(trace, [])

    # Should have bonus
    assert rewards['bonuses'] > 0
