"""Tests for verifiers."""

import pytest

from reason_agent.reasoning.verifiers.policy_verifier import PolicyVerifier
from reason_agent.reasoning.verifiers.math_verifier import MathVerifier, UnitTestVerifier


def test_policy_verifier():
    """Test policy verification."""
    verifier = PolicyVerifier()

    # Mock trace with policy violation
    trace = {
        'steps': [
            {
                'tool': 'policy_checker',
                'result': {
                    'result': {
                        'violation': True,
                        'count': 5,
                    },
                    'evidence': {
                        'device_id': 'device_123',
                    }
                }
            }
        ]
    }

    result = verifier.verify('no_shared_device_abuse', trace, threshold=3)

    assert result['passed'] is False
    assert result['score'] == 0.0


def test_math_verifier():
    """Test math verification."""
    verifier = MathVerifier()

    # Correct solution
    result = verifier.verify(
        problem="What is 2 + 2?",
        solution="Let's compute: 2 + 2 = 4. Answer: 4",
        expected_answer=4
    )

    assert result['correct'] is True
    assert result['score'] == 1.0
    assert result['extracted_answer'] == 4.0

    # Wrong solution
    result = verifier.verify(
        problem="What is 2 + 2?",
        solution="Answer: 5",
        expected_answer=4
    )

    assert result['correct'] is False
    assert result['score'] == 0.0


def test_unittest_verifier():
    """Test unit test verification."""
    verifier = UnitTestVerifier()

    code = "def add(a, b):\n    return a + b"
    test_cases = [
        {'input': (2, 2), 'expected': 4},
        {'input': (0, 0), 'expected': 0},
    ]

    result = verifier.verify(code, test_cases)

    assert 'passed' in result
    assert 'score' in result
    assert result['total_tests'] == 2
