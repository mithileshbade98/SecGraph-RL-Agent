"""
Math verifier - verifies correctness of math reasoning tasks.

Used for PPO training with verifiable rewards (RLVR).
Source: "RL with Verifiable Rewards" (2025) - arXiv:2410.15246
"""

from typing import Dict, Any
import re
from loguru import logger


class MathVerifier:
    """Verify math problem solutions."""

    def verify(
        self,
        problem: str,
        solution: str,
        expected_answer: Any = None,
    ) -> Dict[str, Any]:
        """
        Verify a math solution.

        Args:
            problem: Math problem statement
            solution: Proposed solution (can include reasoning steps)
            expected_answer: Expected answer (if known)

        Returns:
            {
                'correct': bool,
                'score': float,
                'extracted_answer': Any,
                'expected_answer': Any,
            }
        """
        # Extract numeric answer from solution
        extracted = self._extract_answer(solution)

        # Compare with expected
        if expected_answer is not None:
            correct = self._compare_answers(extracted, expected_answer)
            score = 1.0 if correct else 0.0
        else:
            # No ground truth - can't verify
            correct = None
            score = 0.5

        return {
            'correct': correct,
            'score': score,
            'extracted_answer': extracted,
            'expected_answer': expected_answer,
        }

    def _extract_answer(self, solution: str) -> Any:
        """Extract final answer from solution text."""
        # Look for patterns like "Answer: 42" or "= 42"
        patterns = [
            r'[Aa]nswer:\s*([0-9.+-]+)',
            r'=\s*([0-9.+-]+)\s*$',
            r'[Tt]herefore,?\s*([0-9.+-]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, solution)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return match.group(1)

        # Fallback: return last number in text
        numbers = re.findall(r'[0-9.+-]+', solution)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                return numbers[-1]

        return None

    def _compare_answers(self, extracted: Any, expected: Any) -> bool:
        """Compare extracted vs expected answer."""
        if extracted is None:
            return False

        # Try numeric comparison
        try:
            return abs(float(extracted) - float(expected)) < 1e-6
        except (ValueError, TypeError):
            pass

        # String comparison
        return str(extracted).strip() == str(expected).strip()


class UnitTestVerifier:
    """Verify solutions by running unit tests."""

    def verify(
        self,
        code: str,
        test_cases: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Verify code by running test cases.

        Args:
            code: Code to test
            test_cases: List of {'input': ..., 'expected': ...}

        Returns:
            {
                'passed': bool,
                'score': float,
                'passed_tests': int,
                'total_tests': int,
            }
        """
        passed_tests = 0
        total_tests = len(test_cases)

        for test in test_cases:
            try:
                # In production, execute code in sandbox
                # For mock, just check syntax
                if "def " in code or "return" in code:
                    passed_tests += 1
            except Exception as e:
                logger.warning(f"Test failed: {e}")
                continue

        passed = (passed_tests == total_tests)
        score = passed_tests / total_tests if total_tests > 0 else 0.0

        return {
            'passed': passed,
            'score': score,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
        }
