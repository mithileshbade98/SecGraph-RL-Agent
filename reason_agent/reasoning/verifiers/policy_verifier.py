"""
Policy verifier - checks graph patterns against security policies.

Evaluates claims like "no suspicious account cluster" on as-of snapshots.
Returns objective pass/fail with evidence.
"""

from typing import Dict, Any, Optional
from loguru import logger


class PolicyVerifier:
    """Verify policy compliance on graph patterns."""

    def __init__(self, graph_loader=None):
        """
        Initialize policy verifier.

        Args:
            graph_loader: BiTemporalGraphLoader instance (optional for mock mode)
        """
        self.graph_loader = graph_loader

    def verify(
        self,
        policy_name: str,
        trace: Dict[str, Any],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Verify a policy against a reasoning trace.

        Args:
            policy_name: Policy to check (e.g., 'no_shared_device_abuse')
            trace: Reasoning trace with steps and results
            threshold: Policy threshold

        Returns:
            {
                'passed': bool,
                'score': float,
                'evidence': Dict,
                'message': str,
            }
        """
        logger.info(f"Verifying policy: {policy_name}")

        if policy_name == "no_shared_device_abuse":
            return self._verify_shared_device_policy(trace, threshold or 3)
        elif policy_name == "velocity_compliance":
            return self._verify_velocity_policy(trace, threshold or 50)
        elif policy_name == "no_burst_anomaly":
            return self._verify_burst_policy(trace, threshold or 3.0)
        else:
            return {
                'passed': False,
                'score': 0.0,
                'evidence': {},
                'message': f'Unknown policy: {policy_name}',
            }

    def _verify_shared_device_policy(
        self,
        trace: Dict[str, Any],
        max_accounts: int,
    ) -> Dict[str, Any]:
        """Policy: No more than N accounts per device."""
        # Extract results from trace
        for step in trace.get('steps', []):
            if step.get('tool') == 'policy_checker':
                result = step.get('result', {}).get('result', {})
                count = result.get('count', 0)
                violation = result.get('violation', False)

                passed = not violation and count < max_accounts

                return {
                    'passed': passed,
                    'score': 1.0 if passed else 0.0,
                    'evidence': {
                        'accounts_per_device': count,
                        'threshold': max_accounts,
                        'device_id': step.get('result', {}).get('evidence', {}).get('device_id'),
                    },
                    'message': f"Shared device policy: {count}/{max_accounts} accounts",
                }

        return {
            'passed': False,
            'score': 0.0,
            'evidence': {},
            'message': 'No policy checker result found in trace',
        }

    def _verify_velocity_policy(
        self,
        trace: Dict[str, Any],
        max_events: int,
    ) -> Dict[str, Any]:
        """Policy: Event velocity within limits."""
        for step in trace.get('steps', []):
            if step.get('tool') == 'velocity_checker':
                result = step.get('result', {}).get('result', {})
                violation = result.get('violation', False)
                actual = result.get('actual_events', 0)

                passed = not violation

                return {
                    'passed': passed,
                    'score': 1.0 if passed else 0.0,
                    'evidence': {
                        'actual_events': actual,
                        'threshold': max_events,
                    },
                    'message': f"Velocity policy: {actual}/{max_events} events",
                }

        return {'passed': True, 'score': 1.0, 'evidence': {}, 'message': 'No velocity check'}

    def _verify_burst_policy(
        self,
        trace: Dict[str, Any],
        threshold_multiplier: float,
    ) -> Dict[str, Any]:
        """Policy: No abnormal burst activity."""
        for step in trace.get('steps', []):
            if step.get('tool') == 'burst_detector':
                result = step.get('result', {}).get('result', {})
                burst_detected = result.get('burst_detected', False)
                burstiness_score = result.get('burstiness_score', 0.0)

                passed = not burst_detected

                return {
                    'passed': passed,
                    'score': 1.0 if passed else 0.0,
                    'evidence': {
                        'burstiness_score': burstiness_score,
                        'threshold': threshold_multiplier,
                    },
                    'message': f"Burst policy: score {burstiness_score:.2f}",
                }

        return {'passed': True, 'score': 1.0, 'evidence': {}, 'message': 'No burst check'}
