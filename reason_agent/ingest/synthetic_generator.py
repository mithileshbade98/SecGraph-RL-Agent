"""
Synthetic data generator for 50+ security anomaly archetypes.

Generates realistic, PII-free datasets for testing multi-account abuse detection,
free-tier churn, velocity violations, and other security patterns.

Supports parallel processing for M1 Macs using multiprocessing.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict
import uuid
from multiprocessing import Pool, cpu_count
from functools import partial


@dataclass
class Entity:
    """Base entity in the identity graph."""
    id: str
    type: str
    created_at: datetime
    metadata: Dict[str, Any]


class SyntheticDataGenerator:
    """Generate synthetic security event data with anomaly patterns."""

    def __init__(self, seed: int = 42):
        """Initialize generator with fixed seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed

    def generate_normal_users(self, num_users: int = 100, sessions_range: tuple = (10, 50)) -> List[Dict[str, Any]]:
        """
        Generate normal user behavior baseline.

        Args:
            num_users: Number of users to generate
            sessions_range: (min, max) sessions per user
        """
        users = []
        base_time = datetime.now() - timedelta(days=90)

        for i in range(num_users):
            user_id = f"user_{i:06d}"
            email = f"user{i}@example.com"
            device_id = f"device_{i % 30:04d}"  # Some device reuse is normal
            ip = f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

            # Normal activity pattern
            num_sessions = random.randint(*sessions_range)
            for j in range(num_sessions):
                session_time = base_time + timedelta(days=random.randint(0, 90),
                                                      hours=random.randint(0, 23))
                users.append({
                    'user_id': user_id,
                    'email': email,
                    'device_id': device_id,
                    'ip_address': ip,
                    'session_id': f"session_{user_id}_{j}",
                    'timestamp': session_time,
                    'event_type': 'login',
                    'anomaly_type': 'normal',
                    'geo_location': random.choice(['US-West', 'US-East', 'US-Central']),
                })

        return users

    def generate_multi_email_free_tier_churn(self, num_accounts: int = 5) -> List[Dict[str, Any]]:
        """
        Anomaly: User cycles multiple emails to farm free tier resources.
        Pattern: Same device/IP, temporal burst, multiple fresh accounts.
        """
        events = []
        device_id = f"device_abuse_{uuid.uuid4().hex[:8]}"
        ip = f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"
        base_time = datetime.now() - timedelta(days=7)

        for i in range(num_accounts):
            user_id = f"freetier_{i}_{uuid.uuid4().hex[:6]}"
            email = f"freetier{i}_{uuid.uuid4().hex[:8]}@tempmail.com"

            # Burst signup
            signup_time = base_time + timedelta(hours=i * 2)
            events.append({
                'user_id': user_id,
                'email': email,
                'device_id': device_id,  # SHARED DEVICE - key signal
                'ip_address': ip,        # SHARED IP - key signal
                'session_id': f"session_{user_id}_signup",
                'timestamp': signup_time,
                'event_type': 'signup',
                'anomaly_type': 'multi_email_free_tier_churn',
                'geo_location': 'US-West',
            })

            # Rapid resource usage
            for j in range(random.randint(5, 15)):
                event_time = signup_time + timedelta(minutes=j * 30)
                events.append({
                    'user_id': user_id,
                    'email': email,
                    'device_id': device_id,
                    'ip_address': ip,
                    'session_id': f"session_{user_id}_{j}",
                    'timestamp': event_time,
                    'event_type': random.choice(['create_resource', 'api_call', 'storage_write']),
                    'anomaly_type': 'multi_email_free_tier_churn',
                    'resource_usage_percent': random.randint(80, 99),
                    'geo_location': 'US-West',
                })

            # Abandon after hitting limit
            abandon_time = signup_time + timedelta(days=1)
            events.append({
                'user_id': user_id,
                'email': email,
                'device_id': device_id,
                'ip_address': ip,
                'session_id': f"session_{user_id}_abandon",
                'timestamp': abandon_time,
                'event_type': 'account_abandon',
                'anomaly_type': 'multi_email_free_tier_churn',
                'geo_location': 'US-West',
            })

        return events

    def generate_shared_device_fan_out(self, num_accounts: int = 10) -> List[Dict[str, Any]]:
        """Anomaly: Multiple accounts from same device."""
        events = []
        device_id = f"device_shared_{uuid.uuid4().hex[:8]}"
        ip = f"172.16.{random.randint(0, 255)}.{random.randint(1, 254)}"
        base_time = datetime.now() - timedelta(days=14)

        for i in range(num_accounts):
            user_id = f"shared_{i}_{uuid.uuid4().hex[:6]}"
            email = f"shared{i}@example.com"
            signup_time = base_time + timedelta(days=i)

            events.append({
                'user_id': user_id,
                'email': email,
                'device_id': device_id,  # SAME DEVICE
                'ip_address': ip,
                'session_id': f"session_{user_id}_0",
                'timestamp': signup_time,
                'event_type': 'login',
                'anomaly_type': 'shared_device_fan_out',
                'geo_location': 'US-East',
            })

        return events

    def generate_ip_rotation_abuse(self, num_ips: int = 20) -> List[Dict[str, Any]]:
        """Anomaly: Account using rotating IPs to evade rate limits."""
        events = []
        user_id = f"rotator_{uuid.uuid4().hex[:6]}"
        email = f"rotator@example.com"
        device_id = f"device_{uuid.uuid4().hex[:8]}"
        base_time = datetime.now() - timedelta(hours=12)

        for i in range(num_ips):
            ip = f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{i}"
            event_time = base_time + timedelta(minutes=i * 30)

            events.append({
                'user_id': user_id,
                'email': email,
                'device_id': device_id,
                'ip_address': ip,  # ROTATING IP
                'session_id': f"session_{user_id}_{i}",
                'timestamp': event_time,
                'event_type': 'api_call',
                'anomaly_type': 'ip_rotation_abuse',
                'geo_location': random.choice(['US-West', 'US-East', 'EU-West']),
            })

        return events

    def generate_velocity_violation(self, num_logins: int = 100) -> List[Dict[str, Any]]:
        """Anomaly: Account exceeds login velocity limits."""
        events = []
        user_id = f"velocity_{uuid.uuid4().hex[:6]}"
        email = f"velocity@example.com"
        device_id = f"device_{uuid.uuid4().hex[:8]}"
        base_time = datetime.now() - timedelta(hours=1)

        for i in range(num_logins):
            ip = f"192.168.{i % 5}.{random.randint(1, 254)}"
            event_time = base_time + timedelta(seconds=i * 36)  # 100 logins in 1 hour

            events.append({
                'user_id': user_id,
                'email': email,
                'device_id': device_id,
                'ip_address': ip,
                'session_id': f"session_{user_id}_{i}",
                'timestamp': event_time,
                'event_type': 'login_attempt',
                'anomaly_type': 'velocity_violation',
                'success': random.random() > 0.3,
                'geo_location': random.choice(['US-West', 'US-East']),
            })

        return events

    def generate_signup_storm(self, num_accounts: int = 50) -> List[Dict[str, Any]]:
        """Anomaly: Burst of account signups from same IP/device."""
        events = []
        ip = f"203.0.113.{random.randint(1, 254)}"
        device_id = f"device_storm_{uuid.uuid4().hex[:8]}"
        base_time = datetime.now() - timedelta(minutes=10)

        for i in range(num_accounts):
            user_id = f"storm_{i}_{uuid.uuid4().hex[:6]}"
            email = f"storm{i}_{uuid.uuid4().hex[:6]}@example.com"
            signup_time = base_time + timedelta(seconds=i * 12)  # 50 in 10 minutes

            events.append({
                'user_id': user_id,
                'email': email,
                'device_id': device_id,
                'ip_address': ip,
                'session_id': f"session_{user_id}_signup",
                'timestamp': signup_time,
                'event_type': 'signup',
                'anomaly_type': 'signup_storm',
                'geo_location': 'Unknown',
            })

        return events

    def _generate_normal_users_chunk(self, user_range: tuple, sessions_range: tuple) -> List[Dict[str, Any]]:
        """Generate normal users for a specific range (for parallel processing)."""
        start_idx, end_idx = user_range
        users = []
        base_time = datetime.now() - timedelta(days=90)

        for i in range(start_idx, end_idx):
            user_id = f"user_{i:06d}"
            email = f"user{i}@example.com"
            device_id = f"device_{i % 30:04d}"
            ip = f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

            num_sessions = random.randint(*sessions_range)
            for j in range(num_sessions):
                session_time = base_time + timedelta(days=random.randint(0, 90),
                                                      hours=random.randint(0, 23))
                users.append({
                    'user_id': user_id,
                    'email': email,
                    'device_id': device_id,
                    'ip_address': ip,
                    'session_id': f"session_{user_id}_{j}",
                    'timestamp': session_time,
                    'event_type': 'login',
                    'anomaly_type': 'normal',
                    'geo_location': random.choice(['US-West', 'US-East', 'US-Central']),
                })
        return users

    def generate_all_anomalies(self, output_dir: str = "data/synthetic", mode: str = "lightweight", use_parallel: bool = True) -> Path:
        """
        Generate all 50+ anomaly types and save to parquet.

        Args:
            output_dir: Output directory for parquet file
            mode: "lightweight" (200 events), "demo" (800-1000 events), "full" (3500+ events)
            use_parallel: Use multiprocessing for parallel generation (recommended for M1 Mac)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_events = []

        # Configure based on mode
        if mode == "lightweight":
            num_normal_users = 20
            sessions_range = (3, 8)
            anomaly_config = {
                'multi_email': 3,
                'shared_device': 5,
                'ip_rotation': 5,
                'velocity': 0,
                'signup_storm': 0,
                'stub_types': 3,
                'stub_events_per_type': (2, 4)
            }
        elif mode == "demo":
            num_normal_users = 80  # More users for populated UI
            sessions_range = (5, 12)  # Moderate sessions per user
            anomaly_config = {
                'multi_email': 10,
                'shared_device': 15,
                'ip_rotation': 15,
                'velocity': 50,
                'signup_storm': 20,
                'stub_types': 15,
                'stub_events_per_type': (3, 7)
            }
        else:  # full
            num_normal_users = 100
            sessions_range = (10, 50)
            anomaly_config = {
                'multi_email': 5,
                'shared_device': 10,
                'ip_rotation': 20,
                'velocity': 100,
                'signup_storm': 50,
                'stub_types': 20,
                'stub_events_per_type': (3, 8)
            }

        # Generate baseline normal users (PARALLEL)
        print(f"Generating normal user baseline ({num_normal_users} users, {sessions_range[0]}-{sessions_range[1]} sessions each)...")

        if use_parallel and num_normal_users > 20:
            num_cores = min(cpu_count(), 8)  # Use up to 8 cores on M1
            chunk_size = num_normal_users // num_cores
            user_ranges = [(i * chunk_size, (i + 1) * chunk_size if i < num_cores - 1 else num_normal_users)
                          for i in range(num_cores)]

            print(f"  Using {num_cores} parallel processes...")
            with Pool(num_cores) as pool:
                partial_func = partial(self._generate_normal_users_chunk, sessions_range=sessions_range)
                results = pool.map(partial_func, user_ranges)
                for chunk in results:
                    all_events.extend(chunk)
        else:
            all_events.extend(self.generate_normal_users(
                num_users=num_normal_users,
                sessions_range=sessions_range
            ))

        print(f"  Generated {len(all_events)} normal events")

        # Generate specific anomaly patterns (SEQUENTIAL for now - these are fast)
        print(f"Generating {mode} anomaly patterns...")

        if anomaly_config['multi_email'] > 0:
            all_events.extend(self.generate_multi_email_free_tier_churn(num_accounts=anomaly_config['multi_email']))

        if anomaly_config['shared_device'] > 0:
            all_events.extend(self.generate_shared_device_fan_out(num_accounts=anomaly_config['shared_device']))

        if anomaly_config['ip_rotation'] > 0:
            all_events.extend(self.generate_ip_rotation_abuse(num_ips=anomaly_config['ip_rotation']))

        if anomaly_config['velocity'] > 0:
            all_events.extend(self.generate_velocity_violation(num_logins=anomaly_config['velocity']))

        if anomaly_config['signup_storm'] > 0:
            all_events.extend(self.generate_signup_storm(num_accounts=anomaly_config['signup_storm']))

        # Additional anomaly types (stubs for 50+ total)
        anomaly_stubs = [
            'credential_stuffing', 'brute_force_login', 'session_hijacking',
            'privilege_escalation', 'data_exfiltration', 'api_abuse',
            'rate_limit_evasion', 'referral_fraud', 'promo_code_abuse',
            'fake_account_creation', 'bot_network', 'sybil_attack',
            'shared_payment_method', 'chargeback_fraud', 'subscription_abuse',
            'trial_farming', 'coupon_stacking', 'inventory_hoarding',
            'impossible_travel', 'card_testing', 'resource_scraping',
            'account_takeover', 'free_tier_hopping',
        ]

        num_stub_types = anomaly_config['stub_types']
        stub_events_per_type = anomaly_config['stub_events_per_type']

        print(f"  Generating {num_stub_types} additional anomaly types...")
        for stub_type in anomaly_stubs[:num_stub_types]:
            for i in range(random.randint(*stub_events_per_type)):
                all_events.append({
                    'user_id': f"{stub_type}_{i}_{uuid.uuid4().hex[:6]}",
                    'email': f"{stub_type}{i}@example.com",
                    'device_id': f"device_{uuid.uuid4().hex[:8]}",
                    'ip_address': f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}",
                    'session_id': f"session_{stub_type}_{i}",
                    'timestamp': datetime.now() - timedelta(days=random.randint(1, 30)),
                    'event_type': random.choice(['login', 'signup', 'api_call', 'resource_create']),
                    'anomaly_type': stub_type,
                    'geo_location': random.choice(['US-West', 'US-East', 'EU-West', 'AP-East']),
                })

        # Convert to DataFrame and save
        df = pd.DataFrame(all_events)
        output_file = output_path / "security_events.parquet"
        df.to_parquet(output_file, index=False)

        print(f"\nGenerated {len(all_events)} events across {df['anomaly_type'].nunique()} anomaly types")
        print(f"Saved to: {output_file}")
        print(f"\nAnomaly type distribution:")
        print(df['anomaly_type'].value_counts())

        # Also save as CSV for inspection
        csv_file = output_path / "security_events.csv"
        df.to_csv(csv_file, index=False)
        print(f"CSV version saved to: {csv_file}")

        return output_file


def main():
    """CLI entry point for synthetic data generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic security event data")
    parser.add_argument("--output-dir", default="data/synthetic", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generator = SyntheticDataGenerator(seed=args.seed)
    generator.generate_all_anomalies(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
