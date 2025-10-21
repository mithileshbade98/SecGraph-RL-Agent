"""Tests for data ingestion modules."""

import pytest
from pathlib import Path
import pandas as pd

from reason_agent.ingest.synthetic_generator import SyntheticDataGenerator
from reason_agent.ingest.temporal_preprocess import TemporalFeatureExtractor


def test_synthetic_generator():
    """Test synthetic data generation."""
    generator = SyntheticDataGenerator(seed=42)

    # Generate multi-email abuse
    events = generator.generate_multi_email_free_tier_churn(num_accounts=5)
    assert len(events) > 0
    assert all('device_id' in e for e in events)
    assert all(e['anomaly_type'] == 'multi_email_free_tier_churn' for e in events)

    # Check shared device
    devices = set(e['device_id'] for e in events)
    assert len(devices) == 1, "Should share single device"


def test_temporal_features():
    """Test temporal feature extraction."""
    # Create sample data
    df = pd.DataFrame({
        'user_id': ['user_1'] * 10 + ['user_2'] * 10,
        'event_type': ['login'] * 20,
        'timestamp': pd.date_range('2025-01-01', periods=20, freq='1H'),
        'geo_location': ['US-West'] * 20,
    })

    extractor = TemporalFeatureExtractor(windows_days=[7])

    # Extract features
    features = extractor.extract_all_features(df)

    assert len(features) == 2  # 2 users
    assert 'count_7d' in features.columns
    assert 'burstiness' in features.columns


def test_burstiness_computation():
    """Test burstiness metric."""
    from datetime import datetime, timedelta

    extractor = TemporalFeatureExtractor()

    # Regular events (low burstiness)
    regular = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(10)]
    regular_burst = extractor.compute_burstiness(regular)
    assert -0.5 < regular_burst < 0.5

    # Bursty events (high burstiness)
    bursty = [datetime(2025, 1, 1)] * 5 + [datetime(2025, 1, 2)] * 5
    bursty_burst = extractor.compute_burstiness(bursty)
    assert bursty_burst > 0.5
