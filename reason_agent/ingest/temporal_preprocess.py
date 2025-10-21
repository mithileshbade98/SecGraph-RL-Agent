"""
Temporal preprocessing and feature extraction.

Extracts rolling windows, exponential decay counts, burstiness metrics,
inter-arrival times, and session velocity features for graph ML.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger


class TemporalFeatureExtractor:
    """Extract temporal features from security event data."""

    def __init__(self, windows_days: List[int] = [7, 14, 30], decay_lambda: float = 0.1):
        """
        Initialize temporal feature extractor.

        Args:
            windows_days: Rolling window sizes in days
            decay_lambda: Exponential decay parameter
        """
        self.windows_days = windows_days
        self.decay_lambda = decay_lambda

    def compute_rolling_windows(
        self,
        df: pd.DataFrame,
        entity_col: str,
        event_col: str,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Compute rolling window aggregations.

        Uses time-based filtering instead of pandas rolling windows
        to support both numeric and categorical (string) data.
        """
        df = df.sort_values(timestamp_col)
        features = []

        for window_days in self.windows_days:
            window_timedelta = timedelta(days=window_days)

            # Group by entity and compute rolling stats
            for entity_id in df[entity_col].unique():
                entity_df = df[df[entity_col] == entity_id].copy()

                if len(entity_df) == 0:
                    features.append({
                        'entity_id': entity_id,
                        f'count_{window_days}d': 0,
                        f'unique_events_{window_days}d': 0,
                    })
                    continue

                # Get the last timestamp for this entity
                max_timestamp = entity_df[timestamp_col].max()
                window_start = max_timestamp - window_timedelta

                # Filter to events within the window (last N days)
                window_df = entity_df[entity_df[timestamp_col] >= window_start]

                # Compute aggregations (works with both numeric and string columns)
                rolling_count = len(window_df)
                rolling_unique = window_df[event_col].nunique()

                features.append({
                    'entity_id': entity_id,
                    f'count_{window_days}d': rolling_count,
                    f'unique_events_{window_days}d': rolling_unique,
                })

        return pd.DataFrame(features)

    def compute_exponential_decay_count(
        self,
        timestamps: List[datetime],
        reference_time: Optional[datetime] = None
    ) -> float:
        """
        Compute exponential decay weighted count.

        More recent events have higher weight: weight = exp(-λ * Δt)
        """
        if reference_time is None:
            reference_time = max(timestamps)

        total_weight = 0.0
        for ts in timestamps:
            delta_days = (reference_time - ts).total_seconds() / 86400.0
            weight = np.exp(-self.decay_lambda * delta_days)
            total_weight += weight

        return total_weight

    def compute_burstiness(
        self,
        timestamps: List[datetime],
        expected_rate: Optional[float] = None
    ) -> float:
        """
        Compute burstiness coefficient.

        Burstiness = (σ - μ) / (σ + μ) where σ is std dev, μ is mean of inter-arrival times.
        Range: [-1, 1], where 1 = very bursty, -1 = very regular, 0 = Poisson.
        """
        if len(timestamps) < 2:
            return 0.0

        # Sort timestamps
        sorted_ts = sorted(timestamps)

        # Compute inter-arrival times (in seconds)
        inter_arrivals = []
        for i in range(1, len(sorted_ts)):
            delta = (sorted_ts[i] - sorted_ts[i-1]).total_seconds()
            inter_arrivals.append(delta)

        if not inter_arrivals:
            return 0.0

        mean_iat = np.mean(inter_arrivals)
        std_iat = np.std(inter_arrivals)

        if mean_iat + std_iat == 0:
            return 0.0

        burstiness = (std_iat - mean_iat) / (std_iat + mean_iat)
        return burstiness

    def compute_velocity_features(
        self,
        df: pd.DataFrame,
        entity_col: str,
        geo_col: str = 'geo_location',
        timestamp_col: str = 'timestamp',
        time_window_hours: int = 24
    ) -> pd.DataFrame:
        """
        Compute velocity features (events per time, geographic diversity).

        Detects impossible travel and rapid location changes.
        """
        df = df.sort_values(timestamp_col)
        features = []

        for entity_id in df[entity_col].unique():
            entity_df = df[df[entity_col] == entity_id]

            # Compute events per hour in last N hours
            recent_cutoff = entity_df[timestamp_col].max() - timedelta(hours=time_window_hours)
            recent_events = entity_df[entity_df[timestamp_col] >= recent_cutoff]
            events_per_hour = len(recent_events) / time_window_hours

            # Geographic diversity
            unique_locations = entity_df[geo_col].nunique()
            location_entropy = 0.0
            if unique_locations > 1:
                location_counts = entity_df[geo_col].value_counts(normalize=True)
                location_entropy = -np.sum(location_counts * np.log2(location_counts))

            # Detect rapid location changes
            location_changes = 0
            prev_location = None
            for _, row in entity_df.iterrows():
                if prev_location is not None and row[geo_col] != prev_location:
                    location_changes += 1
                prev_location = row[geo_col]

            features.append({
                'entity_id': entity_id,
                f'events_per_hour_{time_window_hours}h': events_per_hour,
                'unique_locations': unique_locations,
                'location_entropy': location_entropy,
                'location_changes': location_changes,
            })

        return pd.DataFrame(features)

    def create_sequence_pack(
        self,
        df: pd.DataFrame,
        entity_col: str,
        event_col: str,
        timestamp_col: str = 'timestamp',
        max_length: int = 100
    ) -> Dict[str, List[Tuple[str, float, Dict]]]:
        """
        Create ordered event sequences per entity for seq2seq/GNN models.

        Returns: {entity_id: [(event, Δt, meta), ...]}
        """
        df = df.sort_values(timestamp_col)
        sequences = {}

        for entity_id in df[entity_col].unique():
            entity_df = df[df[entity_col] == entity_id]

            sequence = []
            prev_time = None

            for idx, row in entity_df.iterrows():
                event = row[event_col]
                timestamp = row[timestamp_col]

                # Compute delta time from previous event
                if prev_time is None:
                    delta_t = 0.0
                else:
                    delta_t = (timestamp - prev_time).total_seconds()

                # Collect metadata
                meta = {k: v for k, v in row.items()
                        if k not in [entity_col, event_col, timestamp_col]}

                sequence.append((event, delta_t, meta))
                prev_time = timestamp

                # Limit sequence length
                if len(sequence) >= max_length:
                    break

            sequences[entity_id] = sequence

        return sequences

    def extract_all_features(
        self,
        df: pd.DataFrame,
        entity_col: str = 'user_id',
        event_col: str = 'event_type',
        timestamp_col: str = 'timestamp',
        geo_col: str = 'geo_location'
    ) -> pd.DataFrame:
        """Extract all temporal features."""
        logger.info("Extracting rolling window features...")
        rolling_features = self.compute_rolling_windows(df, entity_col, event_col, timestamp_col)

        logger.info("Computing velocity features...")
        velocity_features = self.compute_velocity_features(df, entity_col, geo_col, timestamp_col)

        logger.info("Computing burstiness and decay features...")
        burst_features = []
        for entity_id in df[entity_col].unique():
            entity_df = df[df[entity_col] == entity_id]
            timestamps = list(entity_df[timestamp_col])

            burstiness = self.compute_burstiness(timestamps)
            decay_count = self.compute_exponential_decay_count(timestamps)

            burst_features.append({
                'entity_id': entity_id,
                'burstiness': burstiness,
                'decay_weighted_count': decay_count,
                'total_events': len(timestamps),
            })

        burst_df = pd.DataFrame(burst_features)

        # Merge all features
        logger.info("Merging features...")
        all_features = rolling_features.merge(
            velocity_features, on='entity_id', how='outer'
        ).merge(
            burst_df, on='entity_id', how='outer'
        ).fillna(0)

        return all_features


def main():
    """CLI entry point for temporal feature extraction."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract temporal features from security events")
    parser.add_argument("--input", default="data/synthetic/security_events.parquet",
                        help="Input parquet file")
    parser.add_argument("--output", default="data/synthetic/temporal_features.parquet",
                        help="Output parquet file")
    parser.add_argument("--entity-col", default="user_id", help="Entity column name")
    parser.add_argument("--event-col", default="event_type", help="Event column name")
    args = parser.parse_args()

    # Load data
    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df)} events from {args.input}")

    # Extract features
    extractor = TemporalFeatureExtractor()
    features = extractor.extract_all_features(
        df,
        entity_col=args.entity_col,
        event_col=args.event_col
    )

    # Save features
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(features)} feature vectors to {output_path}")

    # Print sample
    logger.info("\nSample features:")
    logger.info(features.head())


if __name__ == "__main__":
    main()
