#!/usr/bin/env python
"""CLI for data ingestion: generate synthetic data and load into Neo4j."""

import click
from pathlib import Path
from loguru import logger

from reason_agent.ingest.synthetic_generator import SyntheticDataGenerator
from reason_agent.ingest.graph_loader import BiTemporalGraphLoader
from reason_agent.ingest.temporal_preprocess import TemporalFeatureExtractor


@click.command()
@click.option('--config', default='configs/base.yaml', help='Config file')
@click.option('--output-dir', default='data/synthetic', help='Output directory')
@click.option('--seed', default=42, help='Random seed')
@click.option('--load-graph', is_flag=True, help='Load into Neo4j')
@click.option('--extract-features', is_flag=True, help='Extract temporal features')
def main(config, output_dir, seed, load_graph, extract_features):
    """Ingest synthetic security event data."""
    logger.info("Starting data ingestion...")

    # Generate synthetic data
    logger.info("Generating synthetic data...")
    generator = SyntheticDataGenerator(seed=seed)
    parquet_file = generator.generate_all_anomalies(output_dir=output_dir)

    logger.info(f"Generated data: {parquet_file}")

    # Load into Neo4j
    if load_graph:
        logger.info("Loading data into Neo4j...")
        import os
        from dotenv import load_dotenv
        load_dotenv()

        try:
            with BiTemporalGraphLoader(
                uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                user=os.getenv('NEO4J_USER', 'neo4j'),
                password=os.getenv('NEO4J_PASSWORD', 'secgraph123'),
            ) as loader:
                loader.load_from_parquet(parquet_file)
                stats = loader.get_graph_stats()
                logger.info(f"Graph stats: {stats}")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            logger.info("Skipping graph load (Neo4j not available)")

    # Extract temporal features
    if extract_features:
        logger.info("Extracting temporal features...")
        import pandas as pd
        df = pd.read_parquet(parquet_file)

        extractor = TemporalFeatureExtractor()
        features = extractor.extract_all_features(df)

        feature_file = Path(output_dir) / "temporal_features.parquet"
        features.to_parquet(feature_file, index=False)
        logger.info(f"Features saved: {feature_file}")

    logger.info("✅ Ingestion complete!")


if __name__ == '__main__':
    main()
