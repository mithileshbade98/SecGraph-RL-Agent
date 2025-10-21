#!/usr/bin/env python
"""
Initialize all data for SecGraph-RL Agent:
1. Generate synthetic security events
2. Load into Neo4j
3. Extract temporal features
4. Build FAISS index
5. Create sample audit pairs
"""

import sys
import os
import json
import time
from pathlib import Path
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def wait_for_neo4j(max_retries=30):
    """Wait for Neo4j to be ready."""
    from neo4j import GraphDatabase

    uri = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'secgraph123')

    logger.info(f"Waiting for Neo4j at {uri}...")

    for i in range(max_retries):
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
            driver.close()
            logger.info("✓ Neo4j is ready!")
            return True
        except Exception as e:
            logger.warning(f"  Attempt {i+1}/{max_retries}: Neo4j not ready ({e})")
            time.sleep(2)

    logger.error("✗ Neo4j failed to start")
    return False


def main():
    """Run complete data initialization."""

    logger.info("="*60)
    logger.info("SecGraph-RL Agent - Data Initialization")
    logger.info("="*60)

    # Step 0: Wait for Neo4j
    if not wait_for_neo4j():
        logger.error("Neo4j not available, exiting")
        sys.exit(1)

    # Step 1: Generate synthetic data
    logger.info("\n[1/5] Generating synthetic security events...")
    from reason_agent.ingest.synthetic_generator import SyntheticDataGenerator

    generator = SyntheticDataGenerator(seed=42)
    parquet_file = generator.generate_all_anomalies(output_dir="data/synthetic")
    logger.info(f"✓ Generated: {parquet_file}")

    # Step 2: Load into Neo4j
    logger.info("\n[2/5] Loading data into Neo4j...")
    from reason_agent.ingest.graph_loader import BiTemporalGraphLoader

    try:
        with BiTemporalGraphLoader(
            uri=os.getenv('NEO4J_URI', 'bolt://neo4j:7687'),
            user=os.getenv('NEO4J_USER', 'neo4j'),
            password=os.getenv('NEO4J_PASSWORD', 'secgraph123'),
        ) as loader:
            loader.load_from_parquet(parquet_file, batch_size=500)
            stats = loader.get_graph_stats()
            logger.info(f"✓ Graph loaded: {stats}")

            # Detect clusters
            clusters = loader.detect_shared_device_clusters(min_accounts=3)
            logger.info(f"✓ Detected {len(clusters)} suspicious device clusters")
    except Exception as e:
        logger.error(f"✗ Graph loading failed: {e}")
        # Continue anyway for demo purposes

    # Step 3: Extract temporal features
    logger.info("\n[3/5] Extracting temporal features...")
    from reason_agent.ingest.temporal_preprocess import TemporalFeatureExtractor
    import pandas as pd

    df = pd.read_parquet(parquet_file)
    extractor = TemporalFeatureExtractor()
    features = extractor.extract_all_features(df)

    feature_file = Path("data/synthetic/temporal_features.parquet")
    features.to_parquet(feature_file, index=False)
    logger.info(f"✓ Features saved: {feature_file} ({len(features)} entities)")

    # Step 4: Build FAISS index
    logger.info("\n[4/5] Building FAISS vector index...")
    from reason_agent.tools.registry import ToolRegistry
    from reason_agent.embeddings.text_embedder import TextEmbedder
    from reason_agent.embeddings.faiss_index import FAISSIndex

    registry = ToolRegistry()
    tools_data = registry.get_embeddings_data()

    embedder = TextEmbedder(device="cpu")
    texts = [data['text'] for data in tools_data]
    embeddings = embedder.encode(texts, show_progress=True)

    index = FAISSIndex(dimension=embedder.embedding_dim, index_type="Flat", metric="cosine")
    metadata = [data['metadata'] for data in tools_data]
    ids = [data['id'] for data in tools_data]
    index.add(embeddings, metadata=metadata, ids=ids)

    output_path = Path("artifacts/faiss")
    output_path.mkdir(parents=True, exist_ok=True)
    index.save(
        index_path=output_path / "tool_index.faiss",
        metadata_path=output_path / "tool_metadata.pkl"
    )
    logger.info(f"✓ FAISS index built: {len(index)} vectors")

    # Step 5: Create sample audit pairs
    logger.info("\n[5/5] Creating sample audit pairs...")
    audit_path = Path("data/audits")
    audit_path.mkdir(parents=True, exist_ok=True)

    sample_pairs = []
    for i in range(10):
        pair = {
            'prompt': 'Detect multi-account abuse with shared devices',
            'chosen': 'Step 1: Check policy violations. Step 2: Identify device clusters. Conclusion: Found 5 accounts sharing device_123.',
            'rejected': 'Random guess without evidence.',
            'confidence': 0.9,
        }
        sample_pairs.append(pair)

    pairs_file = audit_path / "pairs.jsonl"
    with open(pairs_file, 'w') as f:
        for pair in sample_pairs:
            f.write(json.dumps(pair) + '\n')

    logger.info(f"✓ Created {len(sample_pairs)} sample preference pairs")

    # Create completion marker
    logger.info("\nCreating completion marker...")
    Path("data/.init_complete").touch()

    logger.info("\n" + "="*60)
    logger.info("✓ Data initialization complete!")
    logger.info("="*60)
    logger.info("\nYou can now:")
    logger.info("  - Access API at http://localhost:8000")
    logger.info("  - Access UI at http://localhost:8501")
    logger.info("  - View Neo4j at http://localhost:7474")


if __name__ == "__main__":
    main()
