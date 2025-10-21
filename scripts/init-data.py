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
    import time

    start_time = time.time()

    def log_progress(step, total_steps, message, substep_pct=0):
        """Log progress with percentage."""
        base_pct = ((step - 1) / total_steps) * 100
        step_size = 100 / total_steps
        current_pct = base_pct + (step_size * substep_pct / 100)
        elapsed = time.time() - start_time
        logger.info(f"[{current_pct:5.1f}%] {message} (elapsed: {elapsed:.1f}s)")

    logger.info("="*60)
    logger.info("SecGraph-RL Agent - Data Initialization")
    logger.info("="*60)

    total_steps = 5

    # Step 0: Wait for Neo4j
    log_progress(0, total_steps, "Waiting for Neo4j...")
    if not wait_for_neo4j():
        logger.error("Neo4j not available, exiting")
        sys.exit(1)

    # Step 1: Generate synthetic data
    log_progress(1, total_steps, "[1/5] Generating synthetic security events...")
    from reason_agent.ingest.synthetic_generator import SyntheticDataGenerator

    # Use lightweight mode for low-resource systems (M1 Mac, 8GB RAM)
    lightweight = os.getenv('LIGHTWEIGHT_MODE', 'true').lower() == 'true'
    logger.info(f"  Lightweight mode: {lightweight}")

    generator = SyntheticDataGenerator(seed=42)
    parquet_file = generator.generate_all_anomalies(
        output_dir="data/synthetic",
        lightweight=lightweight
    )
    log_progress(1, total_steps, f"✓ Step 1 complete: {parquet_file}", 100)

    # Step 2: Load into Neo4j
    log_progress(2, total_steps, "[2/5] Loading data into Neo4j...")
    from reason_agent.ingest.graph_loader import BiTemporalGraphLoader

    # Use smaller batch size in lightweight mode
    batch_size = 100 if lightweight else 500

    try:
        with BiTemporalGraphLoader(
            uri=os.getenv('NEO4J_URI', 'bolt://neo4j:7687'),
            user=os.getenv('NEO4J_USER', 'neo4j'),
            password=os.getenv('NEO4J_PASSWORD', 'secgraph123'),
        ) as loader:
            logger.info(f"  Loading with batch_size={batch_size}...")
            log_progress(2, total_steps, "  Loading parquet data...", 30)
            loader.load_from_parquet(parquet_file, batch_size=batch_size)
            log_progress(2, total_steps, "  Computing graph stats...", 70)
            stats = loader.get_graph_stats()
            logger.info(f"  Graph loaded: {stats}")

            # Skip expensive fraud detection in lightweight mode
            if not lightweight:
                log_progress(2, total_steps, "  Detecting fraud clusters...", 85)
                clusters = loader.detect_shared_device_clusters(min_accounts=3)
                logger.info(f"  Detected {len(clusters)} suspicious device clusters")
            else:
                logger.info("  Skipping fraud cluster detection (lightweight mode)")

            log_progress(2, total_steps, "✓ Step 2 complete: Graph loaded", 100)
    except Exception as e:
        logger.error(f"✗ Graph loading failed: {e}")
        log_progress(2, total_steps, "✗ Step 2 failed (continuing anyway)", 100)
        # Continue anyway for demo purposes

    # Step 3: Extract temporal features
    log_progress(3, total_steps, "[3/5] Extracting temporal features...")
    from reason_agent.ingest.temporal_preprocess import TemporalFeatureExtractor
    import pandas as pd

    log_progress(3, total_steps, "  Loading parquet data...", 20)
    df = pd.read_parquet(parquet_file)
    log_progress(3, total_steps, "  Computing rolling windows...", 40)
    extractor = TemporalFeatureExtractor()
    log_progress(3, total_steps, "  Extracting all features...", 60)
    features = extractor.extract_all_features(df)

    feature_file = Path("data/synthetic/temporal_features.parquet")
    features.to_parquet(feature_file, index=False)
    log_progress(3, total_steps, f"✓ Step 3 complete: {len(features)} entities", 100)

    # Step 4: Build FAISS index
    log_progress(4, total_steps, "[4/5] Building FAISS vector index...")
    from reason_agent.tools.registry import ToolRegistry
    from reason_agent.embeddings.text_embedder import TextEmbedder
    from reason_agent.embeddings.faiss_index import FAISSIndex

    log_progress(4, total_steps, "  Loading tool registry...", 20)
    registry = ToolRegistry()
    tools_data = registry.get_embeddings_data()

    log_progress(4, total_steps, "  Encoding texts...", 40)
    embedder = TextEmbedder(device="cpu")
    texts = [data['text'] for data in tools_data]
    embeddings = embedder.encode(texts, show_progress=False)

    log_progress(4, total_steps, "  Building FAISS index...", 70)
    index = FAISSIndex(dimension=embedder.embedding_dim, index_type="Flat", metric="cosine")
    metadata = [data['metadata'] for data in tools_data]
    ids = [data['id'] for data in tools_data]
    index.add(embeddings, metadata=metadata, ids=ids)

    log_progress(4, total_steps, "  Saving index to disk...", 90)
    output_path = Path("artifacts/faiss")
    output_path.mkdir(parents=True, exist_ok=True)
    index.save(
        index_path=output_path / "tool_index.faiss",
        metadata_path=output_path / "tool_metadata.pkl"
    )
    log_progress(4, total_steps, f"✓ Step 4 complete: {len(index)} vectors", 100)

    # Step 5: Create sample audit pairs
    log_progress(5, total_steps, "[5/5] Creating sample audit pairs...")
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

    log_progress(5, total_steps, f"✓ Step 5 complete: {len(sample_pairs)} preference pairs", 100)

    # Create completion marker
    log_progress(5, total_steps, "Creating completion marker...", 100)
    Path("data/.init_complete").touch()

    total_time = time.time() - start_time
    logger.info("\n" + "="*60)
    logger.info(f"✓ Data initialization complete! (Total time: {total_time:.1f}s)")
    logger.info("="*60)
    logger.info("\nYou can now:")
    logger.info("  - Access API at http://localhost:8000")
    logger.info("  - Access UI at http://localhost:8501")
    logger.info("  - View Neo4j at http://localhost:7474")


if __name__ == "__main__":
    main()
