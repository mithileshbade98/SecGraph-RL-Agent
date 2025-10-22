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
import random
from pathlib import Path
from datetime import datetime, timedelta
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

    # Use mode from environment: lightweight (200 events) / demo (800-1000 events) / full (3500+ events)
    mode = os.getenv('DATA_MODE', 'demo')  # Default to demo for populated UI
    use_parallel = os.getenv('USE_PARALLEL', 'true').lower() == 'true'
    logger.info(f"  Data mode: {mode}, Parallel processing: {use_parallel}")

    generator = SyntheticDataGenerator(seed=42)
    parquet_file = generator.generate_all_anomalies(
        output_dir="data/synthetic",
        mode=mode,
        use_parallel=use_parallel
    )
    log_progress(1, total_steps, f"✓ Step 1 complete: {parquet_file}", 100)

    # Step 2: Load into Neo4j
    log_progress(2, total_steps, "[2/5] Loading data into Neo4j...")
    from reason_agent.ingest.graph_loader import BiTemporalGraphLoader

    # Use optimal batch size based on mode
    batch_size = {
        'lightweight': 100,
        'demo': 200,  # Balanced for speed + memory
        'full': 500
    }.get(mode, 200)

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

            # Run fraud detection for demo and full modes (skip for lightweight only)
            if mode != 'lightweight':
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

    # Step 6: Generate sample reasoning traces for UI population
    logger.info("\n[6/6] Generating sample reasoning traces for UI...")
    from reason_agent.reasoning.trace_recorder import TraceRecorder

    trace_recorder = TraceRecorder()

    # Sample queries to execute
    sample_queries = [
        "Detect multi-account abuse with shared devices",
        "Find accounts with impossible travel patterns",
        "Identify credential stuffing attempts",
        "Detect free-tier farming abuse",
        "Find shared IP address fraud rings",
        "Identify velocity violations and rapid signups",
        "Detect bot network activity",
        "Find accounts with chargeback fraud patterns",
        "Identify session hijacking attempts",
        "Detect promo code stacking abuse",
    ]

    logger.info(f"  Generating {len(sample_queries)} sample traces...")
    for idx, query in enumerate(sample_queries):
        # Create mock trace (in production, this would call the planner)
        num_steps = random.randint(3, 5)
        steps = []

        for step_id in range(1, num_steps + 1):
            if step_id == 1:
                steps.append({
                    'step_id': step_id,
                    'thought': 'I should check for shared device violations',
                    'tool': 'check_shared_device_abuse',
                    'parameters': {'threshold': 3},
                    'result': {
                        'success': True,
                        'found': random.randint(3, 15),
                        'device_ids': [f'device_{random.randint(1000, 9999)}' for _ in range(random.randint(2, 5))]
                    },
                    'evidence': ['Neo4j query returned shared devices', 'Temporal clustering detected']
                })
            elif step_id == 2:
                steps.append({
                    'step_id': step_id,
                    'thought': 'Verify temporal burst patterns',
                    'tool': 'get_temporal_features',
                    'parameters': {'window_days': 7},
                    'result': {
                        'success': True,
                        'burstiness': round(random.uniform(0.6, 0.95), 2),
                        'velocity_score': round(random.uniform(0.7, 0.99), 2)
                    },
                    'evidence': ['High burstiness coefficient', 'Velocity anomaly detected']
                })
            else:
                steps.append({
                    'step_id': step_id,
                    'thought': 'Aggregate evidence and conclude',
                    'tool': 'aggregate_evidence',
                    'parameters': {},
                    'result': {
                        'success': True,
                        'conclusion': 'FRAUD DETECTED',
                        'confidence': round(random.uniform(0.85, 0.98), 2)
                    },
                    'evidence': ['Multiple fraud signals', 'High confidence score']
                })

        trace = {
            'trace_id': f"trace_{idx:04d}",
            'query': query,
            'timestamp': datetime.now() - timedelta(hours=idx),
            'num_steps': num_steps,
            'success': True,
            'steps': steps,
            'conclusion': 'Detected fraud with high confidence',
            'confidence': round(random.uniform(0.85, 0.98), 2),
            'rewards': {
                'process_reward': round(random.uniform(0.6, 0.9), 2),
                'final_reward': round(random.uniform(0.8, 1.0), 2),
                'total_reward': round(random.uniform(0.7, 0.95), 2)
            }
        }
        trace_recorder.save_trace(trace)

    logger.info(f"✓ Generated {len(sample_queries)} reasoning traces")

    # Create completion marker
    logger.info("\nCreating completion marker...")
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
