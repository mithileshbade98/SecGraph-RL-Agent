"""
FAISS vector index for fast similarity search.

Supports Flat (exact), IVF (approximate), and HNSW indices on CPU.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import faiss
from pathlib import Path
import pickle
from loguru import logger


class FAISSIndex:
    """FAISS-based vector search index."""

    def __init__(
        self,
        dimension: int,
        index_type: str = "Flat",
        metric: str = "cosine",
        nlist: int = 100,
        nprobe: int = 10,
        ef_construction: int = 200,
        ef_search: int = 50,
    ):
        """
        Initialize FAISS index.

        Args:
            dimension: Embedding dimension
            index_type: Flat | IVF | HNSW
            metric: cosine | l2 | ip (inner product)
            nlist: Number of clusters for IVF
            nprobe: Number of clusters to search in IVF
            ef_construction: Construction parameter for HNSW
            ef_search: Search parameter for HNSW
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.nlist = nlist
        self.nprobe = nprobe
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        # Metadata storage (FAISS only stores vectors, not metadata)
        self.metadata: List[Dict[str, Any]] = []
        self.id_to_idx: Dict[str, int] = {}

        # Create index
        self.index = self._create_index()
        logger.info(f"Created FAISS {index_type} index (dim={dimension}, metric={metric})")

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration."""
        if self.metric == "cosine":
            # For cosine similarity, we assume embeddings are L2-normalized
            # Then inner product = cosine similarity
            base_metric = faiss.METRIC_INNER_PRODUCT
        elif self.metric == "l2":
            base_metric = faiss.METRIC_L2
        elif self.metric == "ip":
            base_metric = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        if self.index_type == "Flat":
            # Exact search (brute force)
            index = faiss.IndexFlatIP(self.dimension) if base_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.dimension)

        elif self.index_type == "IVF":
            # Inverted file index (approximate)
            quantizer = faiss.IndexFlatIP(self.dimension) if base_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, base_metric)
            index.nprobe = self.nprobe

        elif self.index_type == "HNSW":
            # Hierarchical navigable small world (approximate, fast)
            index = faiss.IndexHNSWFlat(self.dimension, 32, base_metric)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        return index

    def add(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ):
        """
        Add vectors to the index.

        Args:
            embeddings: Array of shape (num_vectors, dimension)
            metadata: List of metadata dicts (one per vector)
            ids: List of unique IDs (optional, for ID-based lookup)
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        num_vectors = embeddings.shape[0]
        assert embeddings.shape[1] == self.dimension, f"Expected dim {self.dimension}, got {embeddings.shape[1]}"

        # Train index if needed (IVF requires training)
        if self.index_type == "IVF" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)

        # Add vectors
        start_idx = len(self.metadata)
        self.index.add(embeddings.astype(np.float32))

        # Store metadata
        if metadata is None:
            metadata = [{} for _ in range(num_vectors)]

        self.metadata.extend(metadata)

        # Store ID mappings
        if ids is not None:
            for i, id_val in enumerate(ids):
                self.id_to_idx[id_val] = start_idx + i

        logger.info(f"Added {num_vectors} vectors to index (total: {len(self.metadata)})")

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        return_metadata: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict[str, Any]]]]:
        """
        Search for k nearest neighbors.

        Args:
            query: Query vector (shape: (dimension,) or (num_queries, dimension))
            k: Number of neighbors
            return_metadata: Return metadata for results

        Returns:
            (distances, indices, metadata)
            - distances: Array of shape (num_queries, k)
            - indices: Array of shape (num_queries, k)
            - metadata: List of lists of metadata dicts (if return_metadata=True)
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)

        assert query.shape[1] == self.dimension

        distances, indices = self.index.search(query.astype(np.float32), k)

        if return_metadata:
            result_metadata = []
            for idx_list in indices:
                result_metadata.append([
                    self.metadata[idx] if 0 <= idx < len(self.metadata) else {}
                    for idx in idx_list
                ])
            return distances, indices, result_metadata
        else:
            return distances, indices, None

    def search_by_id(self, id_val: str, k: int = 5) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict[str, Any]]]]:
        """Search for neighbors of a vector identified by ID."""
        if id_val not in self.id_to_idx:
            raise ValueError(f"ID {id_val} not found in index")

        idx = self.id_to_idx[id_val]
        vector = self.index.reconstruct(idx)
        return self.search(vector, k=k)

    def save(self, index_path: Path, metadata_path: Path):
        """Save index and metadata to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")

        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'id_to_idx': self.id_to_idx,
                'config': {
                    'dimension': self.dimension,
                    'index_type': self.index_type,
                    'metric': self.metric,
                }
            }, f)
        logger.info(f"Saved metadata to {metadata_path}")

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> 'FAISSIndex':
        """Load index and metadata from disk."""
        # Load metadata first to get config
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)

        # Create instance
        config = data['config']
        instance = cls(
            dimension=config['dimension'],
            index_type=config['index_type'],
            metric=config['metric'],
        )

        # Load FAISS index
        instance.index = faiss.read_index(str(index_path))
        instance.metadata = data['metadata']
        instance.id_to_idx = data['id_to_idx']

        logger.info(f"Loaded FAISS index from {index_path} ({len(instance.metadata)} vectors)")
        return instance

    def __len__(self) -> int:
        """Return number of vectors in index."""
        return len(self.metadata)


def main():
    """CLI for FAISS index operations."""
    import argparse

    parser = argparse.ArgumentParser(description="FAISS vector index")
    parser.add_argument("--dimension", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--index-type", default="Flat", choices=["Flat", "IVF", "HNSW"],
                        help="Index type")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "l2", "ip"],
                        help="Distance metric")
    parser.add_argument("--test", action="store_true", help="Run test")
    args = parser.parse_args()

    # Create index
    index = FAISSIndex(
        dimension=args.dimension,
        index_type=args.index_type,
        metric=args.metric,
    )

    if args.test:
        # Generate random vectors for testing
        logger.info("Running test...")
        num_vectors = 1000
        vectors = np.random.randn(num_vectors, args.dimension).astype(np.float32)

        # Normalize for cosine similarity
        if args.metric == "cosine":
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        metadata = [{'id': i, 'text': f'doc_{i}'} for i in range(num_vectors)]

        # Add vectors
        index.add(vectors, metadata=metadata)

        # Search
        query = vectors[0]  # Use first vector as query
        distances, indices, result_meta = index.search(query, k=5)

        logger.info("\nSearch results:")
        for i, (dist, idx, meta) in enumerate(zip(distances[0], indices[0], result_meta[0])):
            logger.info(f"{i+1}. Index {idx}, Distance {dist:.4f}, Metadata: {meta}")


if __name__ == "__main__":
    main()
