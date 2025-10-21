"""
Text embedding using sentence-transformers (MiniLM/GTE).

CPU/MPS-friendly embeddings for semantic search over tool cards and documents.
"""

from typing import List, Optional, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger
from pathlib import Path


class TextEmbedder:
    """Text embedding with sentence-transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize text embedder.

        Args:
            model_name: HuggingFace model name (default: MiniLM - fast, 384-dim)
            device: cpu | mps | cuda
            batch_size: Batch size for encoding
            max_length: Max sequence length
            normalize: L2 normalize embeddings (recommended for cosine similarity)
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize

        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)
        self.model.max_seq_length = max_length

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        """
        Encode text(s) to embeddings.

        Args:
            texts: Single text or list of texts
            show_progress: Show progress bar
            convert_to_numpy: Return numpy array (vs torch tensor)

        Returns:
            Embeddings array of shape (num_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=self.normalize,
        )

        return embeddings

    def encode_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode a batch of texts."""
        return self.encode(texts, show_progress=show_progress)

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        return float(np.dot(emb1[0], emb2[0]))

    def batch_similarity(self, query: str, candidates: List[str]) -> np.ndarray:
        """
        Compute similarities between a query and multiple candidates.

        Args:
            query: Query text
            candidates: List of candidate texts

        Returns:
            Array of similarity scores (length = len(candidates))
        """
        query_emb = self.encode(query)
        candidate_embs = self.encode(candidates, show_progress=False)

        # Cosine similarity (assuming normalized embeddings)
        similarities = np.dot(candidate_embs, query_emb[0])
        return similarities

    def get_top_k(
        self,
        query: str,
        candidates: List[str],
        k: int = 5,
        return_scores: bool = True,
    ) -> Union[List[int], tuple]:
        """
        Get top-k most similar candidates to query.

        Args:
            query: Query text
            candidates: List of candidate texts
            k: Number of top results
            return_scores: Also return similarity scores

        Returns:
            If return_scores=True: (indices, scores)
            If return_scores=False: indices
        """
        similarities = self.batch_similarity(query, candidates)
        top_k_indices = np.argsort(similarities)[::-1][:k]

        if return_scores:
            top_k_scores = similarities[top_k_indices]
            return top_k_indices, top_k_scores
        else:
            return top_k_indices

    def save_cache(self, texts: List[str], output_path: Path):
        """
        Pre-compute and save embeddings for a text corpus.

        Args:
            texts: List of texts to embed
            output_path: Path to save embeddings (.npy)
        """
        logger.info(f"Encoding {len(texts)} texts...")
        embeddings = self.encode(texts, show_progress=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embeddings)
        logger.info(f"Saved embeddings to {output_path}")

    def load_cache(self, cache_path: Path) -> np.ndarray:
        """Load pre-computed embeddings."""
        logger.info(f"Loading embeddings from {cache_path}")
        return np.load(cache_path)


def main():
    """CLI for text embedding."""
    import argparse

    parser = argparse.ArgumentParser(description="Text embedding with sentence-transformers")
    parser.add_argument("--text", type=str, help="Text to embed")
    parser.add_argument("--query", type=str, help="Query text for similarity search")
    parser.add_argument("--candidates", type=str, nargs="+", help="Candidate texts")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Model name")
    parser.add_argument("--device", default="cpu", help="Device (cpu/mps/cuda)")
    parser.add_argument("--k", type=int, default=5, help="Top-k results")
    args = parser.parse_args()

    embedder = TextEmbedder(model_name=args.model, device=args.device)

    if args.text:
        # Encode single text
        embedding = embedder.encode(args.text)
        logger.info(f"Embedding shape: {embedding.shape}")
        logger.info(f"First 10 dims: {embedding[0][:10]}")

    elif args.query and args.candidates:
        # Similarity search
        indices, scores = embedder.get_top_k(args.query, args.candidates, k=args.k)
        logger.info(f"\nTop-{args.k} results for query: '{args.query}'")
        for i, (idx, score) in enumerate(zip(indices, scores)):
            logger.info(f"{i+1}. {args.candidates[idx]} (score: {score:.4f})")

    else:
        logger.error("Provide either --text or --query with --candidates")


if __name__ == "__main__":
    main()
