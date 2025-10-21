"""Tests for retrieval and routing."""

import pytest
import numpy as np

from reason_agent.tools.registry import ToolRegistry
from reason_agent.tools.router import ToolRouter
from reason_agent.embeddings.text_embedder import TextEmbedder
from reason_agent.embeddings.faiss_index import FAISSIndex


def test_tool_registry():
    """Test tool registry loading."""
    registry = ToolRegistry()

    # Check tools loaded
    assert len(registry.get_all_tools()) > 0

    # Get specific tool
    tool = registry.get_tool('policy_checker')
    assert tool is not None
    assert tool.name == 'policy_checker'
    assert len(tool.parameters) > 0


def test_tool_router_rule_based():
    """Test rule-based tool routing."""
    registry = ToolRegistry()
    router = ToolRouter(registry, method="rule")

    # Test query routing
    results = router.route("check policy violations", top_k=3)
    assert len(results) > 0

    # Should find policy_checker
    tool_names = [tool.name for tool, _ in results]
    assert 'policy_checker' in tool_names or 'velocity_checker' in tool_names


def test_faiss_index():
    """Test FAISS index operations."""
    dim = 384
    index = FAISSIndex(dimension=dim, index_type="Flat", metric="cosine")

    # Create random embeddings
    embeddings = np.random.randn(10, dim).astype(np.float32)
    # Normalize for cosine
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    metadata = [{'id': i, 'text': f'doc_{i}'} for i in range(10)]
    index.add(embeddings, metadata=metadata)

    # Search
    query = embeddings[0]
    distances, indices, result_meta = index.search(query, k=3)

    assert len(distances[0]) == 3
    assert indices[0][0] == 0  # First result should be itself
    assert distances[0][0] > 0.99  # Cosine similarity ~ 1.0


def test_text_embedder():
    """Test text embedding."""
    embedder = TextEmbedder(device="cpu")

    # Encode text
    text = "Check policy violations"
    embedding = embedder.encode(text)

    assert embedding.shape == (1, embedder.embedding_dim)
    assert embedder.embedding_dim == 384  # MiniLM dimension

    # Similarity
    sim = embedder.similarity("check policy", "policy verification")
    assert 0.0 <= sim <= 1.0
