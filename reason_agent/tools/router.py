"""
Tool router using semantic search.

Routes queries to appropriate tools using vector similarity + optional rule-based fallback.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
from loguru import logger
from reason_agent.tools.registry import ToolRegistry, Tool
from reason_agent.embeddings.text_embedder import TextEmbedder
from reason_agent.embeddings.faiss_index import FAISSIndex


class ToolRouter:
    """Route queries to tools using semantic search."""

    def __init__(
        self,
        registry: ToolRegistry,
        embedder: Optional[TextEmbedder] = None,
        index: Optional[FAISSIndex] = None,
        method: str = "semantic",
        confidence_threshold: float = 0.75,
        fallback_to_rules: bool = True,
    ):
        """
        Initialize tool router.

        Args:
            registry: Tool registry
            embedder: Text embedder (if None, semantic search disabled)
            index: FAISS index (if None, semantic search disabled)
            method: semantic | rule | hybrid
            confidence_threshold: Minimum similarity for semantic routing
            fallback_to_rules: Use rule-based routing if semantic fails
        """
        self.registry = registry
        self.embedder = embedder
        self.index = index
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.fallback_to_rules = fallback_to_rules

        # Rule patterns (loaded from config or defaults)
        self.rules = [
            (r"check|verify|validate|policy", ["policy_checker", "velocity_checker"]),
            (r"cluster|group|related", ["cluster_detector", "entity_linker"]),
            (r"timeline|history|temporal|change", ["timeline_diff", "feature_extractor"]),
            (r"burst|storm|spike|rapid", ["burst_detector", "velocity_checker"]),
            (r"query|search|find", ["graph_search", "kusto_query"]),
        ]

    def route(
        self,
        query: str,
        top_k: int = 3,
        return_scores: bool = True,
    ) -> List[Tuple[Tool, float]]:
        """
        Route query to appropriate tools.

        Args:
            query: Query text
            top_k: Number of tools to return
            return_scores: Return confidence scores

        Returns:
            List of (Tool, score) tuples
        """
        if self.method == "semantic" or self.method == "hybrid":
            # Try semantic search
            if self.embedder is not None and self.index is not None:
                results = self._semantic_search(query, top_k)
                if results and results[0][1] >= self.confidence_threshold:
                    return results

        # Fallback to rule-based
        if self.method == "rule" or (self.method == "hybrid" and self.fallback_to_rules):
            return self._rule_based_search(query, top_k)

        # No results
        return []

    def _semantic_search(
        self,
        query: str,
        top_k: int,
    ) -> List[Tuple[Tool, float]]:
        """Semantic search using embeddings."""
        # Encode query
        query_emb = self.embedder.encode(query)

        # Search index
        distances, indices, metadata = self.index.search(query_emb, k=top_k, return_metadata=True)

        # Convert to tools
        results = []
        for dist, idx, meta in zip(distances[0], indices[0], metadata[0]):
            tool_name = meta.get('id')
            tool = self.registry.get_tool(tool_name)
            if tool:
                # Convert distance to similarity (assuming normalized embeddings)
                similarity = float(dist)
                results.append((tool, similarity))

        return results

    def _rule_based_search(
        self,
        query: str,
        top_k: int,
    ) -> List[Tuple[Tool, float]]:
        """Rule-based search using regex patterns."""
        query_lower = query.lower()
        matched_tools = []

        for pattern, tool_names in self.rules:
            if re.search(pattern, query_lower):
                for tool_name in tool_names:
                    tool = self.registry.get_tool(tool_name)
                    if tool and tool not in [t for t, _ in matched_tools]:
                        # Assign score based on match position (earlier = better)
                        score = 0.8  # Default rule-based score
                        matched_tools.append((tool, score))

        # Limit to top_k
        return matched_tools[:top_k]

    def get_best_tool(self, query: str) -> Optional[Tool]:
        """Get single best tool for query."""
        results = self.route(query, top_k=1)
        if results:
            return results[0][0]
        return None


def main():
    """Test tool router."""
    import argparse

    parser = argparse.ArgumentParser(description="Test tool router")
    parser.add_argument("--query", type=str, required=True, help="Query text")
    parser.add_argument("--method", default="rule", choices=["semantic", "rule", "hybrid"])
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    # Create registry
    registry = ToolRegistry()

    # Create router (rule-based for simple test)
    router = ToolRouter(registry, method=args.method)

    # Route query
    results = router.route(args.query, top_k=args.top_k)

    logger.info(f"\nQuery: {args.query}")
    logger.info(f"Top-{args.top_k} tools:")
    for i, (tool, score) in enumerate(results):
        logger.info(f"  {i+1}. {tool.name} (score: {score:.3f})")
        logger.info(f"      {tool.description}")


if __name__ == "__main__":
    main()
