#!/usr/bin/env python
"""CLI for building FAISS vector index from tool cards."""

import click
from pathlib import Path
from loguru import logger

from reason_agent.tools.registry import ToolRegistry
from reason_agent.embeddings.text_embedder import TextEmbedder
from reason_agent.embeddings.faiss_index import FAISSIndex


@click.command()
@click.option('--config', default='configs/embeddings.yaml', help='Config file')
@click.option('--output-dir', default='artifacts/faiss', help='Output directory')
def main(config, output_dir):
    """Build FAISS index from tool cards."""
    logger.info("Building FAISS index...")

    # Load tools
    registry = ToolRegistry()
    tools_data = registry.get_embeddings_data()
    logger.info(f"Loaded {len(tools_data)} tools")

    # Create embedder
    embedder = TextEmbedder(device="cpu")

    # Embed tools
    texts = [data['text'] for data in tools_data]
    embeddings = embedder.encode(texts, show_progress=True)

    # Create FAISS index
    index = FAISSIndex(dimension=embedder.embedding_dim, index_type="Flat", metric="cosine")

    # Add embeddings with metadata
    metadata = [data['metadata'] for data in tools_data]
    ids = [data['id'] for data in tools_data]
    index.add(embeddings, metadata=metadata, ids=ids)

    # Save index
    output_path = Path(output_dir)
    index.save(
        index_path=output_path / "tool_index.faiss",
        metadata_path=output_path / "tool_metadata.pkl"
    )

    logger.info(f"✅ Index built: {len(index)} vectors")


if __name__ == '__main__':
    main()
