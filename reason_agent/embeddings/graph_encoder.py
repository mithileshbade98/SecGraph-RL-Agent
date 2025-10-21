"""
Graph neural network encoders for temporal identity graphs.

Implements GraphSAGE and GAT interfaces with temporal encoding support.
Why: Graph structure + temporal patterns are key for fraud/anomaly detection.
Source: Graph anomaly detection surveys (2024), GNN-for-TS (arXiv:2404.16036)
"""

from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import Data
import numpy as np
from loguru import logger


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE encoder for node embeddings.

    Aggregates neighbor features via mean/max/lstm pooling.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        aggregator: str = "mean",
    ):
        """
        Initialize GraphSAGE encoder.

        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output embedding dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
            aggregator: mean | max | lstm
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregator))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))

        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggregator))

        logger.info(f"GraphSAGE: {num_layers} layers, {aggregator} aggregation")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)

        Returns:
            Node embeddings (num_nodes, out_channels)
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder.

    Uses multi-head attention to aggregate neighbor features.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        """
        Initialize GAT encoder.

        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output embedding dimension
            num_layers: Number of GNN layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            concat: Concatenate heads (vs average)
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=concat)
        )

        # Hidden layers
        in_ch = hidden_channels * num_heads if concat else hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(in_ch, hidden_channels, heads=num_heads, dropout=dropout, concat=concat)
            )

        # Output layer (average heads for final output)
        self.convs.append(
            GATConv(in_ch, out_channels, heads=1, dropout=dropout, concat=False)
        )

        logger.info(f"GAT: {num_layers} layers, {num_heads} heads")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)

        Returns:
            Node embeddings (num_nodes, out_channels)
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class TemporalEncoder(nn.Module):
    """
    Temporal encoding for time-aware GNNs.

    Encodes temporal information using harmonic functions or learned embeddings.
    """

    def __init__(
        self,
        method: str = "harmonic",
        time_dim: int = 16,
        max_freq: float = 10000.0,
    ):
        """
        Initialize temporal encoder.

        Args:
            method: harmonic | learned
            time_dim: Temporal encoding dimension
            max_freq: Maximum frequency for harmonic encoding
        """
        super().__init__()
        self.method = method
        self.time_dim = time_dim
        self.max_freq = max_freq

        if method == "learned":
            self.time_embedding = nn.Embedding(10000, time_dim)  # Support up to 10k time steps

        logger.info(f"Temporal encoding: {method}, dim={time_dim}")

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Encode timestamps.

        Args:
            timestamps: Timestamps (num_events,) - can be discrete or continuous

        Returns:
            Temporal encodings (num_events, time_dim)
        """
        if self.method == "harmonic":
            # Harmonic encoding (similar to positional encoding in Transformers)
            # Supports continuous timestamps
            device = timestamps.device
            freqs = torch.arange(self.time_dim // 2, device=device).float()
            freqs = freqs / (self.time_dim / 2) * np.log(self.max_freq)
            freqs = torch.exp(freqs)

            # Expand timestamps
            t = timestamps.unsqueeze(-1)  # (num_events, 1)
            freqs = freqs.unsqueeze(0)    # (1, time_dim//2)

            # Compute sin and cos
            angles = t * freqs  # (num_events, time_dim//2)
            encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
            return encoding

        elif self.method == "learned":
            # Learned embeddings (requires discrete time steps)
            timestamps_int = timestamps.long()
            return self.time_embedding(timestamps_int)

        else:
            raise ValueError(f"Unknown temporal encoding method: {self.method}")


class TemporalGNNEncoder(nn.Module):
    """
    Temporal GNN encoder combining spatial and temporal information.

    This is a simplified interface to TGAT/TGN-style models.
    For production, consider using PyTorch Geometric Temporal.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 128,
        time_dim: int = 16,
        num_layers: int = 2,
        gnn_type: str = "graphsage",
        temporal_method: str = "harmonic",
    ):
        """
        Initialize temporal GNN.

        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension (0 if no edge features)
            hidden_dim: Hidden dimension
            time_dim: Temporal encoding dimension
            num_layers: Number of layers
            gnn_type: graphsage | gat
            temporal_method: harmonic | learned
        """
        super().__init__()

        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(method=temporal_method, time_dim=time_dim)

        # Combine node features with temporal encoding
        combined_dim = node_dim + time_dim

        # Spatial GNN
        if gnn_type == "graphsage":
            self.gnn = GraphSAGEEncoder(
                in_channels=combined_dim,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                num_layers=num_layers,
            )
        elif gnn_type == "gat":
            self.gnn = GATEncoder(
                in_channels=combined_dim,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                num_layers=num_layers,
            )
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        logger.info(f"Temporal GNN: {gnn_type} + {temporal_method} encoding")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, node_dim)
            edge_index: Edge indices (2, num_edges)
            timestamps: Node timestamps (num_nodes,)

        Returns:
            Node embeddings (num_nodes, hidden_dim)
        """
        # Encode temporal information
        time_encoding = self.temporal_encoder(timestamps)

        # Concatenate node features with temporal encoding
        x_combined = torch.cat([x, time_encoding], dim=-1)

        # Apply spatial GNN
        embeddings = self.gnn(x_combined, edge_index)

        return embeddings


class GraphEncoderPipeline:
    """
    Full pipeline for graph encoding: feature extraction → GNN → embeddings.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        """
        Initialize graph encoder pipeline.

        Args:
            config: Configuration dict from graph.yaml
            device: cpu | cuda | mps
        """
        self.config = config
        self.device = device

        # Extract config
        graph_ml_config = config.get("graph_ml", {})
        node_dim = graph_ml_config.get("node_embedding_dim", 128)
        hidden_dim = graph_ml_config.get("node_embedding_dim", 128)
        num_layers = graph_ml_config.get("num_layers", 2)

        # Create model (default to GraphSAGE)
        graphsage_config = config.get("graphsage", {})
        self.model = GraphSAGEEncoder(
            in_channels=node_dim,
            hidden_channels=graphsage_config.get("hidden_channels", 128),
            out_channels=hidden_dim,
            num_layers=num_layers,
        ).to(device)

        logger.info(f"Graph encoder pipeline initialized on {device}")

    def encode_graph(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
    ) -> np.ndarray:
        """
        Encode graph to node embeddings.

        Args:
            node_features: Node feature matrix (num_nodes, feature_dim)
            edge_index: Edge index array (2, num_edges)

        Returns:
            Node embeddings (num_nodes, embedding_dim)
        """
        # Convert to tensors
        x = torch.from_numpy(node_features).float().to(self.device)
        edge_index_t = torch.from_numpy(edge_index).long().to(self.device)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(x, edge_index_t)

        return embeddings.cpu().numpy()


def main():
    """Test graph encoders."""
    import argparse

    parser = argparse.ArgumentParser(description="Test graph encoders")
    parser.add_argument("--model", default="graphsage", choices=["graphsage", "gat", "temporal"])
    parser.add_argument("--num-nodes", type=int, default=100)
    parser.add_argument("--num-edges", type=int, default=500)
    parser.add_argument("--feature-dim", type=int, default=64)
    args = parser.parse_args()

    # Generate random graph
    num_nodes = args.num_nodes
    num_edges = args.num_edges
    feature_dim = args.feature_dim

    x = torch.randn(num_nodes, feature_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Test model
    if args.model == "graphsage":
        model = GraphSAGEEncoder(in_channels=feature_dim, hidden_channels=128, out_channels=128)
        embeddings = model(x, edge_index)
        logger.info(f"GraphSAGE output shape: {embeddings.shape}")

    elif args.model == "gat":
        model = GATEncoder(in_channels=feature_dim, hidden_channels=128, out_channels=128)
        embeddings = model(x, edge_index)
        logger.info(f"GAT output shape: {embeddings.shape}")

    elif args.model == "temporal":
        timestamps = torch.arange(num_nodes).float()
        model = TemporalGNNEncoder(node_dim=feature_dim, hidden_dim=128)
        embeddings = model(x, edge_index, timestamps)
        logger.info(f"Temporal GNN output shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
