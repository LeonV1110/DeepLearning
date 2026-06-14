"""Graph attention model for MEG classification.

This keeps the historical ``CNNBaseline1D`` class name for compatibility, but
the implementation now treats each of the 248 sensors as a graph node and uses
attention to mix information across sensors after encoding each sensor's time
series into a node embedding.

Input shape: ``(batch, 248, time)``.
Output shape: ``(batch, num_classes)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalSensorEncoder(nn.Module):
    """Encode each sensor's time series into a compact node embedding."""

    def __init__(
        self,
        hidden_dim: int = 32,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()

        mid_dim = max(hidden_dim // 2, 8)
        second_kernel = max(3, kernel_size - 2)

        self.encoder = nn.Sequential(
            nn.Conv1d(1, mid_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                mid_dim,
                hidden_dim,
                kernel_size=second_kernel,
                padding=second_kernel // 2,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_sensors, time_steps = x.shape
        x = x.reshape(batch_size * num_sensors, 1, time_steps)
        x = self.encoder(x)
        return x.flatten(1).reshape(batch_size, num_sensors, -1)


class GraphAttentionBlock(nn.Module):
    """Dense graph-attention block with a learnable sensor-to-sensor bias."""

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        node_embedding_dim: int = 16,
    ):
        super().__init__()

        if output_dim % num_heads != 0:
            raise ValueError("output_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.node_embedding_dim = node_embedding_dim

        self.query = nn.Linear(input_dim, output_dim, bias=False)
        self.key = nn.Linear(input_dim, output_dim, bias=False)
        self.value = nn.Linear(input_dim, output_dim, bias=False)

        self.node_embeddings = nn.Parameter(
            torch.randn(num_nodes, node_embedding_dim) * 0.02
        )
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.residual_proj = (
            nn.Identity()
            if input_dim == output_dim
            else nn.Linear(input_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape

        residual = self.residual_proj(x)

        query = self.query(x).reshape(
            batch_size, num_nodes, self.num_heads, self.head_dim
        )
        key = self.key(x).reshape(batch_size, num_nodes, self.num_heads, self.head_dim)
        value = self.value(x).reshape(
            batch_size, num_nodes, self.num_heads, self.head_dim
        )

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attention_logits = torch.matmul(query, key.transpose(-1, -2))
        attention_logits = attention_logits / (self.head_dim**0.5)

        graph_bias = torch.matmul(self.node_embeddings, self.node_embeddings.T)
        graph_bias = graph_bias / (self.node_embedding_dim**0.5)
        attention_logits = attention_logits + graph_bias.unsqueeze(0).unsqueeze(0)

        attention = torch.softmax(attention_logits, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, value)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)

        out = self.out_proj(out)
        out = self.dropout(out)
        out = self.norm(out + residual)
        return self.activation(out)


class CNNBaseline1D(nn.Module):
    """Backwards-compatible graph attention classifier.

    The historical ``hidden_channels`` argument now controls the temporal encoder
    width, ``kernel_size`` controls the temporal convolutions, and the graph
    branch is kept intentionally light so it behaves like a small residual
    correction on top of the temporal classifier.
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_sensors: int = 248,
        hidden_channels: int = 32,
        kernel_size: int = 7,
        dropout: float = 0.3,
        graph_hidden: int | None = None,
        num_heads: int = 1,
        num_graph_blocks: int = 1,
        graph_gate_init: float = -3.0,
    ):
        super().__init__()

        if num_graph_blocks < 1:
            raise ValueError("num_graph_blocks must be at least 1")

        graph_hidden = graph_hidden or hidden_channels

        self.temporal_encoder = TemporalSensorEncoder(
            hidden_dim=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        graph_blocks = []
        for block_index in range(num_graph_blocks):
            input_dim = hidden_channels if block_index == 0 else graph_hidden
            graph_blocks.append(
                GraphAttentionBlock(
                    num_nodes=num_sensors,
                    input_dim=input_dim,
                    output_dim=graph_hidden,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )

        self.graph_blocks = nn.ModuleList(graph_blocks)

        self.temporal_head = nn.Linear(hidden_channels * 2, num_classes)
        self.graph_adapter = nn.Sequential(
            nn.Linear(graph_hidden * 2, hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
        )
        nn.init.zeros_(self.graph_adapter[-1].weight)
        nn.init.zeros_(self.graph_adapter[-1].bias)
        self.graph_gate = nn.Parameter(torch.tensor(graph_gate_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (batch, sensors, time), got {tuple(x.shape)}"
            )

        x = self.temporal_encoder(x)
        temporal_mean = x.mean(dim=1)
        temporal_max = x.max(dim=1).values

        for graph_block in self.graph_blocks:
            x = graph_block(x)

        graph_mean = x.mean(dim=1)
        graph_max = x.max(dim=1).values

        temporal_features = torch.cat([temporal_mean, temporal_max], dim=1)
        graph_features = torch.cat([graph_mean, graph_max], dim=1)
        graph_delta = self.graph_adapter(graph_features)

        fused_features = temporal_features + torch.sigmoid(self.graph_gate) * graph_delta
        return self.temporal_head(fused_features)