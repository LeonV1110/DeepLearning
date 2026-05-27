"""
Graph attention model for MEG classification.

The model consumes the same input format as the TCN pipeline:
        (batch, 248 sensors, time)

It can be trained with the existing train_one_epoch / evaluate helpers and
produces class logits for the same labels as the TCN model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalSensorEncoder(nn.Module):
    """Encode each sensor's time series into a node embedding."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_sensors, time_steps = x.shape
        x = x.reshape(batch_size * num_sensors, 1, time_steps)
        x = self.encoder(x)
        x = x.flatten(1)
        return x.reshape(batch_size, num_sensors, -1)


class GraphAttentionBlock(nn.Module):
    """Dense graph-attention block with a learnable graph prior."""

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        if output_dim % num_heads != 0:
            raise ValueError("output_dim must be divisible by num_heads")

        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.query = nn.Linear(input_dim, output_dim, bias=False)
        self.key = nn.Linear(input_dim, output_dim, bias=False)
        self.value = nn.Linear(input_dim, output_dim, bias=False)

        self.graph_bias = nn.Parameter(torch.zeros(num_nodes, num_nodes))
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
        attention_logits = attention_logits + self.graph_bias.unsqueeze(0).unsqueeze(0)

        attention = torch.softmax(attention_logits, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, value)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)

        out = self.out_proj(out)
        out = self.dropout(out)
        out = self.norm(out + residual)
        out = self.activation(out)

        return out


class MEGGraphAttentionNetwork(nn.Module):
    """Graph attention network for MEG classification."""

    def __init__(
        self,
        num_classes: int,
        num_sensors: int = 248,
        temporal_hidden: int = 32,
        graph_hidden: int = 64,
        num_heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.temporal_encoder = TemporalSensorEncoder(
            hidden_dim=temporal_hidden,
            dropout=dropout,
        )

        self.graph_block1 = GraphAttentionBlock(
            num_nodes=num_sensors,
            input_dim=temporal_hidden,
            output_dim=graph_hidden,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.graph_block2 = GraphAttentionBlock(
            num_nodes=num_sensors,
            input_dim=graph_hidden,
            output_dim=graph_hidden,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(graph_hidden * 2, graph_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(graph_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (batch, sensors, time), got {tuple(x.shape)}"
            )

        x = self.temporal_encoder(x)
        x = self.graph_block1(x)
        x = self.graph_block2(x)

        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        x = torch.cat([mean_pool, max_pool], dim=1)

        return self.classifier(x)


class MEGGAT(MEGGraphAttentionNetwork):
    """Backwards-friendly alias for the graph attention network."""


MEGGAN = MEGGraphAttentionNetwork


@torch.no_grad()
def predict(model: nn.Module, loader, device: str):
    """Run inference and return predicted labels with optional probabilities."""

    model.eval()

    all_predictions = []
    all_probabilities = []

    for batch in loader:
        if len(batch) == 2:
            x, _ = batch
        else:
            x, _, _ = batch

        x = x.to(device)
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        all_predictions.append(predictions.cpu())
        all_probabilities.append(probabilities.cpu())

    return torch.cat(all_predictions), torch.cat(all_probabilities)
