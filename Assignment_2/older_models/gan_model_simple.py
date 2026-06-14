import torch
import torch.nn as nn


class TemporalSensorEncoder(nn.Module):
    """Encode each sensor's time series into a compact node embedding.

    Input: (batch, sensors, time)
    Output: (batch, sensors, embed_dim)
    """

    def __init__(self, embed_dim: int = 32, dropout: float = 0.1):
        super().__init__()

        # reduce time series to embedding per sensor
        self.encoder = nn.Sequential(
            nn.Conv1d(1, embed_dim // 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_sensors, time_steps = x.shape

        # reshape using batch and sensor dimensions 
        x = x.view(batch_size * num_sensors, 1, time_steps)
        x = self.encoder(x)  # (batch*sensors, embed_dim, 1)
        x = x.view(batch_size, num_sensors, -1)  # (batch, sensors, embed_dim)
        return x


class GraphAttentionBlock(nn.Module):
    """
    This block computes attention between sensors.
    """

    def __init__(self, num_nodes: int, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()

        # initialize linear layers for query, key, value projections
        self.query = nn.Linear(input_dim, output_dim, bias=False)
        self.key = nn.Linear(input_dim, output_dim, bias=False)
        self.value = nn.Linear(input_dim, output_dim, bias=False)

        # learnable bias for node-to-node logits (acts like a prior adjacency).
        self.graph_bias = nn.Parameter(torch.zeros(num_nodes, num_nodes))

        self.out_proj = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # residuals for if input and output dimensions differ
        self.residual_proj = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_nodes, input_dim)
        batch_size, num_nodes, _ = x.shape

        residual = self.residual_proj(x)

        # Compute Q, K, V in node space.
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Attention logits between nodes: (batch, num_nodes, num_nodes)
        logits = torch.matmul(q, k.transpose(-1, -2))
        logits = logits / (q.size(-1) ** 0.5)

        # Add the learnable graph prior
        logits = logits + self.graph_bias.unsqueeze(0)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        # Aggregate values from other nodes: (batch, num_nodes, output_dim)
        out = torch.matmul(attn, v)

        out = self.out_proj(out)
        out = self.dropout(out)
        out = self.norm(out + residual)
        out = self.activation(out)

        return out


class MEGGraphAttentionNetwork(nn.Module):
    """Compact graph-attention network producing class logits.

    Pipeline:
    - Temporal encoder converts each sensor's timeseries -> embedding
    - Two graph-attention blocks compute interactions between sensors
    - Mean+max pooling over sensors, then a small MLP classifier
    """

    def __init__(
        self,
        num_classes: int,
        num_sensors: int = 248,
        temporal_hidden: int = 32,
        graph_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.temporal_encoder = TemporalSensorEncoder(embed_dim=temporal_hidden, dropout=dropout)

        self.graph_block1 = GraphAttentionBlock(num_nodes=num_sensors, input_dim=temporal_hidden, output_dim=graph_hidden, dropout=dropout)
        self.graph_block2 = GraphAttentionBlock(num_nodes=num_sensors, input_dim=graph_hidden, output_dim=graph_hidden, dropout=dropout)

        # classifier input is double graph_hidden due to mean+max pooling concatenation
        self.classifier = nn.Sequential(
            nn.Linear(graph_hidden * 2, graph_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(graph_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect (batch, sensors, time)
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, sensors, time), got {tuple(x.shape)}")

        # Encode each sensor's time-series into node embeddings
        x = self.temporal_encoder(x)

        # Two rounds of message passing / attention between nodes
        x = self.graph_block1(x)
        x = self.graph_block2(x)

        # Pool over sensors using both mean and max pooling, then concatenate
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        x = torch.cat([mean_pool, max_pool], dim=1)

        # Final classification logits
        return self.classifier(x)


MEGGAN = MEGGraphAttentionNetwork


@torch.no_grad()
def predict(model: nn.Module, loader, device: str):
    """Run inference and return predicted labels and probabilities.

    Returns:
        (predictions, probabilities) as two tensors concatenated over batches.
    """

    model.eval()

    all_predictions = []
    all_probabilities = []

    for batch in loader:
        # Support different loader return conventions (x, y) or (x, y, meta)
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
