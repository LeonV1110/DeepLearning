import torch
import torch.nn as nn
import torch.nn.functional as F


class SensorEncoder(nn.Module):
    """Encode each sensor's time series into a compact node embedding."""

    def __init__(self, hidden_dim: int = 16, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()

        mid_dim = max(hidden_dim // 2, 8)
        second_kernel = max(3, kernel_size - 2)

        self.encoder = nn.Sequential(
            nn.Conv1d(1, mid_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(mid_dim, hidden_dim, kernel_size=second_kernel, padding=second_kernel // 2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_sensors, time_steps = x.shape
        x = x.reshape(batch_size * num_sensors, 1, time_steps)
        x = self.encoder(x)
        return x.flatten(1).reshape(batch_size, num_sensors, -1)


class SensorAttentionBlock(nn.Module):
    """
    Multi-head self-attention with an optional spatial distance prior.

    If sensor_pos is provided (num_sensors, 3), a learnable scalar `dist_scale`
    biases attention so that nearby sensors attend to each other more strongly
    at the start of training. The model can override this prior as it learns.

    Attention weights are cached in `self.last_attn` after each forward pass
    for downstream visualisation (shape: batch, heads, sensors, sensors).
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        sensor_pos: torch.Tensor | None = None,
    ):
        super().__init__()

        if input_dim % num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.query = nn.Linear(input_dim, input_dim, bias=False)
        self.key = nn.Linear(input_dim, input_dim, bias=False)
        self.value = nn.Linear(input_dim, input_dim, bias=False)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # Spatial distance prior
        # dist_scale > 0 penalises distant sensors; the model learns how
        # strongly to enforce the prior. Initialised small so the prior is
        # a gentle nudge rather than a hard constraint.
        if sensor_pos is not None:
            dist = torch.cdist(sensor_pos.float(), sensor_pos.float())  # (N, N)
            self.register_buffer("sensor_dist", dist)
            self.dist_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.sensor_dist = None
            self.dist_scale = None

        # Cached for visualisation — not used in the forward pass
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, feature_dim = x.shape
        residual = x

        query = self.query(x).reshape(batch_size, num_nodes, self.num_heads, self.head_dim)
        key   = self.key(x)  .reshape(batch_size, num_nodes, self.num_heads, self.head_dim)
        value = self.value(x).reshape(batch_size, num_nodes, self.num_heads, self.head_dim)

        query = query.permute(0, 2, 1, 3)  # (B, H, N, D)
        key   = key  .permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attention_logits = torch.matmul(query, key.transpose(-1, -2))
        attention_logits = attention_logits / (self.head_dim ** 0.5)

        # Add spatial prior: penalise large pairwise distances
        if self.sensor_dist is not None:
            spatial_bias = -F.softplus(self.dist_scale) * self.sensor_dist
            attention_logits = attention_logits + spatial_bias.unsqueeze(0).unsqueeze(0)

        attention = torch.softmax(attention_logits, dim=-1)

        # Cache before dropout so visualisations show the clean weights
        self.last_attn = attention.detach()

        attention = self.dropout(attention)

        out = torch.matmul(attention, value)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, feature_dim)
        out = self.out_proj(out)
        out = self.dropout(out)
        out = self.norm(out + residual)
        return self.activation(out)


class GNNLayer(nn.Module):
    """
    One round of learned message passing.

    Computes a soft adjacency from scaled dot-product similarities between
    node features, then aggregates neighbour messages. No fixed graph needed —
    the adjacency is fully data-driven and updated every forward pass.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.scale = hidden_dim ** -0.5
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        adj = torch.bmm(x, x.transpose(1, 2)) * self.scale  # (B, N, N)
        adj = torch.softmax(adj, dim=-1)
        messages = torch.bmm(adj, x)                         # (B, N, D)
        return self.norm(x + self.dropout(messages))


class AttentionReadout(nn.Module):
    """
    Attention-weighted graph readout.

    Rather than mean/max pooling, learns which sensors matter most for
    classification. Produces a single vector per graph.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        weights = torch.softmax(self.score(x), dim=1)  # (B, N, 1)
        return (weights * x).sum(dim=1)                # (B, D)


class MHEGAN(nn.Module):
    """
    Multi-head Encoder with Graph-based Graph Attention Network.

    Architecture:
      - CNN backbone over all sensors for a strong global baseline
      - Per-sensor CNN encoder → node embeddings
      - Sensor ID embeddings (give each sensor a learnable identity)
      - Self-attention with optional spatial distance prior
      - One GNN message-passing layer (learned soft adjacency)
      - Attention-weighted readout → graph head
      - Gated fusion: backbone + sigmoid(gate) * graph branch

    Args:
        num_classes:   Number of output classes.
        num_sensors:   Number of MEG sensors (default 248).
        hidden_channels: Backbone CNN width.
        kernel_size:   Temporal kernel size for CNN layers.
        dropout:       Dropout probability throughout.
        graph_hidden:  Node embedding dimension (defaults to hidden_channels).
        num_heads:     Number of attention heads in SensorAttentionBlock.
        sensor_pos:    Optional (num_sensors, 3) tensor of sensor xyz positions.
                       When provided, attention is biased toward nearby sensors.
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
        sensor_pos: torch.Tensor | None = None,
    ):
        super().__init__()

        graph_hidden = graph_hidden or hidden_channels
        self.num_sensors = num_sensors

        # ── CNN backbone ──────────────────────────────────────────────────────
        self.backbone = nn.Sequential(
            nn.Conv1d(
                in_channels=num_sensors,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels * 2,
                kernel_size=max(3, kernel_size - 2),
                padding=max(3, kernel_size - 2) // 2,
            ),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.backbone_pool = nn.AdaptiveAvgPool1d(1)
        self.backbone_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, num_classes),
        )

        # ── Graph branch ──────────────────────────────────────────────────────

        # Per-sensor temporal encoder
        self.sensor_encoder = SensorEncoder(
            hidden_dim=graph_hidden,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Sensor identity embeddings — help the model associate specific
        # spatial locations with classes; important when data is limited.
        self.sensor_id_embedding = nn.Embedding(num_sensors, graph_hidden)
        nn.init.normal_(self.sensor_id_embedding.weight, mean=0.0, std=0.02)

        # Self-attention (with optional spatial prior)
        self.sensor_attention = SensorAttentionBlock(
            input_dim=graph_hidden,
            num_heads=num_heads,
            dropout=dropout,
            sensor_pos=sensor_pos,
        )

        # One round of GNN message passing (learned soft adjacency)
        self.gnn_layer = GNNLayer(hidden_dim=graph_hidden, dropout=dropout)

        # Attention-weighted readout instead of mean/max pooling
        self.readout = AttentionReadout(hidden_dim=graph_hidden)

        # Classification head
        self.graph_head = nn.Sequential(
            nn.Linear(graph_hidden, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )
        #nn.init.zeros_(self.graph_head[-1].weight)
        #nn.init.zeros_(self.graph_head[-1].bias)

        # Learned gate — initialised near zero so the graph branch only
        # activates once it has learned something useful.
        self.graph_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, sensors, time), got {tuple(x.shape)}")

        # ── Backbone ──────────────────────────────────────────────────────────
        backbone_features = self.backbone(x)
        backbone_logits = self.backbone_classifier(self.backbone_pool(backbone_features))

        # ── Graph branch ──────────────────────────────────────────────────────
        node_features = self.sensor_encoder(x)  # (B, N, D)

        # Add learnable sensor identities
        sensor_ids = torch.arange(self.num_sensors, device=x.device)
        node_features = node_features + self.sensor_id_embedding(sensor_ids).unsqueeze(0)

        # Self-attention (attention weights cached in sensor_attention.last_attn)
        node_features = self.sensor_attention(node_features)

        # GNN message passing (one round of learned neighbourhood aggregation)
        node_features = self.gnn_layer(node_features)

        # Attention-weighted readout → single vector per sample
        graph_summary = self.readout(node_features)  # (B, D)

        graph_logits = self.graph_head(graph_summary)

        # ── Gated fusion ─────────────────────────────────────────────────────
        return backbone_logits + torch.sigmoid(self.graph_gate) * graph_logits

    def get_attention_weights(self) -> torch.Tensor | None:
        """
        Return the attention weight matrix from the last forward pass.

        Shape: (batch, num_heads, num_sensors, num_sensors)
        Useful for visualising which sensor pairs the model attends to.

        Example usage:
            model.eval()
            with torch.no_grad():
                _ = model(x_batch)
            attn = model.get_attention_weights()  # (B, H, N, N)
            # Average over batch and heads for a single 248x248 map:
            attn_map = attn.mean(dim=(0, 1)).numpy()
        """
        return self.sensor_attention.last_attn