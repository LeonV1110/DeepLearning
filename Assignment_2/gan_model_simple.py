import torch
import torch.nn as nn


class SensorEncoder(nn.Module):
    """Encode each sensor's time series into a compact node embedding."""

    def __init__(self, hidden_dim: int = 16, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()

        mid_dim = max(hidden_dim // 2, 8)
        second_kernel = max(3, kernel_size - 2)

        self.encoder = nn.Sequential(
            nn.Conv1d(1, mid_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(mid_dim, hidden_dim, kernel_size=second_kernel, padding=second_kernel // 2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_sensors, time_steps = x.shape
        x = x.reshape(batch_size * num_sensors, 1, time_steps)
        x = self.encoder(x)
        return x.flatten(1).reshape(batch_size, num_sensors, -1)


class SensorAttentionBlock(nn.Module):

    def __init__(self, input_dim: int, num_heads: int = 1, dropout: float = 0.1):
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
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, feature_dim = x.shape
        residual = x

        query = self.query(x).reshape(batch_size, num_nodes, self.num_heads, self.head_dim)
        key = self.key(x).reshape(batch_size, num_nodes, self.num_heads, self.head_dim)
        value = self.value(x).reshape(batch_size, num_nodes, self.num_heads, self.head_dim)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attention_logits = torch.matmul(query, key.transpose(-1, -2))
        attention_logits = attention_logits / (self.head_dim ** 0.5)
        attention = torch.softmax(attention_logits, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, value)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, feature_dim)
        out = self.out_proj(out)
        out = self.dropout(out)
        out = self.norm(out + residual)
        return self.activation(out)


class MEGGAN(nn.Module):

    def __init__(
        self,
        num_classes: int = 4,
        num_sensors: int = 248,
        hidden_channels: int = 32,
        kernel_size: int = 7,
        dropout: float = 0.3,
        graph_hidden: int | None = None,
        num_heads: int = 1,
    ):
        super().__init__()

        graph_hidden = graph_hidden or hidden_channels

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

        self.sensor_encoder = SensorEncoder(
            hidden_dim=graph_hidden,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.sensor_attention = SensorAttentionBlock(
            input_dim=graph_hidden,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.graph_head = nn.Sequential(
            nn.Linear(graph_hidden * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )
        nn.init.zeros_(self.graph_head[-1].weight)
        nn.init.zeros_(self.graph_head[-1].bias)
        self.graph_gate = nn.Parameter(torch.tensor(-3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, sensors, time), got {tuple(x.shape)}")

        backbone_features = self.backbone(x)
        backbone_logits = self.backbone_classifier(self.backbone_pool(backbone_features))

        node_features = self.sensor_encoder(x)
        node_features = self.sensor_attention(node_features)
        graph_mean = node_features.mean(dim=1)
        graph_max = node_features.max(dim=1).values
        graph_features = torch.cat([graph_mean, graph_max], dim=1)
        graph_logits = self.graph_head(graph_features)

        return backbone_logits + torch.sigmoid(self.graph_gate) * graph_logits
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Returns attention weights (B, H, N, N) — one matrix per head."""
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, sensors, time), got {tuple(x.shape)}")

        node_features = self.sensor_encoder(x)
        batch_size, num_nodes, feature_dim = node_features.shape

        query = self.sensor_attention.query(node_features).reshape(
            batch_size, num_nodes, self.sensor_attention.num_heads, self.sensor_attention.head_dim
        )
        key = self.sensor_attention.key(node_features).reshape(
            batch_size, num_nodes, self.sensor_attention.num_heads, self.sensor_attention.head_dim
        )

        query = query.permute(0, 2, 1, 3)  # (B, H, N, head_dim)
        key   = key.permute(0, 2, 1, 3)

        attention_logits = torch.matmul(query, key.transpose(-1, -2))  # (B, H, N, N)
        attention_logits = attention_logits / (self.sensor_attention.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_logits, dim=-1)

        return attention_weights  # (B, H, N, N) — caller decides how to aggregate heads