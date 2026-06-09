import torch
import torch.nn as nn


class TemporalSensorEncoder(nn.Module):
    def __init__(self, embed_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_ch = 1
        for dilation in (1, 2, 4):
            layers += [
                nn.Conv1d(in_ch, embed_dim, kernel_size=3, padding=dilation, dilation=dilation),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = embed_dim

        layers.append(nn.AdaptiveAvgPool1d(1))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, t = x.shape
        x = x.view(b * n, 1, t)
        x = self.encoder(x)
        x = x.view(b, n, -1)
        return x


class GraphAttentionBlock(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.25,
        attention_dropout: float = 0.2,
        num_neighbors: int = 8,
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

        self.graph_bias = nn.Parameter(torch.zeros(1, 1, num_nodes, num_nodes))

        mask = torch.full((1, 1, num_nodes, num_nodes), float("-inf"))
        for i in range(num_nodes):
            low = max(0, i - num_neighbors)
            high = min(num_nodes, i + num_neighbors + 1)
            mask[0, 0, i, low:high] = 0.0

        self.register_buffer("attention_mask", mask)

        self.out_proj = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.activation = nn.GELU()

        self.residual_proj = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        residual = self.residual_proj(x)

        q = self.query(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        logits = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        mask = self.attention_mask
        mask = mask.to(logits.dtype).to(logits.device)
        logits = logits + self.graph_bias + mask

        attn = torch.softmax(logits, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, -1)

        out = self.out_proj(out)
        out = self.dropout(out)
        out = self.norm(out + residual)
        out = self.activation(out)

        return out


class MEGGraphAttentionNetwork(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_sensors: int = 248,
        temporal_hidden: int = 32,
        graph_hidden: int = 64,
        num_heads: int = 4,
        dropout: float = 0.25,
        attention_dropout: float = 0.2,
        num_neighbors: int = 8,
    ):
        super().__init__()

        self.temporal_encoder = TemporalSensorEncoder(embed_dim=temporal_hidden, dropout=dropout)
        self.node_pos_emb = nn.Embedding(num_sensors, temporal_hidden)

        self.graph_block1 = GraphAttentionBlock(
            num_nodes=num_sensors,
            input_dim=temporal_hidden,
            output_dim=graph_hidden,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_neighbors=num_neighbors,
        )

        self.graph_block2 = GraphAttentionBlock(
            num_nodes=num_sensors,
            input_dim=graph_hidden,
            output_dim=graph_hidden,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_neighbors=num_neighbors,
        )

        self.classifier = nn.Sequential(
            nn.Linear(graph_hidden * 2, graph_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(graph_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape (batch, sensors, time), got {tuple(x.shape)}")

        b, n, _ = x.shape

        x = self.temporal_encoder(x)

        pos = self.node_pos_emb.weight.unsqueeze(0)
        x = x + pos

        x = self.graph_block1(x)
        x = self.graph_block2(x)

        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        x = torch.cat([mean_pool, max_pool], dim=1)

        return self.classifier(x)


MEGGAN = MEGGraphAttentionNetwork


@torch.no_grad()
def predict(model: nn.Module, loader, device: str):
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
