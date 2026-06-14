import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, num_classes=4, num_sensors=248, hidden_size=32, dropout=0.3):
        super().__init__()

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.LazyLinear(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):

        return self.network(x)