import torch
import torch.nn as nn


class CNNBaseline1D(nn.Module):
    def __init__(self, num_classes=4, num_sensors=248, hidden_channels=32, dropout=0.3):
        super().__init__()

        self.network = nn.Sequential(
            # Input shape: (batch, 248 sensors, time)

            nn.Conv1d(
                in_channels=num_sensors,
                out_channels=hidden_channels,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels * 2,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, num_classes),
        )

    def forward(self, x):

        return self.network(x)  