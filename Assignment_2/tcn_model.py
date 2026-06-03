import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)

        return out

class MEGTCN(nn.Module):
    def __init__(self, num_classes, hidden_channels=64, kernel_size=7, dropout=0.2):
        super().__init__()

        # Input shape:
        # (batch, 248 sensors, time)

        self.block1 = TCNBlock(
            in_channels=248,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation=1,
            dropout=dropout,
        )

        self.pool1 = nn.MaxPool1d(2)

        self.block2 = TCNBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels * 2,
            kernel_size=kernel_size-2,
            dilation=2,
            dropout=dropout,
        )

        self.pool2 = nn.MaxPool1d(2)

        self.block3 = TCNBlock(
            in_channels=hidden_channels * 2,
            out_channels=hidden_channels * 2,
            kernel_size=kernel_size-4,
            dilation=4,
            dropout=dropout,
        )

        self.pool3 = nn.MaxPool1d(2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.pool3(x)

        x = self.global_pool(x)

        x = self.classifier(x)

        return x