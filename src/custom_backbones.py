"""
Lightweight 1D CNN-based backbones for ProtoECGNet or other ECG classifiers.
Adapted from the original EchoNext integration scripts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 71, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class DeepCNN(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 71, dropout: float = 0.2):
        super().__init__()
        layers = []
        filters = [64, 128, 256, 256, 512]
        kernel_sizes = [7, 5, 5, 3, 3]
        current_in = in_channels
        for f, k in zip(filters, kernel_sizes):
            layers.extend([
                nn.Conv1d(current_in, f, k, padding=k // 2),
                nn.BatchNorm1d(f),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ])
            current_in = f
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class WideCNN(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 71, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 256, 7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class CNN_GRU(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 71,
                 hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 128, 7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.gru = nn.GRU(256, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        _, h = self.gru(x)
        h = torch.cat([h[0], h[1]], dim=-1)
        h = self.dropout(h)
        return self.fc(h)


class CNN_LSTM(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 71,
                 hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 128, 7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(256, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[0], h[1]], dim=-1)
        h = self.dropout(h)
        return self.fc(h)


class CNN_Transformer(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 71,
                 dim: int = 256, heads: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, dim, 7, padding=3),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.permute(2, 0, 1)  # (L, B, C)
        x = self.transformer(x)
        x = x.mean(dim=0)
        x = self.dropout(x)
        return self.fc(x)


class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels, 1)
        self.branch3 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.branch5 = nn.Conv1d(in_channels, out_channels, 5, padding=2)
        self.pool = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = F.relu(self.branch1(x))
        b3 = F.relu(self.branch3(x))
        b5 = F.relu(self.branch5(x))
        bp = F.relu(self.pool(F.avg_pool1d(x, 3, stride=1, padding=1)))
        return torch.cat([b1, b3, b5, bp], dim=1)


class InceptionCNN(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 71, dropout: float = 0.2):
        super().__init__()
        self.incept1 = InceptionBlock1D(in_channels, 32)
        self.incept2 = InceptionBlock1D(128, 64)
        self.incept3 = InceptionBlock1D(256, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.incept1(x)
        x = self.incept2(x)
        x = self.incept3(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


def simple_cnn(**kwargs):
    return SimpleCNN(**kwargs)


def deep_cnn(**kwargs):
    return DeepCNN(**kwargs)


def wide_cnn(**kwargs):
    return WideCNN(**kwargs)


def cnn_gru(**kwargs):
    return CNN_GRU(**kwargs)


def cnn_lstm(**kwargs):
    return CNN_LSTM(**kwargs)


def cnn_transformer(**kwargs):
    return CNN_Transformer(**kwargs)


def inception_cnn(**kwargs):
    return InceptionCNN(**kwargs)

