"""Compact causal temporal convolution used for the MVP experiment."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel_size[0] - 1) * self.dilation[0]
        return super().forward(F.pad(inputs, (padding, 0)))


class RelayTemporalNet(nn.Module):
    def __init__(self, feature_count: int, class_count: int, hidden: int = 16):
        super().__init__()
        self.conv1 = CausalConv1d(feature_count, hidden, kernel_size=3)
        self.conv2 = CausalConv1d(hidden, hidden, kernel_size=3)
        self.head = nn.Linear(hidden * 2, class_count)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Input is [batch, time, feature].
        hidden = inputs.transpose(1, 2)
        hidden = F.relu(self.conv1(hidden))
        hidden = F.relu(self.conv2(hidden))
        pooled = torch.cat([hidden.mean(dim=2), hidden[:, :, -1]], dim=1)
        return self.head(pooled)
