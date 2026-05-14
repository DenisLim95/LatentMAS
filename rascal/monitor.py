"""
KVMonitor — small MLP probe over pooled KV features.

Architecturally the analog of `Representational_Steganography/monitor.py`,
just scaled up: that prototype's input was an 8-d bottleneck vector, ours
is a per-layer pooled KV feature (~9k dim for Qwen3-4B). The hidden width
is widened proportionally; the rest of the structure (single hidden layer
+ activation + dropout + linear head) is unchanged.
"""

from __future__ import annotations

import torch.nn as nn


class KVMonitor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 512,
        num_classes: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)
