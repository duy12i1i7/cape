from __future__ import annotations

import torch
from torch import nn


def _norm(channels: int) -> nn.Module:
    groups = 8
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            _norm(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallCNNBackbone(nn.Module):
    """Small single-GPU-friendly CNN returning C3/C4/C5 feature maps."""

    out_channels = {"c3": 128, "c4": 192, "c5": 256}

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.stem = ConvBlock(3, width, stride=2)          # H/2
        self.stage2 = ConvBlock(width, width * 2, stride=2)  # H/4
        self.stage3 = ConvBlock(width * 2, 128, stride=2)    # H/8
        self.stage4 = ConvBlock(128, 192, stride=2)          # H/16
        self.stage5 = ConvBlock(192, 256, stride=2)          # H/32

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.stage2(x)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return {"c3": c3, "c4": c4, "c5": c5}
