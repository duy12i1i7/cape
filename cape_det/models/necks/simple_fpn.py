from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SimpleFPN(nn.Module):
    def __init__(self, in_channels: dict[str, int], out_channels: int = 128) -> None:
        super().__init__()
        self.lateral3 = nn.Conv2d(in_channels["c3"], out_channels, 1)
        self.lateral4 = nn.Conv2d(in_channels["c4"], out_channels, 1)
        self.lateral5 = nn.Conv2d(in_channels["c5"], out_channels, 1)
        self.out3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        c3, c4, c5 = features["c3"], features["c4"], features["c5"]
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return {"p3": self.out3(p3), "p4": self.out4(p4), "p5": self.out5(p5)}
