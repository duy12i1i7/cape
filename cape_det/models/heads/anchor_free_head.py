from __future__ import annotations

import torch
from torch import nn


class HeadTower(nn.Module):
    def __init__(self, channels: int, depth: int = 2) -> None:
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(16, channels),
                nn.SiLU(inplace=True),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AnchorFreeHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, use_objectness: bool = True) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.use_objectness = bool(use_objectness)
        self.cls_tower = HeadTower(channels)
        self.box_tower = HeadTower(channels)
        self.cls_logits = nn.Conv2d(channels, num_classes, 3, padding=1)
        self.box_reg = nn.Conv2d(channels, 4, 3, padding=1)
        self.objectness = nn.Conv2d(channels, 1, 3, padding=1) if use_objectness else None
        nn.init.constant_(self.cls_logits.bias, -4.6)
        if self.objectness is not None:
            nn.init.constant_(self.objectness.bias, -4.6)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        for name, feature in features.items():
            cls_feat = self.cls_tower(feature)
            box_feat = self.box_tower(feature)
            outputs[name] = {
                "class_logits": self.cls_logits(cls_feat),
                "box_reg": torch.nn.functional.softplus(self.box_reg(box_feat)),
            }
            if self.objectness is not None:
                outputs[name]["objectness"] = self.objectness(box_feat)
        return outputs
