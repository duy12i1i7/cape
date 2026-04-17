from __future__ import annotations

import torch
from torch import nn

from .types import HypothesisReadout


class CapeReadout(nn.Module):
    def __init__(
        self,
        hypothesis_dim: int = 16,
        latent_dim: int = 64,
        num_classes: int = 1,
        min_box_frac: float = 0.004,
        max_box_frac: float = 0.20,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.min_box_frac = float(min_box_frac)
        self.max_box_frac = float(max_box_frac)
        self.class_head = nn.Sequential(
            nn.Linear(hypothesis_dim + latent_dim, latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, params: torch.Tensor, latent: torch.Tensor, image_sizes: list[tuple[int, int]]) -> HypothesisReadout:
        center = params[..., 0:2].sigmoid()
        wh_frac = self.min_box_frac + params[..., 2:4].sigmoid() * (self.max_box_frac - self.min_box_frac)
        conf_logits = params[..., 11]
        conf = conf_logits.sigmoid()
        class_logits = self.class_head(torch.cat([params, latent], dim=-1))
        class_scores, labels = class_logits.sigmoid().max(dim=-1)
        scores = conf * class_scores
        boxes = []
        boxes_norm = []
        for idx, (height, width) in enumerate(image_sizes):
            scale = params.new_tensor([width, height, width, height])
            cxcy = center[idx]
            wh = wh_frac[idx]
            xyxy_norm = torch.cat([cxcy - wh * 0.5, cxcy + wh * 0.5], dim=-1).clamp(0.0, 1.0)
            boxes_norm.append(xyxy_norm)
            boxes.append(xyxy_norm * scale)
        return HypothesisReadout(
            boxes=torch.stack(boxes, dim=0),
            scores=scores,
            labels=labels,
            class_logits=class_logits,
            conf_logits=conf_logits,
            conf_scores=conf,
            centers_norm=center,
            sizes_norm=wh_frac,
            boxes_norm=torch.stack(boxes_norm, dim=0),
        )
