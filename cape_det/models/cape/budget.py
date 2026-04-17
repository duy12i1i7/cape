from __future__ import annotations

import torch
from torch import nn


class HypothesisValueHead(nn.Module):
    def __init__(self, hypothesis_dim: int = 16, latent_dim: int = 64, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.net = nn.Sequential(
            nn.Linear(hypothesis_dim + latent_dim + 1, latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, params: torch.Tensor, latent: torch.Tensor, compatibility: torch.Tensor | None = None) -> torch.Tensor:
        if compatibility is None:
            compatibility = params.new_zeros((*params.shape[:2], 1))
        if not self.enabled:
            return params[..., 11]
        return self.net(torch.cat([params, latent, compatibility], dim=-1)).squeeze(-1)


def topk_active_mask(value_logits: torch.Tensor, max_active: int) -> torch.Tensor:
    b, k = value_logits.shape
    active = min(int(max_active), k)
    if active <= 0:
        return torch.zeros_like(value_logits, dtype=torch.bool)
    inds = value_logits.topk(active, dim=1).indices
    mask = torch.zeros_like(value_logits, dtype=torch.bool)
    return mask.scatter(1, inds, True)
