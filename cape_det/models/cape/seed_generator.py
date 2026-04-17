from __future__ import annotations

import torch
from torch import nn


class HypothesisSeedGenerator(nn.Module):
    """Generate fixed-size human hypotheses from feature maps, not image patches."""

    def __init__(self, channels: int, num_hypotheses: int = 128, hypothesis_dim: int = 16, latent_dim: int = 64) -> None:
        super().__init__()
        self.num_hypotheses = int(num_hypotheses)
        self.hypothesis_dim = int(hypothesis_dim)
        self.latent_dim = int(latent_dim)
        self.shared = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(16, channels),
            nn.SiLU(inplace=True),
        )
        self.heatmap = nn.Conv2d(channels, 1, 1)
        self.param_head = nn.Conv2d(channels, hypothesis_dim, 1)
        self.latent_head = nn.Conv2d(channels, latent_dim, 1)
        nn.init.constant_(self.heatmap.bias, -2.0)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # feature: [B,C,H,W] -> params [B,K,D], latent [B,K,L], seed scores [B,K]
        b, _, h, w = feature.shape
        x = self.shared(feature)
        heat = self.heatmap(x).flatten(1).sigmoid()
        k = min(self.num_hypotheses, heat.shape[1])
        scores, indices = heat.topk(k, dim=1)
        params_map = self.param_head(x).flatten(2).transpose(1, 2)
        latent_map = self.latent_head(x).flatten(2).transpose(1, 2)
        gather_param = indices.unsqueeze(-1).expand(-1, -1, self.hypothesis_dim)
        gather_latent = indices.unsqueeze(-1).expand(-1, -1, self.latent_dim)
        params = params_map.gather(1, gather_param)
        latent = latent_map.gather(1, gather_latent)

        ys = torch.div(indices, w, rounding_mode="floor").to(feature.dtype)
        xs = (indices % w).to(feature.dtype)
        centers = torch.stack(((xs + 0.5) / max(w, 1), (ys + 0.5) / max(h, 1)), dim=-1)
        params = params.clone()
        params[..., 0:2] = torch.logit(centers.clamp(1e-4, 1 - 1e-4))
        params[..., 2:4] = params[..., 2:4].clamp(-2.0, 2.0)
        params[..., 11] = torch.logit(scores.clamp(1e-4, 1 - 1e-4))
        return params, latent, scores
