from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LocalEvidenceEncoder(nn.Module):
    """Sample feature evidence around hypothesis centers without crop-based re-detection."""

    def __init__(self, channels: int = 128, latent_dim: int = 64, footprint_size: int = 11) -> None:
        super().__init__()
        self.footprint_size = int(footprint_size)
        self.evidence_proj = nn.Conv2d(channels, 1, 1)
        self.summary = nn.Sequential(
            nn.Linear(channels + 2, latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

    def _make_grid(self, params: torch.Tensor) -> torch.Tensor:
        b, k, _ = params.shape
        r = self.footprint_size
        center = params[..., 0:2].sigmoid()
        wh = params[..., 2:4].sigmoid().clamp(min=0.003, max=0.3)
        offsets = torch.linspace(-1.0, 1.0, r, device=params.device, dtype=params.dtype)
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        base = torch.stack([xx, yy], dim=-1).view(1, 1, r, r, 2)
        grid01 = center.view(b, k, 1, 1, 2) + base * wh.view(b, k, 1, 1, 2) * 0.5
        return grid01.mul(2.0).sub(1.0)

    def forward(
        self, feature: torch.Tensor, params: torch.Tensor, footprint: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # feature [B,C,H,W], params [B,K,D] -> sampled [B,K,C,R,R]
        b, c, _, _ = feature.shape
        k = params.shape[1]
        r = self.footprint_size
        grid = self._make_grid(params).view(b, k * r, r, 2)
        sampled = F.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.view(b, c, k, r, r).permute(0, 2, 1, 3, 4).contiguous()
        flat = sampled.view(b * k, c, r, r)
        evidence_map = self.evidence_proj(flat).view(b, k, 1, r, r)
        pooled = sampled.mean(dim=(-1, -2))
        if footprint is None:
            compatibility = evidence_map.sigmoid().mean(dim=(-1, -2, -3), keepdim=False).unsqueeze(-1)
            residual_scalar = compatibility
        else:
            residual = evidence_map.sigmoid() - footprint
            residual_scalar = residual.abs().mean(dim=(-1, -2, -3), keepdim=False).unsqueeze(-1)
            compatibility = 1.0 - residual_scalar.clamp(0.0, 1.0)
        summary_input = torch.cat([pooled, compatibility, residual_scalar], dim=-1)
        summary = self.summary(summary_input)
        return sampled, evidence_map, summary, compatibility
