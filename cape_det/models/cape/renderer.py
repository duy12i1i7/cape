from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DifferentiableFootprintRenderer(nn.Module):
    """Apply degradation-aware smoothing to learned compositional footprints."""

    def __init__(self, footprint_size: int = 11) -> None:
        super().__init__()
        self.footprint_size = int(footprint_size)
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        kernel = kernel / kernel.sum()
        self.register_buffer("blur_kernel", kernel.view(1, 1, 3, 3))

    def forward(self, footprint: torch.Tensor, blur_latent: torch.Tensor) -> torch.Tensor:
        # footprint: [B,K,1,R,R], blur_latent: [B,K] or [B,K,1]
        b, k, c, r, _ = footprint.shape
        flat = footprint.reshape(b * k, c, r, r)
        blurred = F.conv2d(flat, self.blur_kernel, padding=1)
        alpha = blur_latent.reshape(b * k, 1, 1, 1).sigmoid()
        degraded = flat * (1.0 - alpha) + blurred * alpha
        return degraded.reshape(b, k, c, r, r).clamp(0.0, 1.0)
