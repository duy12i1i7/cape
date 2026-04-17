from __future__ import annotations

import torch
from torch import nn


class IterativeHypothesisRefiner(nn.Module):
    def __init__(self, hypothesis_dim: int = 16, latent_dim: int = 64) -> None:
        super().__init__()
        self.delta = nn.Sequential(
            nn.Linear(hypothesis_dim + latent_dim + latent_dim + 1, latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(latent_dim, hypothesis_dim),
        )
        self.state = nn.GRUCell(hypothesis_dim + latent_dim + 1, latent_dim)

    def forward(
        self,
        params: torch.Tensor,
        latent: torch.Tensor,
        evidence_summary: torch.Tensor,
        compatibility: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Updates are gated by active hypotheses; no image regions are reprocessed.
        update_input = torch.cat([params, latent, evidence_summary, compatibility], dim=-1)
        delta = self.delta(update_input).tanh() * 0.15
        mask = active_mask.unsqueeze(-1).to(params.dtype)
        new_params = params + delta * mask
        b, k, _ = params.shape
        state_in = torch.cat([new_params, evidence_summary, compatibility], dim=-1).reshape(b * k, -1)
        latent_flat = latent.reshape(b * k, -1)
        new_latent = self.state(state_in, latent_flat).reshape_as(latent)
        new_latent = latent * (1.0 - mask) + new_latent * mask
        return new_params, new_latent
