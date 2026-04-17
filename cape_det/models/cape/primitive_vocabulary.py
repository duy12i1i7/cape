from __future__ import annotations

import torch
from torch import nn


class PrimitiveVocabulary(nn.Module):
    """Learned primitive generator used compositionally by each hypothesis."""

    def __init__(self, num_primitives: int = 4, footprint_size: int = 11, code_dim: int = 32) -> None:
        super().__init__()
        self.num_primitives = int(num_primitives)
        self.footprint_size = int(footprint_size)
        self.codes = nn.Embedding(num_primitives, code_dim)
        self.generator = nn.Sequential(
            nn.Linear(code_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, footprint_size * footprint_size),
        )

    def bases(self) -> torch.Tensor:
        ids = torch.arange(self.num_primitives, device=self.codes.weight.device)
        bases = self.generator(self.codes(ids)).view(self.num_primitives, 1, self.footprint_size, self.footprint_size)
        return bases.sigmoid()

    def forward(self, mixture_logits: torch.Tensor) -> torch.Tensor:
        # mixture_logits: [B,K,P] -> footprint [B,K,1,R,R]
        weights = mixture_logits.softmax(dim=-1)
        bases = self.bases()
        return torch.einsum("bkp,pcrs->bkcrs", weights, bases)
