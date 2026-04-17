from __future__ import annotations

import torch
from torch import nn

from .budget import HypothesisValueHead, topk_active_mask
from .evidence_encoder import LocalEvidenceEncoder
from .primitive_vocabulary import PrimitiveVocabulary
from .readout import CapeReadout
from .refiner import IterativeHypothesisRefiner
from .renderer import DifferentiableFootprintRenderer
from .seed_generator import HypothesisSeedGenerator
from .types import CapeInternals, CapePredictions


class CapeBranch(nn.Module):
    """Hypothesis-centric CAPE branch.

    This branch never routes image patches, crops, or tiles into a second
    detector. It samples feature evidence around compact human hypotheses and
    allocates refinement budget over hypotheses only.
    """

    def __init__(self, channels: int, num_classes: int, config: dict) -> None:
        super().__init__()
        self.config = config
        self.num_hypotheses = int(config.get("num_hypotheses", 128))
        self.hypothesis_dim = int(config.get("hypothesis_dim", 16))
        self.latent_dim = int(config.get("latent_dim", 64))
        self.num_primitives = int(config.get("num_primitives", 4))
        self.footprint_size = int(config.get("footprint_size", 11))
        self.max_active = int(config.get("max_active_hypotheses", 64))
        self.max_steps = int(config.get("max_refinement_steps", 3))
        self.enable_renderer = bool(config.get("enable_renderer", True))
        self.enable_refinement = bool(config.get("enable_refinement", True))

        self.seed_generator = HypothesisSeedGenerator(channels, self.num_hypotheses, self.hypothesis_dim, self.latent_dim)
        self.primitive_vocab = PrimitiveVocabulary(self.num_primitives, self.footprint_size)
        self.renderer = DifferentiableFootprintRenderer(self.footprint_size)
        self.evidence = LocalEvidenceEncoder(channels, self.latent_dim, self.footprint_size)
        self.value_head = HypothesisValueHead(self.hypothesis_dim, self.latent_dim, enabled=bool(config.get("enable_value_head", True)))
        self.refiner = IterativeHypothesisRefiner(self.hypothesis_dim, self.latent_dim)
        self.readout = CapeReadout(
            self.hypothesis_dim,
            self.latent_dim,
            num_classes,
            min_box_frac=float(config.get("min_box_frac", 0.004)),
            max_box_frac=float(config.get("max_box_frac", 0.20)),
        )

    def _render(self, params: torch.Tensor) -> torch.Tensor | None:
        if not self.enable_renderer:
            return None
        mixture_logits = params[..., 6 : 6 + self.num_primitives]
        footprint = self.primitive_vocab(mixture_logits)
        return self.renderer(footprint, params[..., 10])

    def forward(self, features: dict[str, torch.Tensor], image_sizes: list[tuple[int, int]]) -> CapePredictions:
        seed_params, latent, _ = self.seed_generator(features["p3"])
        params = seed_params
        active_masks = []
        seed_readout = self.readout(seed_params, latent, image_sizes)
        seed_footprints = self._render(params)
        _, seed_evidence_map, evidence_summary, seed_compatibility = self.evidence(features["p3"], params, seed_footprints)
        seed_value_logits = self.value_head(params, latent, seed_compatibility)
        footprints = seed_footprints
        evidence_map = seed_evidence_map
        compatibility = seed_compatibility
        value_logits = seed_value_logits

        steps = self.max_steps if self.enable_refinement else 0
        for _ in range(steps):
            active = topk_active_mask(value_logits, self.max_active)
            active_masks.append(active)
            params, latent = self.refiner(params, latent, evidence_summary, compatibility, active)
            footprints = self._render(params)
            _, evidence_map, evidence_summary, compatibility = self.evidence(features["p3"], params, footprints)
            value_logits = self.value_head(params, latent, compatibility)

        final_readout = self.readout(params, latent, image_sizes)
        if active_masks:
            mask_tensor = torch.stack(active_masks, dim=1)
            avg_active = mask_tensor.any(dim=1).to(params.dtype).sum(dim=1).mean()
            avg_budget = mask_tensor.to(params.dtype).sum(dim=(1, 2)).mean()
        else:
            mask_tensor = None
            avg_active = params.new_tensor(0.0)
            avg_budget = params.new_tensor(0.0)
        internals = CapeInternals(
            seed_params=seed_params,
            final_params=params,
            latent=latent,
            seed_readout=seed_readout,
            final_readout=final_readout,
            seed_footprints=seed_footprints,
            final_footprints=footprints,
            seed_evidence_maps=seed_evidence_map,
            final_evidence_maps=evidence_map,
            seed_compatibility=seed_compatibility,
            final_compatibility=compatibility,
            seed_value_logits=seed_value_logits,
            final_value_logits=value_logits,
            footprints=footprints,
            evidence_maps=evidence_map,
            compatibility=compatibility,
            value_logits=seed_value_logits,
            active_masks=mask_tensor,
            avg_active_hypotheses=avg_active,
            avg_refinement_budget_used=avg_budget,
        )
        return CapePredictions(
            boxes=final_readout.boxes,
            scores=final_readout.scores,
            labels=final_readout.labels,
            class_logits=final_readout.class_logits,
            internals=internals,
        )
