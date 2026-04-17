from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HypothesisReadout:
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor
    class_logits: torch.Tensor
    conf_logits: torch.Tensor
    conf_scores: torch.Tensor
    centers_norm: torch.Tensor
    sizes_norm: torch.Tensor
    boxes_norm: torch.Tensor


@dataclass
class CapeInternals:
    seed_params: torch.Tensor
    final_params: torch.Tensor
    latent: torch.Tensor
    seed_readout: HypothesisReadout
    final_readout: HypothesisReadout
    seed_footprints: torch.Tensor | None
    final_footprints: torch.Tensor | None
    seed_evidence_maps: torch.Tensor | None
    final_evidence_maps: torch.Tensor | None
    seed_compatibility: torch.Tensor | None
    final_compatibility: torch.Tensor | None
    seed_value_logits: torch.Tensor | None
    final_value_logits: torch.Tensor | None
    footprints: torch.Tensor | None
    evidence_maps: torch.Tensor | None
    compatibility: torch.Tensor | None
    value_logits: torch.Tensor | None
    active_masks: torch.Tensor | None
    avg_active_hypotheses: torch.Tensor
    avg_refinement_budget_used: torch.Tensor


@dataclass
class CapePredictions:
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor
    class_logits: torch.Tensor
    internals: CapeInternals
