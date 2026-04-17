from __future__ import annotations

import torch

from .cape_losses import cape_hypothesis_loss
from .detection_losses import global_detection_loss


class CompositeDetectionLoss:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.loss_cfg = dict(config.get("loss", {}))
        cape_cfg = config.get("model", {}).get("cape", {})
        if not cape_cfg.get("enable_sparsity_loss", False):
            self.loss_cfg["sparsity"] = 0.0
        if not cape_cfg.get("enable_value_calibration", False):
            self.loss_cfg["value"] = 0.0
        self.num_classes = int(config.get("model", {}).get("num_classes", 1))

    def __call__(self, outputs: dict, targets: list[dict]) -> dict[str, torch.Tensor]:
        losses = global_detection_loss(outputs["global_raw"], targets, self.num_classes, self.loss_cfg)
        total = losses["global_loss"]
        if outputs.get("cape") is not None:
            cape_losses = cape_hypothesis_loss(outputs["cape"], targets, self.loss_cfg)
            losses.update(cape_losses)
            total = total + cape_losses["cape_loss"]
        losses["loss"] = total
        return losses
