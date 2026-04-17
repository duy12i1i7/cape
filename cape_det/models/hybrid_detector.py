from __future__ import annotations

import torch
from torch import nn

from .backbones.small_cnn import SmallCNNBackbone
from .cape.cape_branch import CapeBranch
from .heads.anchor_free_head import AnchorFreeHead
from .heads.decode import decode_anchor_free_outputs
from .necks.simple_fpn import SimpleFPN
from .postprocess import merge_prediction_lists, postprocess_predictions


class HybridDetector(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config.get("model", config)
        self.config = model_cfg
        self.num_classes = int(model_cfg.get("num_classes", 1))
        channels = int(model_cfg.get("fpn_channels", 128))
        self.score_threshold = float(model_cfg.get("score_threshold", 0.05))
        self.nms_threshold = float(model_cfg.get("nms_threshold", 0.5))
        self.max_detections = int(model_cfg.get("max_detections", 300))
        self.decode_during_train = bool(model_cfg.get("decode_during_train", False))
        self.backbone = SmallCNNBackbone()
        self.neck = SimpleFPN(self.backbone.out_channels, channels)
        self.head = AnchorFreeHead(channels, self.num_classes, use_objectness=bool(model_cfg.get("use_objectness", True)))
        cape_cfg = model_cfg.get("cape", {})
        self.cape_enabled = bool(cape_cfg.get("enabled", model_cfg.get("mode") == "cape"))
        self.cape = CapeBranch(channels, self.num_classes, cape_cfg) if self.cape_enabled else None

    def _image_sizes(self, images: torch.Tensor, targets: list[dict] | None = None) -> list[tuple[int, int]]:
        if targets is not None:
            return [tuple(map(int, t.get("size", t.get("orig_size", images.shape[-2:])))) for t in targets]
        return [(int(images.shape[-2]), int(images.shape[-1])) for _ in range(images.shape[0])]

    def _empty_predictions(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "boxes": images.new_empty((0, 4)),
                "scores": images.new_empty((0,)),
                "labels": torch.empty((0,), dtype=torch.long, device=images.device),
            }
            for _ in range(images.shape[0])
        ]

    def forward(self, images: torch.Tensor, targets: list[dict] | None = None) -> dict:
        image_sizes = self._image_sizes(images, targets)
        features = self.neck(self.backbone(images))
        global_raw = self.head(features)
        cape_out = None
        if self.cape is not None:
            cape_out = self.cape(features, image_sizes)

        should_decode = (not self.training) or self.decode_during_train
        if should_decode:
            global_preds = decode_anchor_free_outputs(global_raw, image_sizes, self.score_threshold)
            raw_predictions = global_preds
            if cape_out is not None:
                cape_preds = []
                for b in range(images.shape[0]):
                    cape_preds.append(
                        {
                            "boxes": cape_out.boxes[b],
                            "scores": cape_out.scores[b],
                            "labels": cape_out.labels[b],
                        }
                    )
                raw_predictions = merge_prediction_lists(global_preds, cape_preds)
            predictions = postprocess_predictions(raw_predictions, self.score_threshold, self.nms_threshold, self.max_detections)
        else:
            predictions = self._empty_predictions(images)
        return {
            "predictions": predictions,
            "predictions_decoded": should_decode,
            "global_raw": global_raw,
            "features": features,
            "cape": cape_out,
            "image_sizes": image_sizes,
        }


def build_model(config: dict) -> HybridDetector:
    return HybridDetector(config)
