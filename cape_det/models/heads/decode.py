from __future__ import annotations

import torch


DEFAULT_STRIDES = {"p3": 8, "p4": 16, "p5": 32}


def _grid_centers(height: int, width: int, stride: int, device) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack(((xs + 0.5) * stride, (ys + 0.5) * stride), dim=-1).reshape(-1, 2)


def decode_anchor_free_outputs(
    outputs: dict[str, dict[str, torch.Tensor]],
    image_sizes: list[tuple[int, int]],
    score_threshold: float = 0.05,
    max_candidates: int = 1000,
    strides: dict[str, int] | None = None,
) -> list[dict[str, torch.Tensor]]:
    strides = strides or DEFAULT_STRIDES
    batch_size = next(iter(outputs.values()))["class_logits"].shape[0]
    decoded: list[dict[str, torch.Tensor]] = []
    for b in range(batch_size):
        boxes_all, scores_all, labels_all = [], [], []
        for level, out in outputs.items():
            cls = out["class_logits"][b].sigmoid()
            obj = out.get("objectness")
            if obj is not None:
                cls = cls * obj[b].sigmoid()
            reg = out["box_reg"][b]
            _, h, w = cls.shape
            stride = strides[level]
            centers = _grid_centers(h, w, stride, cls.device)
            scores, labels = cls.permute(1, 2, 0).reshape(-1, cls.shape[0]).max(dim=1)
            keep = scores > score_threshold
            if keep.sum() == 0:
                continue
            scores = scores[keep]
            labels = labels[keep]
            centers_kept = centers[keep]
            ltrb = reg.permute(1, 2, 0).reshape(-1, 4)[keep] * stride
            boxes = torch.stack(
                [
                    centers_kept[:, 0] - ltrb[:, 0],
                    centers_kept[:, 1] - ltrb[:, 1],
                    centers_kept[:, 0] + ltrb[:, 2],
                    centers_kept[:, 1] + ltrb[:, 3],
                ],
                dim=1,
            )
            height_i, width_i = image_sizes[b]
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width_i)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height_i)
            if scores.numel() > max_candidates:
                topk = scores.topk(max_candidates).indices
                boxes, scores, labels = boxes[topk], scores[topk], labels[topk]
            boxes_all.append(boxes)
            scores_all.append(scores)
            labels_all.append(labels)
        if boxes_all:
            decoded.append({"boxes": torch.cat(boxes_all), "scores": torch.cat(scores_all), "labels": torch.cat(labels_all)})
        else:
            device = next(iter(outputs.values()))["class_logits"].device
            decoded.append(
                {
                    "boxes": torch.empty((0, 4), device=device),
                    "scores": torch.empty((0,), device=device),
                    "labels": torch.empty((0,), dtype=torch.long, device=device),
                }
            )
    return decoded
