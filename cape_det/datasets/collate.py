from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def pad_to_divisor(image: torch.Tensor, divisor: int) -> torch.Tensor:
    _, h, w = image.shape
    pad_h = int(math.ceil(h / divisor) * divisor - h)
    pad_w = int(math.ceil(w / divisor) * divisor - w)
    return F.pad(image, (0, pad_w, 0, pad_h))


def detection_collate(batch: list[tuple[torch.Tensor, dict]], pad_divisor: int = 32):
    images, targets = zip(*batch)
    padded = [pad_to_divisor(img, pad_divisor) for img in images]
    max_h = max(img.shape[1] for img in padded)
    max_w = max(img.shape[2] for img in padded)
    batch_images = []
    for img in padded:
        batch_images.append(F.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1])))
    return torch.stack(batch_images, dim=0), list(targets)
