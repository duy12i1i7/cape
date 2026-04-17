from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


def draw_boxes(
    image: Image.Image | np.ndarray,
    boxes: Iterable[Iterable[float]],
    labels: Iterable[str] | None = None,
    scores: Iterable[float] | None = None,
    color: str = "red",
) -> Image.Image:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image).astype("uint8"))
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    labels = list(labels or [])
    scores = list(scores or [])
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        parts = []
        if idx < len(labels):
            parts.append(labels[idx])
        if idx < len(scores):
            parts.append(f"{scores[idx]:.2f}")
        if parts:
            draw.text((x1, max(0.0, y1 - 12)), " ".join(parts), fill=color)
    return out


def save_boxes(path: str | Path, image: Image.Image | np.ndarray, boxes, labels=None, scores=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    draw_boxes(image, boxes, labels, scores).save(path)
