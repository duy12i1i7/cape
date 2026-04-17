from __future__ import annotations

from typing import Any


def count_parameters(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def estimate_flops(_: Any, __: Any = None) -> float:
    try:
        return float("nan")
    except Exception:
        return float("nan")
