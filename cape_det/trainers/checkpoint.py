from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scaler=None,
    epoch: int = 0,
    metrics: dict[str, Any] | None = None,
    config: dict | None = None,
) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model, optimizer=None, scaler=None, map_location="cpu") -> dict:
    import torch

    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"], strict=False)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    return payload
