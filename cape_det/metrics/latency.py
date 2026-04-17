from __future__ import annotations

import time
from typing import Any


def benchmark_latency(model: Any, example_batch, warmup_iters: int = 10, timed_iters: int = 50) -> dict[str, float]:
    import torch

    model.eval()
    device = next(model.parameters()).device
    images = example_batch.to(device)
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(timed_iters):
            _ = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    latency_ms = elapsed * 1000.0 / max(timed_iters, 1)
    return {"Latency_ms": latency_ms, "FPS": 1000.0 / latency_ms if latency_ms > 0 else float("nan")}
