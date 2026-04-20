from __future__ import annotations

from typing import Iterable

import torch


def targets_to_device(targets: list[dict], device: torch.device) -> list[dict]:
    moved = []
    for target in targets:
        item = {}
        for key, value in target.items():
            item[key] = value.to(device) if hasattr(value, "to") else value
        moved.append(item)
    return moved


def train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler=None,
    amp: bool = True,
    grad_clip_norm: float | None = 1.0,
    logger=None,
    log_every: int = 20,
    limit_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    num_steps = 0
    for step, (images, targets) in enumerate(loader):
        if limit_batches is not None and step >= int(limit_batches):
            break
        images = images.to(device, non_blocking=True)
        targets = targets_to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            outputs = model(images, targets)
            losses = criterion(outputs, targets)
            loss = losses["loss"]
        if scaler is not None and amp and device.type == "cuda":
            scaler.scale(loss).backward()
            if grad_clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        num_steps += 1
        if logger is not None and (step + 1) % log_every == 0:
            logger.info("epoch=%s step=%s loss=%.4f", epoch, step + 1, totals["loss"] / max(num_steps, 1))
    return {key: value / max(num_steps, 1) for key, value in totals.items()}


@torch.no_grad()
def validate(model: torch.nn.Module, loader: Iterable, evaluator, device: torch.device, limit_batches: int | None = None) -> dict:
    model.eval()
    active = []
    budget = []
    for step, (images, targets) in enumerate(loader):
        if limit_batches is not None and step >= int(limit_batches):
            break
        images = images.to(device, non_blocking=True)
        targets_device = targets_to_device(targets, device)
        outputs = model(images, targets_device)
        predictions = []
        for pred in outputs["predictions"]:
            predictions.append({k: v.detach().cpu() for k, v in pred.items()})
        evaluator.update(predictions, targets)
        if outputs.get("cape") is not None:
            active.append(float(outputs["cape"].internals.avg_active_hypotheses.detach().cpu()))
            budget.append(float(outputs["cape"].internals.avg_refinement_budget_used.detach().cpu()))
    avg_active = sum(active) / max(len(active), 1) if active else 0.0
    avg_budget = sum(budget) / max(len(budget), 1) if budget else 0.0
    return evaluator.compute(avg_active_hypotheses=avg_active, avg_refinement_budget_used=avg_budget)
