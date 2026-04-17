from __future__ import annotations

from functools import partial

import torch
from torch.utils.data import DataLoader

from cape_det.datasets import build_dataset, validate_label_mapper_num_classes
from cape_det.datasets.collate import detection_collate
from cape_det.losses.total_loss import CompositeDetectionLoss
from cape_det.models import build_model


def build_dataloader(config: dict, split: str, shuffle: bool) -> DataLoader:
    dataset = build_dataset(config, split=split)
    dataset_cfg = config.get("dataset", {})
    batch_size = int(config.get("train", {}).get("batch_size", 2))
    num_workers = int(dataset_cfg.get("num_workers", 2))
    pad_divisor = int(dataset_cfg.get("pad_divisor", 32))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(detection_collate, pad_divisor=pad_divisor),
        pin_memory=torch.cuda.is_available(),
    )


def build_optimizer(config: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    train_cfg = config.get("train", {})
    lr = float(train_cfg.get("lr", 2e-4))
    backbone_lr_mult = float(train_cfg.get("backbone_lr_mult", 0.25))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone"):
            backbone_params.append(param)
        else:
            other_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * backbone_lr_mult},
            {"params": other_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )


def build_training_components(config: dict):
    validate_label_mapper_num_classes(config)
    model = build_model(config)
    criterion = CompositeDetectionLoss(config)
    optimizer = build_optimizer(config, model)
    return model, criterion, optimizer
