from __future__ import annotations

from pathlib import Path

import torch

from cape_det.metrics.unified_evaluator import UnifiedEvaluator
from cape_det.trainers.build import build_dataloader, build_training_components
from cape_det.trainers.checkpoint import load_checkpoint, save_checkpoint
from cape_det.trainers.loops import train_one_epoch, validate
from cape_det.utils.logger import get_logger
from cape_det.utils.profiler import count_parameters
from cape_det.utils.seed import set_seed


class Trainer:
    def __init__(self, config: dict) -> None:
        self.config = config
        set_seed(int(config.get("seed", 1337)))
        self.logger = get_logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.criterion, self.optimizer = build_training_components(config)
        self.model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(config.get("train", {}).get("amp", True)) and self.device.type == "cuda")
        self.output_dir = Path(config.get("output_dir", "outputs"))
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.start_epoch = 0
        resume = config.get("train", {}).get("resume")
        if resume:
            payload = load_checkpoint(resume, self.model, self.optimizer, self.scaler, map_location=self.device)
            self.start_epoch = int(payload.get("epoch", 0)) + 1
        self.logger.info("trainable parameters: %s", count_parameters(self.model))

    def fit(self) -> None:
        train_cfg = self.config.get("train", {})
        train_split = self.config.get("dataset", {}).get("train_split", "train")
        val_split = self.config.get("dataset", {}).get("val_split", "val")
        train_loader = build_dataloader(self.config, split=train_split, shuffle=True)
        val_loader = build_dataloader(self.config, split=val_split, shuffle=False)
        epochs = int(train_cfg.get("epochs", 12))
        best_ap = -1.0
        for epoch in range(self.start_epoch, epochs):
            train_metrics = train_one_epoch(
                self.model,
                train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                epoch,
                scaler=self.scaler,
                amp=bool(train_cfg.get("amp", True)),
                grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
                logger=self.logger,
                log_every=int(train_cfg.get("log_every", 20)),
                limit_batches=train_cfg.get("limit_train_batches"),
            )
            self.logger.info("epoch=%s train=%s", epoch, train_metrics)
            if (epoch + 1) % int(train_cfg.get("validate_every", 1)) == 0:
                evaluator = UnifiedEvaluator(
                    self.config,
                    dataset_name=self.config.get("dataset", {}).get("name", "unknown"),
                    eval_mode=self.config.get("dataset", {}).get("label_mode", "human_unified_single"),
                )
                val_metrics = validate(self.model, val_loader, evaluator, self.device, limit_batches=train_cfg.get("limit_val_batches"))
                self.logger.info("epoch=%s val AP50_95=%.4f AP_tiny=%.4f", epoch, val_metrics["AP50_95"], val_metrics["AP_tiny"])
                save_checkpoint(self.checkpoint_dir / "last.pt", self.model, self.optimizer, self.scaler, epoch, val_metrics, self.config)
                if val_metrics["AP50_95"] >= best_ap:
                    best_ap = val_metrics["AP50_95"]
                    save_checkpoint(self.checkpoint_dir / "best.pt", self.model, self.optimizer, self.scaler, epoch, val_metrics, self.config)
