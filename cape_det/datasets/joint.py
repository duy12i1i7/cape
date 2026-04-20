from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Any

from cape_det.utils.config import load_config

from .common import summarize_records
from .label_mapping import build_label_mapper
from .tinyperson import TinyPersonDataset
from .visdrone import VisDroneDataset


class JointHumanDataset:
    """Concatenate human-focused datasets while preserving each adapter's parser.

    This is a clean extension point for future joint training. It keeps VisDrone
    and TinyPerson parsing and label mapping inside their adapters instead of
    moving dataset-specific rules into the model.
    """

    def __init__(self, config: dict[str, Any], split: str = "train", transforms=None) -> None:
        self.config = config
        self.split = split
        self.dataset_name = config.get("name", "joint_humans_visdrone_tinyperson")
        self.mapper = build_label_mapper({"dataset": config})
        self.datasets = []
        for entry in config.get("datasets", []):
            dataset_cfg = self._load_dataset_config(entry)
            dataset_cfg["label_mode"] = config.get("label_mode", dataset_cfg.get("label_mode", "human_unified_single"))
            if "visdrone_people_policy" in config:
                dataset_cfg["visdrone_people_policy"] = config["visdrone_people_policy"]
            name = dataset_cfg.get("name", "").lower()
            if name == "visdrone":
                self.datasets.append(VisDroneDataset(dataset_cfg, split=split, transforms=transforms))
            elif name == "tinyperson":
                self.datasets.append(TinyPersonDataset(dataset_cfg, split=split, transforms=transforms))
            else:
                raise ValueError(f"Unsupported joint dataset member: {dataset_cfg.get('name')}")
        self.cumulative_sizes: list[int] = []
        running = 0
        for dataset in self.datasets:
            running += len(dataset)
            self.cumulative_sizes.append(running)

    def _load_dataset_config(self, entry: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(entry, dict):
            return dict(entry.get("dataset", entry))
        cfg = load_config(Path(entry))
        return dict(cfg.get("dataset", cfg))

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, index: int):
        if index < 0:
            index = len(self) + index
        dataset_idx = bisect_right(self.cumulative_sizes, index)
        previous = self.cumulative_sizes[dataset_idx - 1] if dataset_idx > 0 else 0
        return self.datasets[dataset_idx][index - previous]

    def summarize(self) -> dict[str, Any]:
        summaries = [dataset.summarize() for dataset in self.datasets]
        records = []
        for dataset in self.datasets:
            records.extend(dataset.records)
        combined = summarize_records(records)
        combined["members"] = summaries
        return combined
