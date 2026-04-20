from __future__ import annotations

from pathlib import Path

from PIL import Image

from .common import Annotation, DatasetRecord, summarize_records, target_from_annotations, xywh_to_xyxy
from .label_mapping import VISDRONE_CATEGORY_NAMES, build_label_mapper
from .transforms import build_transforms


def parse_visdrone_annotation_file(path: str | Path) -> list[Annotation]:
    path = Path(path)
    annotations: list[Annotation] = []
    if not path.exists():
        return annotations
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        x, y, w, h = [float(v) for v in parts[:4]]
        score = int(float(parts[4]))
        category_id = int(float(parts[5]))
        truncation = int(float(parts[6])) if len(parts) > 6 and parts[6] != "" else 0
        occlusion = int(float(parts[7])) if len(parts) > 7 and parts[7] != "" else 0
        category_name = VISDRONE_CATEGORY_NAMES.get(category_id, str(category_id))
        annotations.append(
            Annotation(
                box_xyxy=xywh_to_xyxy((x, y, w, h)),
                category_id=category_id,
                category_name=category_name,
                ignore=category_id == 0 or score == 0,
                raw={"score": score, "truncation": truncation, "occlusion": occlusion},
            )
        )
    return annotations


class VisDroneDataset:
    def __init__(self, config: dict, split: str = "train", transforms=None) -> None:
        self.config = config
        self.split = split
        self.root = Path(config.get("root", "data/visdrone"))
        self.dataset_name = "visdrone"
        self.mapper = build_label_mapper({"dataset": config})
        image_template = config.get("image_dir_template", "images/{split}")
        ann_template = config.get("annotation_dir_template", "annotations/{split}")
        self.image_dir = self.root / image_template.format(split=split)
        self.annotation_dir = self.root / ann_template.format(split=split)
        self.transforms = transforms if transforms is not None else build_transforms({"dataset": config}, train=split == config.get("train_split", "train"))
        self.records = self._load_records()

    def _load_records(self) -> list[DatasetRecord]:
        if not self.image_dir.exists():
            return []
        image_paths = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        )
        records: list[DatasetRecord] = []
        for image_path in image_paths:
            ann_path = self.annotation_dir / f"{image_path.stem}.txt"
            width = height = None
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception:
                pass
            records.append(
                DatasetRecord(
                    image_path=image_path,
                    image_id=image_path.stem,
                    width=width,
                    height=height,
                    annotations=parse_visdrone_annotation_file(ann_path),
                    dataset_name=self.dataset_name,
                )
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        width, height = image.size
        target = target_from_annotations(record.annotations, self.mapper, width, height, record.image_id, self.dataset_name)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def summarize(self) -> dict:
        return summarize_records(self.records)
