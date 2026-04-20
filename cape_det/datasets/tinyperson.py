from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .common import Annotation, DatasetRecord, summarize_records, target_from_annotations, xywh_to_xyxy
from .transforms import build_transforms
from .label_mapping import build_label_mapper


def load_tinyperson_records(root: Path, image_dir: Path, annotation_file: Path) -> list[DatasetRecord]:
    if not annotation_file.exists():
        return []
    with annotation_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    categories = {cat.get("id"): str(cat.get("name", cat.get("id"))) for cat in data.get("categories", [])}
    images = {img["id"]: img for img in data.get("images", [])}
    anns_by_image: dict[int | str, list[Annotation]] = {image_id: [] for image_id in images}
    for ann in data.get("annotations", []):
        image_id = ann.get("image_id")
        category_id = ann.get("category_id", "person")
        category_name = categories.get(category_id, str(category_id))
        ignore = bool(ann.get("ignore", 0) or ann.get("iscrowd", 0))
        if "bbox" not in ann:
            continue
        anns_by_image.setdefault(image_id, []).append(
            Annotation(
                box_xyxy=xywh_to_xyxy(ann["bbox"]),
                category_id=category_id,
                category_name=category_name,
                ignore=ignore,
                raw=ann,
            )
        )
    records: list[DatasetRecord] = []
    for image_id, img in images.items():
        file_name = img.get("file_name", f"{image_id}.jpg")
        image_path = Path(file_name)
        if not image_path.is_absolute():
            image_path = image_dir / image_path
        records.append(
            DatasetRecord(
                image_path=image_path,
                image_id=str(image_id),
                width=img.get("width"),
                height=img.get("height"),
                annotations=anns_by_image.get(image_id, []),
                dataset_name="tinyperson",
            )
        )
    return records


class TinyPersonDataset:
    def __init__(self, config: dict, split: str = "train", transforms=None) -> None:
        self.config = config
        self.split = split
        self.root = Path(config.get("root", "data/tinyperson"))
        self.dataset_name = "tinyperson"
        self.mapper = build_label_mapper({"dataset": config})
        image_template = config.get("image_dir_template", "images/{split}")
        ann_template = config.get("annotation_file_template", "annotations/{split}.json")
        self.image_dir = self.root / image_template.format(split=split)
        self.annotation_file = self.root / ann_template.format(split=split)
        self.transforms = transforms if transforms is not None else build_transforms({"dataset": config}, train=split == config.get("train_split", "train"))
        self.records = load_tinyperson_records(self.root, self.image_dir, self.annotation_file)

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
