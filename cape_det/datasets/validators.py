from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .formats import IMAGE_SUFFIXES, iter_images


class DatasetValidationError(RuntimeError):
    pass


def _count_visdrone_annotations(label_dir: Path, image_dir: Path) -> tuple[int, int, int]:
    annotations = 0
    ignored = 0
    invalid = 0
    if not label_dir.exists():
        return annotations, ignored, invalid
    image_stems = {path.stem for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
    label_stems = {path.stem for path in label_dir.glob("*.txt")}
    invalid += len(image_stems - label_stems)
    for path in label_dir.glob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                invalid += 1
                continue
            try:
                w = float(parts[2])
                h = float(parts[3])
                score = int(float(parts[4]))
                category_id = int(float(parts[5]))
            except ValueError:
                invalid += 1
                continue
            annotations += 1
            ignored += int(category_id == 0 or score == 0)
            invalid += int(w < 1 or h < 1)
    return annotations, ignored, invalid


def _count_coco_annotations(annotation_file: Path, image_dir: Path) -> tuple[int, int, int, int]:
    if not annotation_file.exists():
        return 0, 0, 0, 0
    with annotation_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images", [])
    missing_images = 0
    for image in images:
        path = image_dir / str(image.get("file_name", ""))
        missing_images += int(not path.exists())
    annotations = 0
    ignored = 0
    invalid = 0
    for ann in data.get("annotations", []):
        bbox = ann.get("bbox")
        if not bbox or len(bbox) < 4:
            invalid += 1
            continue
        annotations += 1
        ignored += int(bool(ann.get("ignore", 0) or ann.get("iscrowd", 0)))
        invalid += int(float(bbox[2]) < 1 or float(bbox[3]) < 1)
    return len(images), annotations, ignored, invalid + missing_images


def validate_prepared_dataset(
    dataset_name: str,
    root: str | Path,
    splits: list[str] | tuple[str, ...] = ("train", "val"),
    allow_empty_test: bool = True,
) -> dict[str, Any]:
    dataset_name = dataset_name.lower()
    root = Path(root)
    if not root.exists():
        raise DatasetValidationError(f"Prepared root does not exist: {root}")

    split_summaries: dict[str, dict[str, int]] = {}
    for split in splits:
        image_dir = root / "images" / split
        images = iter_images(image_dir)
        if not images and not (allow_empty_test and split == "test"):
            raise DatasetValidationError(f"No prepared images for {dataset_name}/{split}: {image_dir}")
        if dataset_name == "visdrone":
            label_dir = root / "labels" / split
            if not label_dir.exists() and not (allow_empty_test and split == "test"):
                raise DatasetValidationError(f"Missing VisDrone label dir for {split}: {label_dir}")
            annotations, ignored, invalid = _count_visdrone_annotations(label_dir, image_dir)
            split_summaries[split] = {
                "images": len(images),
                "annotations": annotations,
                "ignored_annotations": ignored,
                "invalid_boxes": invalid,
            }
        elif dataset_name == "tinyperson":
            ann_file = root / "labels" / f"{split}.json"
            if not ann_file.exists() and not (allow_empty_test and split == "test"):
                raise DatasetValidationError(f"Missing TinyPerson annotation file for {split}: {ann_file}")
            image_count, annotations, ignored, invalid = _count_coco_annotations(ann_file, image_dir)
            if image_count and image_count != len(images):
                invalid += abs(image_count - len(images))
            split_summaries[split] = {
                "images": len(images),
                "annotations": annotations,
                "ignored_annotations": ignored,
                "invalid_boxes": invalid,
            }
        else:
            raise DatasetValidationError(f"Unsupported prepared dataset: {dataset_name}")

    return {"dataset": dataset_name, "root": str(root), "splits": split_summaries}


def prepared_dataset_is_valid(dataset_name: str, root: str | Path, splits: list[str] | tuple[str, ...] = ("train", "val")) -> bool:
    try:
        validate_prepared_dataset(dataset_name, root, splits=splits)
        return True
    except DatasetValidationError:
        return False
