from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from cape_det.utils.config import save_config


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def safe_link_or_copy(src: str | Path, dst: str | Path, mode: str = "symlink") -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_prepared_config(
    dataset_name: str,
    root: str | Path,
    config_path: str | Path,
    label_mode: str = "human_unified_single",
    extra_dataset_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    dataset_cfg: dict[str, Any] = {
        "name": dataset_name,
        "root": str(root),
        "prepared": True,
        "auto_prepare": True,
        "train_split": "train",
        "val_split": "val",
        "test_split": "test",
        "image_dir_template": "images/{split}",
        "label_mode": label_mode,
        "class_names": ["person"] if label_mode == "human_unified_single" else None,
    }
    if dataset_name == "visdrone":
        dataset_cfg.update(
            {
                "annotation_dir_template": "labels/{split}",
                "visdrone_people_policy": "merge",
                "visdrone_ignore_category_ids": [0],
            }
        )
        if label_mode == "human_split":
            dataset_cfg["class_names"] = ["pedestrian", "people"]
    elif dataset_name == "tinyperson":
        dataset_cfg.update(
            {
                "annotation_file_template": "labels/{split}.json",
                "tinyperson_person_names": ["person", "people", "pedestrian", "sea_person", "earth_person"],
                "tinyperson_person_ids": [1, 2, "person"],
            }
        )
        if label_mode == "human_split":
            dataset_cfg["class_names"] = ["person"]
    if dataset_cfg.get("class_names") is None:
        dataset_cfg.pop("class_names")
    if extra_dataset_fields:
        dataset_cfg.update(extra_dataset_fields)
    config = {"dataset": dataset_cfg}
    save_config(config, config_path)
    return config


def split_summary(split: str, images: int, annotations: int, ignored: int = 0, invalid: int = 0) -> dict[str, Any]:
    return {
        "split": split,
        "images": int(images),
        "annotations": int(annotations),
        "ignored_annotations": int(ignored),
        "invalid_boxes": int(invalid),
    }
