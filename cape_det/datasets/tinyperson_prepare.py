from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from .formats import iter_images, read_json, safe_link_or_copy, split_summary, write_json, write_prepared_config
from .validators import validate_prepared_dataset


LogFn = Callable[[str], None]


def _log(message: str, logger: LogFn | None = None) -> None:
    if logger is not None:
        logger(message)
    else:
        print(message)


def _candidate_annotation_files(raw_root: Path, split: str) -> list[Path]:
    names = [
        f"{split}.json",
        f"tiny_set_{split}.json",
        f"tiny_set_{split}_all.json",
        f"tiny_set_{split}_all_erase.json",
        f"tiny_set_{split}_sw640_sh512_all.json",
        f"tiny_set_{split}_sw640_sh512_all_erase.json",
    ]
    candidates: list[Path] = []
    for name in names:
        candidates.extend(
            [
                raw_root / "annotations" / name,
                raw_root / "labels" / name,
                raw_root / name,
            ]
        )
    return candidates


def _candidate_image_dirs(raw_root: Path, split: str) -> list[Path]:
    return [
        raw_root / "images" / split,
        raw_root / "Images" / split,
        raw_root / split,
        raw_root / "images",
        raw_root / "Images",
        raw_root / "JPEGImages",
    ]


def _resolve_manual_path(config: dict[str, Any] | None, split: str, key: str) -> Path | None:
    manual = (config or {}).get("manual_paths", {})
    value = manual.get(split, {}).get(key)
    return Path(value).expanduser() if value else None


def _find_annotation_file(raw_root: Path, split: str, config: dict[str, Any] | None) -> Path | None:
    manual = _resolve_manual_path(config, split, "annotation_file")
    if manual is not None and manual.exists():
        return manual
    for candidate in _candidate_annotation_files(raw_root, split):
        if candidate.exists():
            return candidate
    return None


def _find_image_dir(raw_root: Path, split: str, config: dict[str, Any] | None) -> Path | None:
    manual = _resolve_manual_path(config, split, "image_dir")
    if manual is not None and manual.exists():
        return manual
    for candidate in _candidate_image_dirs(raw_root, split):
        if candidate.exists():
            return candidate
    return None


def _find_image_path(file_name: str, image_dir: Path, raw_root: Path) -> Path | None:
    raw_name = Path(file_name)
    candidates = []
    if raw_name.is_absolute():
        candidates.append(raw_name)
    else:
        candidates.extend(
            [
                image_dir / raw_name,
                image_dir / raw_name.name,
                raw_root / raw_name,
                raw_root / "images" / raw_name,
                raw_root / "Images" / raw_name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(image_dir.rglob(raw_name.name))
    return matches[0] if matches else None


def _prepare_split(
    raw_root: Path,
    prepared_root: Path,
    split: str,
    link_mode: str,
    config: dict[str, Any] | None,
    logger: LogFn | None = None,
) -> dict[str, Any]:
    ann_file = _find_annotation_file(raw_root, split, config)
    image_dir = _find_image_dir(raw_root, split, config)
    if ann_file is None or image_dir is None:
        if split == "test":
            return split_summary(split, 0, 0)
        raise FileNotFoundError(
            f"Could not locate TinyPerson split '{split}' under {raw_root}. "
            "Provide COCO-style images/json via TINYPERSON_RAW_ROOT or scripts/prepare_datasets.py manual flags."
        )

    data = read_json(ann_file)
    prepared_data = deepcopy(data)
    image_dst = prepared_root / "images" / split
    label_dst = prepared_root / "labels"
    image_dst.mkdir(parents=True, exist_ok=True)
    label_dst.mkdir(parents=True, exist_ok=True)

    linked_images = 0
    missing_images = 0
    for image in prepared_data.get("images", []):
        file_name = str(image.get("file_name", ""))
        src = _find_image_path(file_name, image_dir, raw_root)
        if src is None:
            missing_images += 1
            continue
        dst_name = src.name
        safe_link_or_copy(src, image_dst / dst_name, mode=link_mode)
        image["file_name"] = dst_name
        linked_images += 1

    annotations = 0
    ignored = 0
    invalid = missing_images
    for ann in prepared_data.get("annotations", []):
        bbox = ann.get("bbox")
        if not bbox or len(bbox) < 4:
            invalid += 1
            continue
        annotations += 1
        ignored += int(bool(ann.get("ignore", 0) or ann.get("iscrowd", 0)))
        invalid += int(float(bbox[2]) < 1 or float(bbox[3]) < 1)

    out_file = label_dst / f"{split}.json"
    write_json(out_file, prepared_data)
    summary = split_summary(split, linked_images, annotations, ignored=ignored, invalid=invalid)
    _log(f"[dataset] prepared TinyPerson/{split}: {summary}", logger)
    return summary


def prepare_tinyperson(
    raw_root: str | Path,
    prepared_root: str | Path,
    label_mode: str = "human_unified_single",
    splits: list[str] | tuple[str, ...] = ("train", "val", "test"),
    link_mode: str = "symlink",
    config: dict[str, Any] | None = None,
    logger: LogFn | None = None,
) -> dict[str, Any]:
    raw_root = Path(raw_root)
    prepared_root = Path(prepared_root)
    metadata_dir = prepared_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for split in splits:
        summaries[split] = _prepare_split(raw_root, prepared_root, split, link_mode, config, logger=logger)

    config_path = metadata_dir / "tinyperson_prepared.yaml"
    prepared_config = write_prepared_config("tinyperson", prepared_root, config_path, label_mode=label_mode)
    validation = validate_prepared_dataset("tinyperson", prepared_root, splits=tuple(s for s in splits if s != "test"))
    payload = {
        "dataset": "tinyperson",
        "raw_root": str(raw_root),
        "prepared_root": str(prepared_root),
        "config_path": str(config_path),
        "splits": summaries,
        "validation": validation,
    }
    write_json(metadata_dir / "split_summaries.json", payload)
    return {"config_path": config_path, "config": prepared_config, "summary": payload}
