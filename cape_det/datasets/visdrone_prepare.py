from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .formats import iter_images, safe_link_or_copy, split_summary, write_json, write_prepared_config
from .validators import validate_prepared_dataset


LogFn = Callable[[str], None]


VISDRONE_SPLIT_DIRS = {
    "train": ("VisDrone2019-DET-train", "train"),
    "val": ("VisDrone2019-DET-val", "val"),
    "test": ("VisDrone2019-DET-test-dev", "VisDrone2019-DET-test-challenge", "test"),
}


def _log(message: str, logger: LogFn | None = None) -> None:
    if logger is not None:
        logger(message)
    else:
        print(message)


def _find_split_dirs(raw_root: Path, split: str) -> tuple[Path, Path] | None:
    candidates = VISDRONE_SPLIT_DIRS.get(split, (split,))
    for candidate in candidates:
        base = raw_root / candidate
        image_dir = base / "images"
        ann_dir = base / "annotations"
        if image_dir.exists() and ann_dir.exists():
            return image_dir, ann_dir

    image_dir = raw_root / "images" / split
    ann_dir = raw_root / "annotations" / split
    if image_dir.exists() and ann_dir.exists():
        return image_dir, ann_dir

    image_dir = raw_root / split / "images"
    ann_dir = raw_root / split / "annotations"
    if image_dir.exists() and ann_dir.exists():
        return image_dir, ann_dir
    return None


def _prepare_split(raw_root: Path, prepared_root: Path, split: str, link_mode: str, logger: LogFn | None = None) -> dict[str, Any]:
    split_dirs = _find_split_dirs(raw_root, split)
    if split_dirs is None:
        if split == "test":
            return split_summary(split, 0, 0)
        raise FileNotFoundError(
            f"Could not locate VisDrone raw split '{split}' under {raw_root}. "
            "Expected VisDrone2019-DET-{train,val,test-dev}/images + annotations."
        )
    image_src, ann_src = split_dirs
    image_dst = prepared_root / "images" / split
    label_dst = prepared_root / "labels" / split
    image_dst.mkdir(parents=True, exist_ok=True)
    label_dst.mkdir(parents=True, exist_ok=True)

    images = iter_images(image_src)
    annotation_count = 0
    ignored = 0
    invalid = 0
    for image_path in images:
        safe_link_or_copy(image_path, image_dst / image_path.name, mode=link_mode)
        src_ann = ann_src / f"{image_path.stem}.txt"
        dst_ann = label_dst / f"{image_path.stem}.txt"
        if src_ann.exists():
            if not dst_ann.exists():
                dst_ann.write_text(src_ann.read_text(encoding="utf-8"), encoding="utf-8")
            lines = [line for line in src_ann.read_text(encoding="utf-8").splitlines() if line.strip()]
            annotation_count += len(lines)
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 6:
                    invalid += 1
                    continue
                try:
                    ignored += int(int(float(parts[4])) == 0 or int(float(parts[5])) == 0)
                    invalid += int(float(parts[2]) < 1 or float(parts[3]) < 1)
                except ValueError:
                    invalid += 1
        elif not dst_ann.exists():
            dst_ann.write_text("", encoding="utf-8")
    summary = split_summary(split, len(images), annotation_count, ignored=ignored, invalid=invalid)
    _log(f"[dataset] prepared VisDrone/{split}: {summary}", logger)
    return summary


def prepare_visdrone(
    raw_root: str | Path,
    prepared_root: str | Path,
    label_mode: str = "human_unified_single",
    splits: list[str] | tuple[str, ...] = ("train", "val", "test"),
    link_mode: str = "symlink",
    logger: LogFn | None = None,
) -> dict[str, Any]:
    raw_root = Path(raw_root)
    prepared_root = Path(prepared_root)
    metadata_dir = prepared_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for split in splits:
        summaries[split] = _prepare_split(raw_root, prepared_root, split, link_mode, logger=logger)

    config_path = metadata_dir / "visdrone_prepared.yaml"
    prepared_config = write_prepared_config("visdrone", prepared_root, config_path, label_mode=label_mode)
    validation = validate_prepared_dataset("visdrone", prepared_root, splits=tuple(s for s in splits if s != "test"))
    payload = {
        "dataset": "visdrone",
        "raw_root": str(raw_root),
        "prepared_root": str(prepared_root),
        "config_path": str(config_path),
        "splits": summaries,
        "validation": validation,
    }
    write_json(metadata_dir / "split_summaries.json", payload)
    return {"config_path": config_path, "config": prepared_config, "summary": payload}
