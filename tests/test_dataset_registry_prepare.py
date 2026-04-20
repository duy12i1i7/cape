import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from cape_det.datasets.prepare import ensure_prepared_dataset
from cape_det.datasets.registry import dataset_prepared_root, dataset_raw_root, get_dataset_spec
from cape_det.datasets.validators import validate_prepared_dataset


def _write_visdrone_raw(root: Path) -> None:
    for split, dirname in [("train", "VisDrone2019-DET-train"), ("val", "VisDrone2019-DET-val")]:
        image_dir = root / dirname / "images"
        ann_dir = root / dirname / "annotations"
        image_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)
        Image.new("RGB", (32, 24)).save(image_dir / f"{split}.jpg")
        ann_dir.joinpath(f"{split}.txt").write_text("1,2,5,6,1,1,0,0\n10,10,8,8,0,0,0,0\n", encoding="utf-8")


def _write_tinyperson_raw(root: Path) -> None:
    for split in ["train", "val"]:
        image_dir = root / "images" / split
        ann_dir = root / "annotations"
        image_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24)).save(image_dir / f"{split}.jpg")
        data = {
            "images": [{"id": split, "file_name": f"{split}.jpg", "width": 32, "height": 24}],
            "categories": [{"id": 1, "name": "person"}],
            "annotations": [{"id": 1, "image_id": split, "category_id": 1, "bbox": [1, 2, 5, 6]}],
        }
        ann_dir.joinpath(f"{split}.json").write_text(json.dumps(data), encoding="utf-8")


def test_registry_env_roots(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CAPE_DATA_ROOT", str(tmp_path / "cape_data"))
    monkeypatch.setenv("VISDRONE_RAW_ROOT", str(tmp_path / "vis_raw"))
    spec = get_dataset_spec("visdrone")
    assert dataset_raw_root(spec) == tmp_path / "vis_raw"
    assert dataset_prepared_root(spec) == tmp_path / "cape_data" / "prepared" / "visdrone"


def test_visdrone_prepare_generates_config_and_reuses_cache(tmp_path: Path):
    raw = tmp_path / "raw_vis"
    prepared = tmp_path / "prepared_vis"
    _write_visdrone_raw(raw)
    result = ensure_prepared_dataset("visdrone", raw_root=raw, prepared_root=prepared, allow_download=False)
    assert Path(result["config_path"]).exists()
    summary = validate_prepared_dataset("visdrone", prepared, splits=("train", "val"))
    assert summary["splits"]["train"]["images"] == 1
    assert summary["splits"]["train"]["ignored_annotations"] == 1

    cached = ensure_prepared_dataset("visdrone", raw_root=tmp_path / "missing", prepared_root=prepared, allow_download=False)
    assert cached["prepared_root"] == prepared


def test_tinyperson_prepare_from_coco_raw(tmp_path: Path):
    raw = tmp_path / "raw_tiny"
    prepared = tmp_path / "prepared_tiny"
    _write_tinyperson_raw(raw)
    result = ensure_prepared_dataset("tinyperson", raw_root=raw, prepared_root=prepared, allow_download=False)
    assert Path(result["config_path"]).exists()
    summary = validate_prepared_dataset("tinyperson", prepared, splits=("train", "val"))
    assert summary["splits"]["val"]["images"] == 1
    assert (prepared / "labels" / "train.json").exists()


def test_download_dispatch_is_mockable(monkeypatch, tmp_path: Path):
    prepared = tmp_path / "prepared_vis"

    def fake_download(urls, raw_root, retries=3, logger=None):
        _write_visdrone_raw(Path(raw_root))
        return [Path(raw_root)]

    monkeypatch.setattr("cape_det.datasets.prepare.download_and_extract", fake_download)
    config = {"dataset": {"download_urls": ["https://example.invalid/fake.zip"]}}
    result = ensure_prepared_dataset("visdrone", config=config, raw_root=tmp_path / "raw_vis", prepared_root=prepared)
    assert Path(result["config_path"]).exists()


def test_existing_empty_raw_root_falls_back_to_download(monkeypatch, tmp_path: Path):
    raw = tmp_path / "raw_vis"
    raw.mkdir()
    prepared = tmp_path / "prepared_vis"

    def fake_download(urls, raw_root, retries=3, logger=None):
        _write_visdrone_raw(Path(raw_root))
        return [Path(raw_root)]

    monkeypatch.setattr("cape_det.datasets.prepare.download_and_extract", fake_download)
    config = {"dataset": {"download_urls": ["https://example.invalid/fake.zip"]}}
    result = ensure_prepared_dataset("visdrone", config=config, raw_root=raw, prepared_root=prepared)
    assert Path(result["config_path"]).exists()


def test_tinyperson_manual_paths_work_without_raw_root(tmp_path: Path):
    raw = tmp_path / "manual_tiny"
    _write_tinyperson_raw(raw)
    prepared = tmp_path / "prepared_tiny_manual"
    config = {
        "dataset": {
            "manual_paths": {
                "train": {
                    "image_dir": str(raw / "images" / "train"),
                    "annotation_file": str(raw / "annotations" / "train.json"),
                },
                "val": {
                    "image_dir": str(raw / "images" / "val"),
                    "annotation_file": str(raw / "annotations" / "val.json"),
                },
            }
        }
    }
    result = ensure_prepared_dataset(
        "tinyperson",
        config=config,
        raw_root=tmp_path / "does_not_exist",
        prepared_root=prepared,
        allow_download=False,
    )
    assert Path(result["config_path"]).exists()
    summary = validate_prepared_dataset("tinyperson", prepared, splits=("train", "val"))
    assert summary["splits"]["train"]["images"] == 1


def test_dataset_name_cli_prepare_flow(tmp_path: Path):
    raw = tmp_path / "raw_vis"
    prepared = tmp_path / "prepared_vis"
    _write_visdrone_raw(raw)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/sanity_check_dataset.py",
            "--dataset",
            "visdrone",
            "--raw-root",
            str(raw),
            "--prepared-root",
            str(prepared),
            "--no-download",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "annotations" in completed.stdout
