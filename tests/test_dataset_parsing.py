import json
from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from cape_det.datasets.tinyperson import TinyPersonDataset
from cape_det.datasets.label_mapping import IGNORE_ANY_LABEL
from cape_det.datasets.visdrone import VisDroneDataset, parse_visdrone_annotation_file


def test_visdrone_parser_xywh_to_xyxy(tmp_path: Path):
    ann = tmp_path / "0001.txt"
    ann.write_text("10,20,5,6,1,1,0,0\n", encoding="utf-8")
    parsed = parse_visdrone_annotation_file(ann)
    assert parsed[0].box_xyxy == (10.0, 20.0, 15.0, 26.0)
    assert parsed[0].category_name == "pedestrian"


def test_dataset_sample_contracts(tmp_path: Path):
    vis_root = tmp_path / "vis"
    (vis_root / "images/train").mkdir(parents=True)
    (vis_root / "annotations/train").mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(vis_root / "images/train/a.jpg")
    (vis_root / "annotations/train/a.txt").write_text("1,2,5,6,1,1,0,0\n", encoding="utf-8")
    ds = VisDroneDataset({"root": str(vis_root), "num_workers": 0}, "train")
    image, target = ds[0]
    assert image.shape[0] == 3
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [0]

    tiny_root = tmp_path / "tiny"
    (tiny_root / "images/train").mkdir(parents=True)
    (tiny_root / "annotations").mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(tiny_root / "images/train/a.jpg")
    data = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 32, "height": 24}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 5, 6]}],
    }
    (tiny_root / "annotations/train.json").write_text(json.dumps(data), encoding="utf-8")
    ds = TinyPersonDataset({"root": str(tiny_root), "num_workers": 0}, "train")
    _, target = ds[0]
    assert target["dataset"] == "tinyperson"
    assert target["boxes"].shape == (1, 4)


def test_visdrone_preserves_ignored_regions_as_ignore_boxes(tmp_path: Path):
    vis_root = tmp_path / "vis"
    (vis_root / "images/train").mkdir(parents=True)
    (vis_root / "annotations/train").mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(vis_root / "images/train/a.jpg")
    (vis_root / "annotations/train/a.txt").write_text(
        "1,2,5,6,1,1,0,0\n10,10,8,8,0,0,0,0\n",
        encoding="utf-8",
    )
    ds = VisDroneDataset({"root": str(vis_root), "num_workers": 0}, "train")
    _, target = ds[0]

    assert target["boxes"].shape == (2, 4)
    assert target["labels"].tolist() == [0, IGNORE_ANY_LABEL]
    assert target["ignore"].tolist() == [False, True]
