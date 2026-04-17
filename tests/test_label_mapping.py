import pytest

from cape_det.datasets.label_mapping import IGNORE_ANY_LABEL, LabelMapper, build_label_mapper, validate_label_mapper_num_classes


def test_visdrone_unified_merge_and_ignore_policy():
    mapper = LabelMapper("human_unified_single", "merge", dataset_name="visdrone")
    assert mapper.map_annotation("visdrone", 1, None) == (0, False)
    assert mapper.map_annotation("visdrone", 2, None) == (0, False)
    assert mapper.map_annotation("visdrone", 0, None) == (IGNORE_ANY_LABEL, True)
    mapper_ignore = LabelMapper("human_unified_single", "ignore", dataset_name="visdrone")
    assert mapper_ignore.map_annotation("visdrone", 2, None) == (0, True)


def test_split_modes_are_dataset_specific():
    vis = LabelMapper("human_split", dataset_name="visdrone")
    tiny = LabelMapper("human_split", dataset_name="tinyperson")
    assert vis.class_names == ["pedestrian", "people"]
    assert tiny.class_names == ["person"]
    assert vis.map_annotation("visdrone", 1, None) == (0, False)
    assert vis.map_annotation("visdrone", 2, None) == (1, False)
    assert vis.map_annotation("visdrone", 0, None) == (IGNORE_ANY_LABEL, True)
    assert tiny.map_annotation("tinyperson", 1, "person") == (0, False)


def test_tinyperson_mapping_is_config_driven_in_split_mode():
    mapper = build_label_mapper(
        {
            "dataset": {
                "name": "tinyperson",
                "label_mode": "human_split",
                "tinyperson_person_names": ["human"],
                "tinyperson_person_ids": [42],
            }
        }
    )
    assert mapper.class_names == ["person"]
    assert mapper.map_annotation("tinyperson", 42, "human") == (0, False)
    assert mapper.map_annotation("tinyperson", 1, "person") is None


def test_mapper_num_classes_must_match_model_num_classes():
    validate_label_mapper_num_classes(
        {
            "dataset": {"name": "visdrone", "label_mode": "human_split"},
            "model": {"num_classes": 2},
        }
    )
    with pytest.raises(ValueError, match="num_classes"):
        validate_label_mapper_num_classes(
            {
                "dataset": {"name": "tinyperson", "label_mode": "human_split"},
                "model": {"num_classes": 2},
            }
        )
