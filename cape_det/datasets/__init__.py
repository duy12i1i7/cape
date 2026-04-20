from .label_mapping import LabelMapper, build_label_mapper, validate_label_mapper_num_classes
from .prepare import DatasetPreparationError, ensure_prepared_dataset, resolve_dataset_config
from .registry import DatasetSpec, default_experiment_config, get_dataset_spec, registry_snapshot
from .visdrone import VisDroneDataset
from .tinyperson import TinyPersonDataset
from .joint import JointHumanDataset

__all__ = [
    "DatasetPreparationError",
    "DatasetSpec",
    "LabelMapper",
    "build_label_mapper",
    "default_experiment_config",
    "ensure_prepared_dataset",
    "get_dataset_spec",
    "registry_snapshot",
    "resolve_dataset_config",
    "validate_label_mapper_num_classes",
    "VisDroneDataset",
    "TinyPersonDataset",
    "JointHumanDataset",
    "build_dataset",
]


def build_dataset(config: dict, split: str):
    dataset_cfg = config.get("dataset", config)
    if dataset_cfg.get("auto_prepare", False) and not dataset_cfg.get("_resolved_prepared", False):
        config = resolve_dataset_config(config)
        dataset_cfg = config.get("dataset", config)
    name = dataset_cfg.get("name", "").lower()
    if name == "visdrone":
        dataset = VisDroneDataset(dataset_cfg, split=split)
        validate_label_mapper_num_classes(config, dataset.mapper)
        return dataset
    if name == "tinyperson":
        dataset = TinyPersonDataset(dataset_cfg, split=split)
        validate_label_mapper_num_classes(config, dataset.mapper)
        return dataset
    if name == "joint_humans_visdrone_tinyperson":
        dataset = JointHumanDataset(dataset_cfg, split=split)
        validate_label_mapper_num_classes(config, dataset.mapper)
        return dataset
    raise ValueError(f"Unsupported dataset: {dataset_cfg.get('name')}")
