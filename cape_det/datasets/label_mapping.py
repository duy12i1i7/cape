from __future__ import annotations

from dataclasses import dataclass
from typing import Any


IGNORE_ANY_LABEL = -1

VISDRONE_CATEGORY_NAMES: dict[int, str] = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}


@dataclass(frozen=True)
class LabelMappingResult:
    label: int
    ignore: bool


class LabelMapper:
    """Config-driven human label protocol for VisDrone and TinyPerson."""

    def __init__(
        self,
        label_mode: str = "human_unified_single",
        visdrone_people_policy: str = "merge",
        tinyperson_person_names: tuple[str, ...] = ("person", "people", "pedestrian"),
        dataset_name: str | None = None,
        visdrone_category_names: dict[int | str, str] | None = None,
        visdrone_ignore_category_ids: tuple[int | str, ...] = (0,),
        tinyperson_person_ids: tuple[int | str, ...] = (1, "person"),
    ) -> None:
        if label_mode not in {"human_unified_single", "human_split"}:
            raise ValueError(f"Unsupported label_mode: {label_mode}")
        if visdrone_people_policy not in {"merge", "ignore"}:
            raise ValueError("visdrone_people_policy must be 'merge' or 'ignore'")
        self.label_mode = label_mode
        self.visdrone_people_policy = visdrone_people_policy
        self.visdrone_category_names = {
            str(key): str(value).lower()
            for key, value in (visdrone_category_names or VISDRONE_CATEGORY_NAMES).items()
        }
        self.visdrone_ignore_category_ids = {str(value) for value in visdrone_ignore_category_ids}
        self.tinyperson_person_names = {n.lower() for n in tinyperson_person_names}
        self.tinyperson_person_ids = {str(value) for value in tinyperson_person_ids}
        self.dataset_name = (dataset_name or "").lower()

    @property
    def class_names(self) -> list[str]:
        if self.label_mode == "human_unified_single":
            return ["person"]
        if self.dataset_name == "visdrone":
            return ["pedestrian", "people"]
        if self.dataset_name == "tinyperson":
            return ["person"]
        return ["pedestrian", "people", "person"]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def _visdrone_name(self, category_id: int | str, category_name: str | None = None) -> str:
        if category_name:
            return str(category_name).lower()
        return self.visdrone_category_names.get(str(category_id), str(category_id)).lower()

    def map_annotation(
        self,
        dataset_name: str,
        category_id: int | str,
        category_name: str | None = None,
        source_ignore: bool = False,
    ) -> tuple[int, bool] | None:
        dataset = dataset_name.lower()
        if dataset == "visdrone":
            return self._map_visdrone(category_id, category_name, source_ignore)
        if dataset == "tinyperson":
            return self._map_tinyperson(category_id, category_name, source_ignore)
        if dataset.startswith("joint"):
            name = str(category_name or category_id).lower()
            if name in {"pedestrian", "people", "person"}:
                return (0, bool(source_ignore)) if self.label_mode == "human_unified_single" else (self.class_names.index(name), bool(source_ignore))
        return None

    def _map_visdrone(
        self, category_id: int | str, category_name: str | None, source_ignore: bool
    ) -> tuple[int, bool] | None:
        name = self._visdrone_name(category_id, category_name)
        if name == "ignored" or str(category_id) in self.visdrone_ignore_category_ids:
            return IGNORE_ANY_LABEL, True
        if self.label_mode == "human_unified_single":
            if name == "pedestrian":
                return 0, bool(source_ignore)
            if name == "people":
                return 0, bool(source_ignore or self.visdrone_people_policy == "ignore")
            return None
        if name == "pedestrian":
            return 0, bool(source_ignore)
        if name == "people":
            return 1, bool(source_ignore)
        return None

    def _map_tinyperson(
        self, category_id: int | str, category_name: str | None, source_ignore: bool
    ) -> tuple[int, bool] | None:
        name = str(category_name or category_id).lower()
        if name in self.tinyperson_person_names or str(category_id) in self.tinyperson_person_ids:
            if self.label_mode == "human_unified_single":
                return 0, bool(source_ignore)
            return self.class_names.index("person"), bool(source_ignore)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_mode": self.label_mode,
            "visdrone_people_policy": self.visdrone_people_policy,
            "visdrone_ignore_category_ids": sorted(self.visdrone_ignore_category_ids),
            "tinyperson_person_names": sorted(self.tinyperson_person_names),
            "tinyperson_person_ids": sorted(self.tinyperson_person_ids),
            "class_names": self.class_names,
        }


def build_label_mapper(config: dict[str, Any]) -> LabelMapper:
    dataset_cfg = config.get("dataset", config)
    return LabelMapper(
        label_mode=dataset_cfg.get("label_mode", config.get("eval", {}).get("mode", "human_unified_single")),
        visdrone_people_policy=dataset_cfg.get("visdrone_people_policy", "merge"),
        visdrone_category_names=dataset_cfg.get("visdrone_category_names"),
        visdrone_ignore_category_ids=tuple(dataset_cfg.get("visdrone_ignore_category_ids", (0,))),
        tinyperson_person_names=tuple(dataset_cfg.get("tinyperson_person_names", ("person", "people", "pedestrian"))),
        tinyperson_person_ids=tuple(dataset_cfg.get("tinyperson_person_ids", (1, "person"))),
        dataset_name=dataset_cfg.get("name"),
    )


def validate_label_mapper_num_classes(config: dict[str, Any], mapper: LabelMapper | None = None) -> None:
    model_cfg = config.get("model", {})
    if "num_classes" not in model_cfg:
        return
    mapper = mapper or build_label_mapper(config)
    model_num_classes = int(model_cfg["num_classes"])
    if mapper.num_classes != model_num_classes:
        raise ValueError(
            "Label mapper num_classes does not match model num_classes: "
            f"mapper={mapper.num_classes} classes={mapper.class_names}, model={model_num_classes}. "
            "Set dataset.label_mode/dataset mapping config and model.num_classes consistently."
        )
