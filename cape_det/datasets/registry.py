from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    aliases: tuple[str, ...]
    prepared_config_name: str
    raw_env: str
    prepared_subdir: str
    download_urls: tuple[str, ...] = field(default_factory=tuple)
    citation_url: str = ""
    notes: str = ""

    def matches(self, value: str) -> bool:
        normalized = normalize_dataset_name(value)
        return normalized == self.name or normalized in self.aliases


def normalize_dataset_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def default_data_root() -> Path:
    return Path(os.environ.get("CAPE_DATA_ROOT", REPO_ROOT / "data")).expanduser()


def _split_env_urls(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(url.strip() for url in value.split(",") if url.strip())


def _visdrone_urls() -> tuple[str, ...]:
    override = _split_env_urls(os.environ.get("VISDRONE_DOWNLOAD_URLS"))
    if override:
        return override
    return (
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip",
    )


def _tinyperson_urls() -> tuple[str, ...]:
    override = _split_env_urls(os.environ.get("TINYPERSON_DOWNLOAD_URLS") or os.environ.get("TINYPERSON_DOWNLOAD_URL"))
    if override:
        return override
    return ("tinyperson://minimal",)


DATASET_SPECS: dict[str, DatasetSpec] = {
    "visdrone": DatasetSpec(
        name="visdrone",
        aliases=("visdrone2019", "visdrone_det", "visdrone2019_det"),
        prepared_config_name="visdrone_prepared.yaml",
        raw_env="VISDRONE_RAW_ROOT",
        prepared_subdir="prepared/visdrone",
        download_urls=_visdrone_urls(),
        citation_url="https://github.com/VisDrone/VisDrone-Dataset",
        notes="VisDrone auto-download uses public Ultralytics-hosted DET archives when no raw root is provided.",
    ),
    "tinyperson": DatasetSpec(
        name="tinyperson",
        aliases=("tiny_person", "tiny_persons"),
        prepared_config_name="tinyperson_prepared.yaml",
        raw_env="TINYPERSON_RAW_ROOT",
        prepared_subdir="prepared/tinyperson",
        download_urls=_tinyperson_urls(),
        citation_url="https://github.com/ucas-vg/TinyBenchmark",
        notes=(
            "TinyPerson auto-download uses official Google Drive release assets when gdown is installed. "
            "Set TINYPERSON_RAW_ROOT for a local copy or TINYPERSON_DOWNLOAD_URLS for a mirror."
        ),
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    normalized = normalize_dataset_name(name)
    for spec in DATASET_SPECS.values():
        if spec.matches(normalized):
            return spec
    raise KeyError(f"Unsupported dataset '{name}'. Supported datasets: {', '.join(sorted(DATASET_SPECS))}")


def dataset_raw_root(spec: DatasetSpec, config: dict[str, Any] | None = None, raw_root: str | Path | None = None) -> Path:
    dataset_cfg = (config or {}).get("dataset", config or {})
    if raw_root is not None:
        return Path(raw_root).expanduser()
    if dataset_cfg.get("raw_root"):
        return Path(dataset_cfg["raw_root"]).expanduser()
    if os.environ.get(spec.raw_env):
        return Path(os.environ[spec.raw_env]).expanduser()
    return default_data_root() / "raw" / spec.name


def dataset_prepared_root(
    spec: DatasetSpec,
    config: dict[str, Any] | None = None,
    prepared_root: str | Path | None = None,
) -> Path:
    dataset_cfg = (config or {}).get("dataset", config or {})
    if prepared_root is not None:
        return Path(prepared_root).expanduser()
    if dataset_cfg.get("prepared_root"):
        return Path(dataset_cfg["prepared_root"]).expanduser()
    if dataset_cfg.get("prepared") and dataset_cfg.get("root"):
        return Path(dataset_cfg["root"]).expanduser()
    return default_data_root() / spec.prepared_subdir


def default_experiment_config(dataset_name: str, model_mode: str = "cape") -> Path:
    spec = get_dataset_spec(dataset_name)
    model_mode = normalize_dataset_name(model_mode)
    if model_mode not in {"cape", "baseline"}:
        raise ValueError("model_mode must be 'cape' or 'baseline'")
    return REPO_ROOT / "configs" / "experiments" / f"{spec.name}_{model_mode}.yaml"


def prepared_config_path(spec: DatasetSpec, prepared_root: Path) -> Path:
    return prepared_root / "metadata" / spec.prepared_config_name


def registry_snapshot() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "name": spec.name,
            "aliases": list(spec.aliases),
            "raw_env": spec.raw_env,
            "prepared_subdir": spec.prepared_subdir,
            "download_urls": list(spec.download_urls),
            "citation_url": spec.citation_url,
            "notes": spec.notes,
        }
        for name, spec in DATASET_SPECS.items()
    }
