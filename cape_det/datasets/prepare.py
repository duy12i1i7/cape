from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from cape_det.utils.config import deep_merge, load_config

from .download import download_and_extract
from .formats import write_prepared_config
from .registry import (
    DatasetSpec,
    dataset_prepared_root,
    dataset_raw_root,
    get_dataset_spec,
    prepared_config_path,
)
from .tinyperson_prepare import prepare_tinyperson
from .validators import DatasetValidationError, validate_prepared_dataset
from .visdrone_prepare import prepare_visdrone


LogFn = Callable[[str], None]


class DatasetPreparationError(RuntimeError):
    pass


def _log(message: str, logger: LogFn | None = None) -> None:
    if logger is not None:
        logger(message)
    else:
        print(message)


def _dataset_cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    return (config or {}).get("dataset", config or {})


def _splits(config: dict[str, Any] | None) -> tuple[str, ...]:
    dataset_cfg = _dataset_cfg(config)
    splits = [dataset_cfg.get("train_split", "train"), dataset_cfg.get("val_split", "val")]
    test_split = dataset_cfg.get("test_split", "test")
    if test_split:
        splits.append(test_split)
    seen: list[str] = []
    for split in splits:
        if split not in seen:
            seen.append(str(split))
    return tuple(seen)


def _load_prepared_config(config_path: Path, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    if config_path.exists():
        prepared = load_config(config_path)
    else:
        prepared = {}
    if fallback:
        prepared = deep_merge(fallback, prepared)
    return prepared


def _has_manual_paths(config: dict[str, Any] | None) -> bool:
    manual = _dataset_cfg(config).get("manual_paths", {})
    if not isinstance(manual, dict):
        return False
    return any(bool(value) for value in manual.values())


def _regenerate_prepared_config_if_missing(
    spec: DatasetSpec,
    prepared_root: Path,
    config_path: Path,
    config: dict[str, Any] | None,
) -> None:
    if config_path.exists():
        return
    dataset_cfg = _dataset_cfg(config)
    write_prepared_config(
        spec.name,
        prepared_root,
        config_path,
        label_mode=dataset_cfg.get("label_mode", "human_unified_single"),
    )


def _prepare_from_raw(
    spec: DatasetSpec,
    raw_root: Path,
    prepared_root: Path,
    config: dict[str, Any] | None,
    splits: tuple[str, ...],
    logger: LogFn | None = None,
) -> dict[str, Any]:
    dataset_cfg = _dataset_cfg(config)
    label_mode = dataset_cfg.get("label_mode", "human_unified_single")
    link_mode = dataset_cfg.get("prepare_link_mode", "symlink")
    if spec.name == "visdrone":
        return prepare_visdrone(raw_root, prepared_root, label_mode=label_mode, splits=splits, link_mode=link_mode, logger=logger)
    if spec.name == "tinyperson":
        return prepare_tinyperson(
            raw_root,
            prepared_root,
            label_mode=label_mode,
            splits=splits,
            link_mode=link_mode,
            config=dataset_cfg,
            logger=logger,
        )
    raise DatasetPreparationError(f"No preparation routine registered for {spec.name}")


def ensure_prepared_dataset(
    dataset: str,
    config: dict[str, Any] | None = None,
    raw_root: str | Path | None = None,
    prepared_root: str | Path | None = None,
    allow_download: bool = True,
    force_prepare: bool = False,
    logger: LogFn | None = None,
) -> dict[str, Any]:
    spec = get_dataset_spec(dataset)
    raw = dataset_raw_root(spec, config=config, raw_root=raw_root)
    prepared = dataset_prepared_root(spec, config=config, prepared_root=prepared_root)
    splits = _splits(config)
    validation_splits = tuple(split for split in splits if split != "test")
    config_path = prepared_config_path(spec, prepared)
    urls = tuple(_dataset_cfg(config).get("download_urls", ()) or spec.download_urls)

    if not force_prepare:
        try:
            validation = validate_prepared_dataset(spec.name, prepared, splits=validation_splits)
            _log(f"[dataset] using prepared {spec.name}: {prepared}", logger)
            _regenerate_prepared_config_if_missing(spec, prepared, config_path, config)
            prepared_config = _load_prepared_config(config_path, fallback=config)
            return {"config_path": config_path, "config": prepared_config, "summary": validation, "prepared_root": prepared}
        except DatasetValidationError as exc:
            _log(f"[dataset] prepared {spec.name} not ready: {exc}", logger)

    if raw.exists() or _has_manual_paths(config):
        _log(f"[dataset] preparing {spec.name} from raw root: {raw}", logger)
        try:
            result = _prepare_from_raw(spec, raw, prepared, config, splits, logger=logger)
            prepared_config = _load_prepared_config(Path(result["config_path"]), fallback=config)
            result["config"] = prepared_config
            result["prepared_root"] = prepared
            return result
        except (FileNotFoundError, DatasetValidationError) as exc:
            if not (allow_download and urls):
                raise
            _log(f"[dataset] raw preparation failed; falling back to download for {spec.name}: {exc}", logger)

    if allow_download and urls:
        _log(f"[dataset] raw root missing; attempting download for {spec.name}: {raw}", logger)
        raw.mkdir(parents=True, exist_ok=True)
        download_and_extract(urls, raw, logger=logger)
        result = _prepare_from_raw(spec, raw, prepared, config, splits, logger=logger)
        prepared_config = _load_prepared_config(Path(result["config_path"]), fallback=config)
        result["config"] = prepared_config
        result["prepared_root"] = prepared
        return result

    message = (
        f"Dataset '{spec.name}' is not prepared and raw data was not found at {raw}. "
        f"Set CAPE_DATA_ROOT or {spec.raw_env}, or pass --raw-root. "
    )
    if not spec.download_urls:
        message += spec.notes
    raise DatasetPreparationError(message)


def resolve_dataset_config(
    config: dict[str, Any],
    dataset_name: str | None = None,
    raw_root: str | Path | None = None,
    prepared_root: str | Path | None = None,
    allow_download: bool = True,
    force_prepare: bool = False,
    logger: LogFn | None = None,
) -> dict[str, Any]:
    dataset_cfg = _dataset_cfg(config)
    name = dataset_name or dataset_cfg.get("registry_name") or dataset_cfg.get("name")
    if not name:
        raise DatasetPreparationError("Cannot resolve dataset: missing dataset.name")
    auto_prepare = bool(dataset_cfg.get("auto_prepare", False)) or dataset_name is not None
    if not auto_prepare:
        return config
    result = ensure_prepared_dataset(
        name,
        config=config,
        raw_root=raw_root,
        prepared_root=prepared_root,
        allow_download=allow_download,
        force_prepare=force_prepare,
        logger=logger,
    )
    resolved = result["config"]
    resolved.setdefault("dataset", {})["_resolved_prepared"] = True
    return resolved
