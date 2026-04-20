from __future__ import annotations

import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable


LogFn = Callable[[str], None]


def _log(message: str, logger: LogFn | None = None) -> None:
    if logger is not None:
        logger(message)
    else:
        print(message)


def download_file(
    url: str,
    destination: str | Path,
    retries: int = 3,
    timeout: int = 60,
    logger: LogFn | None = None,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        _log(f"[dataset] reuse cached archive: {destination}", logger)
        return destination

    last_error: Exception | None = None
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            _log(f"[dataset] downloading ({attempt}/{retries}): {url}", logger)
            with urllib.request.urlopen(url, timeout=timeout) as response, tmp_path.open("wb") as f:
                shutil.copyfileobj(response, f)
            tmp_path.replace(destination)
            return destination
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def extract_archive(archive_path: str | Path, destination: str | Path, logger: LogFn | None = None) -> Path:
    archive_path = Path(archive_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / f".extracted_{archive_path.name}.complete"
    if marker.exists():
        _log(f"[dataset] reuse extracted archive: {archive_path.name}", logger)
        return destination

    _log(f"[dataset] extracting {archive_path} -> {destination}", logger)
    suffixes = "".join(archive_path.suffixes).lower()
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(destination)
    elif suffixes.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")) or archive_path.suffix.lower() == ".tar":
        with tarfile.open(archive_path) as tf:
            tf.extractall(destination)
    else:
        raise ValueError(f"Unsupported archive type: {archive_path}")
    marker.write_text("ok\n", encoding="utf-8")
    return destination


def download_and_extract(
    urls: list[str] | tuple[str, ...],
    raw_root: str | Path,
    retries: int = 3,
    logger: LogFn | None = None,
) -> list[Path]:
    raw_root = Path(raw_root)
    archive_dir = raw_root / "archives"
    extracted_paths: list[Path] = []
    for url in urls:
        filename = url.rstrip("/").split("/")[-1]
        if not filename:
            raise ValueError(f"Cannot infer archive filename from URL: {url}")
        archive_path = download_file(url, archive_dir / filename, retries=retries, logger=logger)
        extracted_paths.append(extract_archive(archive_path, raw_root, logger=logger))
    return extracted_paths
