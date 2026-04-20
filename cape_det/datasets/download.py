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

TINYPERSON_MINIMAL_ASSETS: dict[str, str] = {
    "tiny_set/train.tar.gz": "1tqGbW7_3X_-CpQvZ9ls3tYTJafsoKOr4",
    "tiny_set/test.tar.gz": "1uq148D2Nxs3JiHJmZW8zT1DEnd6cPelF",
    "tiny_set/annotations/tiny_set_train.json": "1vo-ggU2lltIIze9tIMhBCRpFEz3h2Hv4",
    "tiny_set/annotations/task/tiny_set_test_all.json": "16mIDH58dukozi2iQwBqDTURYBDNNWrxy",
}


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


def is_google_drive_folder_url(url: str) -> bool:
    return "drive.google.com" in url and ("/folders/" in url or "drive/folders" in url)


def is_google_drive_url(url: str) -> bool:
    return "drive.google.com" in url


def is_tinyperson_minimal_url(url: str) -> bool:
    return url.strip().lower() in {"tinyperson://minimal", "tinyperson:minimal"}


def download_google_drive_folder(url: str, destination: str | Path, logger: LogFn | None = None) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".gdrive_folder_download.complete"
    if marker.exists():
        _log(f"[dataset] reuse Google Drive folder download: {destination}", logger)
        return destination
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "TinyPerson Google Drive folder auto-download requires gdown. "
            "Install with `python -m pip install gdown` or use TINYPERSON_RAW_ROOT."
        ) from exc
    _log(f"[dataset] downloading Google Drive folder: {url}", logger)
    result = gdown.download_folder(url=url, output=str(destination), quiet=False, use_cookies=False)
    if result is None:
        raise RuntimeError(f"Failed to download Google Drive folder: {url}")
    marker.write_text("ok\n", encoding="utf-8")
    return destination


def download_google_drive_file(url: str, destination: str | Path, logger: LogFn | None = None) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        _log(f"[dataset] reuse cached Google Drive file: {destination}", logger)
        return destination
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive auto-download requires gdown. Install with `python -m pip install gdown`."
        ) from exc
    _log(f"[dataset] downloading Google Drive file: {url}", logger)
    result = gdown.download(url=url, output=str(destination), quiet=False, fuzzy=True, use_cookies=False)
    if result is None:
        raise RuntimeError(f"Failed to download Google Drive file: {url}")
    return Path(result)


def _archive_output_dir(path: Path) -> Path:
    name = path.name
    if name.endswith(".tar.gz"):
        return path.with_name(name[:-7])
    if name.endswith(".tgz"):
        return path.with_name(name[:-4])
    if name.endswith(".tar.bz2"):
        return path.with_name(name[:-8])
    if name.endswith(".tar.xz"):
        return path.with_name(name[:-7])
    return path.with_suffix("")


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


def extract_archives_under(root: str | Path, logger: LogFn | None = None) -> list[Path]:
    root = Path(root)
    extracted: list[Path] = []
    if not root.exists():
        return extracted
    for archive in sorted(root.rglob("*")):
        if not archive.is_file():
            continue
        lower = archive.name.lower()
        if not (
            lower.endswith(".zip")
            or lower.endswith(".tar")
            or lower.endswith(".tar.gz")
            or lower.endswith(".tgz")
            or lower.endswith(".tar.bz2")
            or lower.endswith(".tar.xz")
        ):
            continue
        extracted.append(extract_archive(archive, _archive_output_dir(archive), logger=logger))
    return extracted


def download_tinyperson_minimal_assets(raw_root: str | Path, logger: LogFn | None = None) -> list[Path]:
    raw_root = Path(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for rel_path, file_id in TINYPERSON_MINIMAL_ASSETS.items():
        destination = raw_root / rel_path
        url = f"https://drive.google.com/uc?id={file_id}"
        downloaded.append(download_google_drive_file(url, destination, logger=logger))
    extracted = extract_archives_under(raw_root, logger=logger)
    return downloaded + extracted


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
        if is_tinyperson_minimal_url(url):
            extracted_paths.extend(download_tinyperson_minimal_assets(raw_root, logger=logger))
            continue
        if is_google_drive_folder_url(url):
            extracted_paths.append(download_google_drive_folder(url, raw_root, logger=logger))
            extracted_paths.extend(extract_archives_under(raw_root, logger=logger))
            continue
        filename = url.rstrip("/").split("/")[-1]
        if not filename:
            raise ValueError(f"Cannot infer archive filename from URL: {url}")
        if is_google_drive_url(url):
            archive_path = download_google_drive_file(url, archive_dir / filename, logger=logger)
        else:
            archive_path = download_file(url, archive_dir / filename, retries=retries, logger=logger)
        extracted_paths.append(extract_archive(archive_path, raw_root, logger=logger))
    return extracted_paths
