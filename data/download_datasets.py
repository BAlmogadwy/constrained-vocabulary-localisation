# Copyright (c) 2024.
# SPDX-License-Identifier: MIT
"""
Utilities for downloading benchmark datasets used in the traditional detector
evaluation pipeline. The script currently supports COCO 2017 and LVIS v1.0.

Example:
    python data/download_datasets.py --datasets coco2017 lvis_v1
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import tarfile
from typing import Dict, Iterable, List
import zipfile

try:
    import requests
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "The requests package is required to download datasets. "
        "Install it with `pip install requests`."
    ) from exc


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "raw"


def _sha256sum(path: Path) -> str:
    """Compute the SHA256 checksum for a file."""
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _download_file(url: str, destination: Path, overwrite: bool = False) -> None:
    """Download a file if it does not already exist."""
    if destination.exists() and not overwrite:
        print(f"[skip] {destination.name} already exists")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {destination}")

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)


def _extract(archive_path: Path, target_dir: Path) -> None:
    """Extract a .zip or .tar(.gz) archive."""
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    if archive_path.suffix == ".zip":
        mode = "zip"
    elif archive_path.suffix in {".gz", ".tgz", ".bz2", ".xz"} or archive_path.suffixes[-2:] == [
        ".tar",
        ".gz",
    ]:
        mode = "tar"
    elif archive_path.suffix == ".tar":
        mode = "tar"
    else:
        print(f"[skip] No extraction rule for {archive_path.name}")
        return

    marker = target_dir / f".extracted_{archive_path.stem}"
    if marker.exists():
        print(f"[skip] {archive_path.name} already extracted")
        return

    print(f"[extract] {archive_path} -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    if mode == "zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(target_dir)
    else:
        with tarfile.open(archive_path) as tf:
            tf.extractall(target_dir)

    marker.touch()


DatasetFile = Dict[str, str]


def _dataset_manifest() -> Dict[str, Dict[str, Iterable[DatasetFile]]]:
    """Return metadata describing supported datasets."""
    return {
        "coco2017": {
            "description": "Full COCO 2017 release (train/val images + annotations).",
            "files": [
                {
                    "url": "http://images.cocodataset.org/zips/train2017.zip",
                    "destination": "coco2017/train2017.zip",
                    "extract_to": "coco2017",
                },
                {
                    "url": "http://images.cocodataset.org/zips/val2017.zip",
                    "destination": "coco2017/val2017.zip",
                    "extract_to": "coco2017",
                    "sha256": "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05",
                },
                {
                    "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                    "destination": "coco2017/annotations_trainval2017.zip",
                    "extract_to": "coco2017",
                    "sha256": "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268",
                },
            ],
        },
        "lvis_v1": {
            "description": "LVIS v1.0 annotations for zero-shot benchmarks (images are shared with COCO).",
            "files": [
                {
                    "url": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
                    "destination": "lvis/lvis_v1_train.json.zip",
                    "extract_to": "lvis",
                    "sha256": "334a4caa374030a7817cf050364525e910f7960f9b6968cef47cffbf3893f8ba",
                },
                {
                    "url": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
                    "destination": "lvis/lvis_v1_val.json.zip",
                    "extract_to": "lvis",
                    "sha256": "5cae9a3c79aadb667550c2b5dcf7f4d86e059a41ec91ef690225b667e28e9ba5",
                },
            ],
        },
    }


def download_dataset(
    dataset_name: str,
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    """Download and optionally extract an entire dataset."""
    manifest = _dataset_manifest()
    if dataset_name not in manifest:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Valid options: {sorted(manifest)}")

    dataset_info = manifest[dataset_name]
    print(f"==> Downloading {dataset_name}: {dataset_info['description']}")
    for file_cfg in dataset_info["files"]:
        destination = output_dir / file_cfg["destination"]
        _download_file(file_cfg["url"], destination, overwrite=overwrite)

        expected_hash = file_cfg.get("sha256")
        if expected_hash:
            actual_hash = _sha256sum(destination)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Checksum mismatch for {destination} (expected {expected_hash}, got {actual_hash})"
                )

        extract_root = file_cfg.get("extract_to")
        if extract_root:
            target = output_dir / extract_root
            _extract(destination, target)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download datasets for zero-shot benchmarking.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["coco2017", "lvis_v1"],
        help="Datasets to download. Use 'all' to fetch everything.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory where datasets will be stored.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the list of supported datasets and exit.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    manifest = _dataset_manifest()
    if args.list:
        print("Available datasets:")
        for name, info in manifest.items():
            print(f"  - {name}: {info['description']}")
        return

    datasets = manifest.keys() if "all" in args.datasets else args.datasets
    for ds in datasets:
        download_dataset(ds, args.output_dir, overwrite=args.overwrite)

    print("All requested datasets processed.")


if __name__ == "__main__":
    main(sys.argv[1:])
