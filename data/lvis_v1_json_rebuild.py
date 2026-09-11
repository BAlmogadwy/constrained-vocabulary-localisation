#!/usr/bin/env python3
"""
Rebuild the LVIS v1 unseen validation subset specified in
`data/processed/lvis_v1/val/lvis_v1_val_unseen.json`.

The script populates `data/subsets/lvis_v1_val_unseen/images` (by default) by
symlinking to locally available COCO val2017 images when possible, and falls
back to downloading missing images using the URLs provided in the metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


DEFAULT_METADATA = "data/processed/lvis_v1/val/lvis_v1_val_unseen.json"
DEFAULT_OUTPUT_DIR = "data/subsets/lvis_v1_val_unseen/images"
DEFAULT_COCO_ROOT = "data/raw/coco2017/val2017"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct LVIS inference subset from metadata JSON."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(DEFAULT_METADATA),
        help="Path to LVIS metadata JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory where reconstructed images will be stored.",
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=Path(DEFAULT_COCO_ROOT),
        help="Path to the local COCO val2017 image directory.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy local images instead of creating symlinks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files in the output directory.",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def infer_filename(image_entry: Dict) -> str:
    candidates: Iterable[Optional[str]] = (
        image_entry.get("file_name"),
        image_entry.get("coco_url"),
        image_entry.get("flickr_url"),
    )
    for candidate in candidates:
        if not candidate:
            continue
        name = Path(str(candidate)).name
        if name:
            return name
    raise ValueError(f"Unable to infer filename for image entry: {image_entry}")


def symlink_or_copy(src: Path, dst: Path, use_copy: bool) -> None:
    if use_copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def download_file(url: str, dst: Path) -> None:
    ensure_directory(dst.parent)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(dst.parent)) as tmp_handle:
        tmp_path = Path(tmp_handle.name)
        try:
            with urlopen(url) as response:
                shutil.copyfileobj(response, tmp_handle)
        except (HTTPError, URLError) as exc:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    dst.unlink(missing_ok=True)
    os.replace(tmp_path, dst)


def rebuild_dataset(
    metadata_path: Path,
    output_dir: Path,
    coco_root: Path,
    use_copy: bool,
    overwrite: bool,
) -> None:
    data = load_metadata(metadata_path)
    images = data.get("images", [])
    ensure_directory(output_dir)

    stats = {
        "total": len(images),
        "linked": 0,
        "downloaded": 0,
        "skipped": 0,
        "errors": 0,
    }

    for idx, image_entry in enumerate(images, start=1):
        try:
            filename = infer_filename(image_entry)
        except ValueError as exc:
            print(f"[warn] {exc}", file=sys.stderr)
            stats["errors"] += 1
            continue

        destination = output_dir / filename
        if destination.exists():
            if overwrite:
                destination.unlink()
            else:
                stats["skipped"] += 1
                continue

        local_candidate = coco_root / filename
        if local_candidate.exists():
            try:
                symlink_or_copy(local_candidate, destination, use_copy)
                stats["linked"] += 1
                continue
            except OSError as exc:
                print(
                    f"[warn] Failed to create {'copy' if use_copy else 'symlink'} for {local_candidate}: {exc}",
                    file=sys.stderr,
                )

        urls = [image_entry.get("coco_url"), image_entry.get("flickr_url")]
        success = False
        for url in urls:
            if not url:
                continue
            try:
                print(f"[download] {filename} ← {url}")
                download_file(url, destination)
                stats["downloaded"] += 1
                success = True
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] Download failed for {url}: {exc}", file=sys.stderr)
        if not success:
            stats["errors"] += 1

        if idx % 50 == 0 or idx == stats["total"]:
            print(
                f"[progress] {idx}/{stats['total']} processed "
                f"(linked={stats['linked']}, downloaded={stats['downloaded']}, "
                f"skipped={stats['skipped']}, errors={stats['errors']})"
            )

    print(
        "\nCompleted rebuild:\n"
        f"  total images     : {stats['total']}\n"
        f"  linked/copied    : {stats['linked']}\n"
        f"  downloaded       : {stats['downloaded']}\n"
        f"  skipped existing : {stats['skipped']}\n"
        f"  errors           : {stats['errors']}"
    )


def main() -> int:
    args = parse_args()
    try:
        rebuild_dataset(
            metadata_path=args.metadata,
            output_dir=args.output_dir,
            coco_root=args.coco_root,
            use_copy=args.copy,
            overwrite=args.overwrite,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
