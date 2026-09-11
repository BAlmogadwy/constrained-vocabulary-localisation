# Copyright (c) 2024.
# SPDX-License-Identifier: MIT
"""Download pretrained weights for traditional zero-shot detectors."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List

try:
    import requests
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "The requests package is required to download model weights. "
        "Install it with `pip install requests`."
    ) from exc


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "models" / "traditional" / "weights"


def _sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest() -> Dict[str, Dict[str, str]]:
    return {
        "yolo_world": {
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-world.pt",
            "filename": "yolov8x-world.pt",
            "sha256": "9b99398e46cffbf2b9a7e668512fa295f0d710d173ae0a815ec706ced5d1099b",
        },
        "grounding_dino": {
            "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
            "filename": "groundingdino_swinb_cogcoor.pth",
            "sha256": "46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7",
        },
        "owl_vit": {
            "url": "https://huggingface.co/google/owlvit-base-patch32/resolve/main/pytorch_model.bin",
            "filename": "owlvit-base-patch32.bin",
            "sha256": "02437334ba11db56a312a6c3f0f351f610a1075751bfe36b689636a826d5b531",
        },
    }


def download_weight(
    model_name: str,
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    manifest = _manifest()
    if model_name not in manifest:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(manifest)}")

    cfg = manifest[model_name]
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / cfg["filename"]
    if destination.exists() and not overwrite:
        print(f"[skip] {model_name} weights already downloaded")
        return destination

    print(f"[download] {model_name}: {cfg['url']}")
    with requests.get(cfg["url"], stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)

    expected = cfg["sha256"]
    actual = _sha256sum(destination)
    if actual != expected:
        raise RuntimeError(
            f"Checksum mismatch for {destination} (expected {expected}, got {actual})"
        )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pretrained weights for traditional models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo_world", "grounding_dino", "owl_vit"],
        help="Model identifiers to download. Use 'all' to fetch everything.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where weights will be stored.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download weights even if the file already exists.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit.",
    )
    return parser.parse_args()


def main(argv: List[str] | None = None) -> None:
    args = parse_args()
    manifest = _manifest()
    if args.list:
        print("Available models:")
        for name, cfg in manifest.items():
            print(f"  - {name}: {cfg['filename']}")
        return

    models = manifest.keys() if "all" in args.models else args.models
    for model_name in models:
        download_weight(model_name, args.output_dir, overwrite=args.overwrite)
    print("All requested models processed.")


if __name__ == "__main__":
    main()
