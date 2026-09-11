# Copyright (c) 2024.
# SPDX-License-Identifier: MIT
"""
Run inference with traditional zero-shot detectors on a directory of images.

Example:
    python experiments/run_traditional_inference.py \
        --model grounding_dino \
        --images-dir data/raw/coco2017/val2017 \
        --labels-file prompts/coco_unseen_labels.txt \
        --output results/raw/grounding_dino/coco2017_val.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.traditional import (
    GroundingDINODetector,
    OWLVitDetector,
    YOLOWorldDetector,
)


MODEL_REGISTRY = {
    "yolo_world": YOLOWorldDetector,
    "grounding_dino": GroundingDINODetector,
    "owl_vit": OWLVitDetector,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with traditional detectors.")
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_REGISTRY.keys(),
        help="Traditional detector to run.",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        type=Path,
        help="Directory containing images for evaluation.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Text labels/prompts to detect (space separated).",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        help="Optional text file with one label per line.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of images processed (useful for smoke-tests).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the JSON results will be written.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device placement (e.g., 'cuda:0', 'cpu').",
    )
    return parser.parse_args()


def load_labels(args: argparse.Namespace) -> List[str]:
    labels: List[str] = []
    if args.labels:
        labels.extend(args.labels)
    if args.labels_file:
        with args.labels_file.open("r", encoding="utf-8") as fh:
            labels.extend([line.strip() for line in fh if line.strip()])
    if not labels:
        raise ValueError("No labels provided. Use --labels or --labels-file.")
    return labels


def gather_images(images_dir: Path, limit: int | None = None) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if limit:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise RuntimeError(f"No image files found in {images_dir}")
    return image_paths


def as_serializable(detections) -> List[Dict]:
    serializable = []
    for det in detections:
        box = [float(value) for value in det.box]
        serializable.append(
            {
                "label": det.label,
                "score": det.score,
                "box_2d": box,
                "box": box,
            }
        )
    return serializable


def main() -> None:
    args = parse_args()
    labels = load_labels(args)

    detector_cls = MODEL_REGISTRY[args.model]
    detector = detector_cls(device=args.device)

    images = gather_images(args.images_dir, args.limit)
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running {args.model} on {len(images)} images...")
    records = []
    for image_path in images:
        start = time.perf_counter()
        detections = detector.predict(image_path, labels)
        latency = time.perf_counter() - start
        records.append(
            {
                "image": str(image_path.relative_to(args.images_dir)),
                "detections": as_serializable(detections),
                "latency_sec": latency,
            }
        )
        print(f"[done] {image_path.name} ({latency:.3f}s, {len(detections)} detections)")

    payload = {
        "model": args.model,
        "labels": labels,
        "images_dir": str(args.images_dir),
        "num_images": len(images),
        "results": records,
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
