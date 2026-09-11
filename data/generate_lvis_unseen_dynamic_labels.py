#!/usr/bin/env python3
"""
Generate per-image label sets for the LVIS unseen validation subset.

For each image, we take the annotated categories and extend them with a
random 50% sample of the remaining unseen categories. The sampling is
deterministic per image to ensure reproducibility, and the resulting
label lists are persisted to disk for later reuse.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_METADATA = Path("data/processed/lvis_v1/val/lvis_v1_val_unseen.json")
DEFAULT_OUTPUT = Path("data/processed/lvis_v1/val/lvis_unseen_dynamic_labels.json")
DEFAULT_CLASSES_OUTPUT = Path("data/processed/lvis_v1/val/lvis_unseen_classes.txt")
DEFAULT_SAMPLE_FRACTION = 0.5
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-image label sets for LVIS unseen validation."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Path to LVIS metadata JSON (unseen split).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON file storing per-image labels.",
    )
    parser.add_argument(
        "--classes-output",
        type=Path,
        default=DEFAULT_CLASSES_OUTPUT,
        help="Optional text file listing all classes present in the subset.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=DEFAULT_SAMPLE_FRACTION,
        help="Fraction of remaining classes to sample as distractors per image (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Global seed used to derive per-image random states.",
    )
    return parser.parse_args()


def load_metadata(metadata_path: Path) -> Dict:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_category_mappings(metadata: Dict) -> Tuple[Dict[int, str], List[str]]:
    categories = metadata.get("categories", [])
    id_to_name = {category["id"]: category["name"] for category in categories}
    all_names = sorted(set(id_to_name.values()))
    return id_to_name, all_names


def collect_positive_labels(metadata: Dict, id_to_name: Dict[int, str]) -> Dict[int, List[str]]:
    annotations = metadata.get("annotations", [])
    positives: Dict[int, set[str]] = {}
    for ann in annotations:
        image_id = ann["image_id"]
        cat_id = ann["category_id"]
        label = id_to_name.get(cat_id)
        if not label:
            continue
        positives.setdefault(image_id, set()).add(label)
    return {image_id: sorted(labels) for image_id, labels in positives.items()}


def derive_filename(image_entry: Dict) -> str:
    if "file_name" in image_entry and image_entry["file_name"]:
        return image_entry["file_name"]
    if "coco_url" in image_entry and image_entry["coco_url"]:
        return Path(image_entry["coco_url"]).name
    if "flickr_url" in image_entry and image_entry["flickr_url"]:
        return Path(image_entry["flickr_url"]).name
    raise ValueError(f"Unable to derive filename for image entry: {image_entry}")


def sample_negatives(
    all_classes: List[str],
    positives: List[str],
    sample_fraction: float,
    rng: random.Random,
) -> List[str]:
    positive_set = set(positives)
    candidates = [label for label in all_classes if label not in positive_set]
    if not candidates or sample_fraction <= 0:
        return []
    sample_size = max(0, min(len(candidates), int(round(len(candidates) * sample_fraction))))
    if sample_size == 0:
        return []
    return sorted(rng.sample(candidates, sample_size))


def build_per_image_labels(
    metadata: Dict,
    sample_fraction: float,
    seed: int,
) -> Tuple[Dict[str, Dict[str, List[str]]], List[str]]:
    id_to_name, all_classes = build_category_mappings(metadata)
    positive_by_image = collect_positive_labels(metadata, id_to_name)
    images = metadata.get("images", [])

    per_image: Dict[str, Dict[str, List[str]]] = {}
    for image_entry in images:
        image_id = image_entry["id"]
        filename = derive_filename(image_entry)
        positives = positive_by_image.get(image_id, [])

        rng = random.Random((seed << 32) + image_id)
        negatives = sample_negatives(all_classes, positives, sample_fraction, rng)

        combined = list(dict.fromkeys(positives + negatives))
        per_image[filename] = {
            "positives": positives,
            "negatives": negatives,
            "labels": combined,
        }
    return per_image, all_classes


def save_classes(classes: Iterable[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for label in classes:
            handle.write(f"{label}\n")


def save_mapping(metadata: Dict, per_image: Dict[str, Dict[str, List[str]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(metadata.get("info", {}).get("description", "")),
        "num_images": len(per_image),
        "labels_per_image": per_image,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)

    per_image_labels, all_classes = build_per_image_labels(
        metadata=metadata,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )

    save_mapping(metadata, per_image_labels, args.output)
    if args.classes_output:
        save_classes(all_classes, args.classes_output)

    print(f"Total classes in LVIS unseen: {len(all_classes)}")
    print(f"Per-image label mapping saved to: {args.output}")
    if args.classes_output:
        print(f"Class list saved to: {args.classes_output}")


if __name__ == "__main__":
    main()
