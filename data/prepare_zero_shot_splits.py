# Copyright (c) 2024.
# SPDX-License-Identifier: MIT
"""
Create zero-shot evaluation splits for COCO and LVIS annotations.

Example:
    python data/prepare_zero_shot_splits.py \
        --dataset coco2017 \
        --annotations data/raw/coco2017/annotations/instances_val2017.json
"""

from __future__ import annotations

import argparse
import json
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, Iterable, List, Set

import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SPLIT_CONFIG = ROOT_DIR / "data" / "configs" / "zero_shot_splits.yaml"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "processed"

_ALIAS_SOURCE = {
    "microwave": ["microwave oven"],
    "tv": ["television", "television set"],
    "couch": ["sofa", "settee"],
    "sofa": ["couch"],
    "laptop": ["laptop computer", "notebook computer"],
    "wine glass": ["wineglass"],
    "tennis racket": ["tennis racquet"],
    "baseball bat": ["baseball bat (sports equipment)"],
    "stapler": ["stapler (fastener)", "stapler_(stapling_machine)"],
}


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


_ALIAS_MAP = {
    _normalize(key): [_normalize(value) for value in values]
    for key, values in _ALIAS_SOURCE.items()
}


def load_split_config(config_path: Path) -> Dict[str, Dict[str, List[str]]]:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _select_category_ids(
    categories: Iterable[Dict],
    names: Iterable[str],
    *,
    strict: bool = True,
) -> Set[int]:
    normalized_to_categories: Dict[str, List[Dict]] = {}
    for cat in categories:
        normalized_to_categories.setdefault(_normalize(cat["name"]), []).append(cat)

    selected: Set[int] = set()
    missing: List[str] = []

    for name in names:
        normalized = _normalize(name)
        matches = normalized_to_categories.get(normalized)
        if not matches:
            alias_candidates = _ALIAS_MAP.get(normalized, [])
            for alias_norm in alias_candidates:
                if alias_norm in normalized_to_categories:
                    matches = normalized_to_categories[alias_norm]
                    break

        if matches:
            selected.update(cat["id"] for cat in matches)
            continue

        suggestions = get_close_matches(normalized, normalized_to_categories.keys(), n=1, cutoff=0.85)
        if suggestions:
            suggestion_name = normalized_to_categories[suggestions[0]][0]["name"]
            if not strict:
                print(f"[info] Matched '{name}' to dataset category '{suggestion_name}'.")
                selected.update(cat["id"] for cat in normalized_to_categories[suggestions[0]])
                continue
            missing.append(f"{name} (closest match: '{suggestion_name}')")
        else:
            missing.append(name)

    if missing:
        raise ValueError(f"Categories not present in annotations: {missing}")

    return selected


def _filter_annotations(
    annotations: Dict,
    category_ids: Set[int],
) -> Dict:
    filtered_annotations = [ann for ann in annotations["annotations"] if ann["category_id"] in category_ids]
    image_ids = {ann["image_id"] for ann in filtered_annotations}

    filtered_images = [img for img in annotations["images"] if img["id"] in image_ids]
    filtered_categories = [cat for cat in annotations["categories"] if cat["id"] in category_ids]

    return {
        key: value
        for key, value in annotations.items()
        if key not in {"annotations", "images", "categories"}
    } | {
        "annotations": filtered_annotations,
        "images": filtered_images,
        "categories": filtered_categories,
    }


def prepare_split(
    annotations_path: Path,
    split_name: str,
    categories: Iterable[str],
    output_dir: Path,
    *,
    strict: bool = True,
) -> Dict[str, int]:
    with annotations_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    category_ids = _select_category_ids(data["categories"], categories, strict=strict)
    filtered = _filter_annotations(data, category_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{annotations_path.stem}_{split_name}.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(filtered, fh)

    stats = {
        "categories": len(filtered["categories"]),
        "annotations": len(filtered["annotations"]),
        "images": len(filtered["images"]),
        "output": str(output_path.relative_to(ROOT_DIR)),
    }
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare zero-shot evaluation splits.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["coco2017", "lvis_v1"],
        help="Dataset name whose split rules will be used.",
    )
    parser.add_argument(
        "--annotations",
        required=True,
        type=Path,
        help="Path to the COCO/LVIS-format annotations file (JSON).",
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=DEFAULT_SPLIT_CONFIG,
        help="YAML file that defines seen/unseen category lists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where processed annotations will be written.",
    )
    parser.add_argument(
        "--subset",
        default="val",
        help="Optional subset identifier used to namespace the output files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error if a requested category is not found (default: auto-match closest names).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_rules = load_split_config(args.split_config)
    dataset_rules = split_rules.get(args.dataset)
    if not dataset_rules:
        raise ValueError(f"No split rules found for dataset '{args.dataset}'.")

    annotations_path = args.annotations.expanduser().resolve()
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    target_dir = args.output_dir / args.dataset / args.subset
    stats = {}
    for split_name, categories in dataset_rules.items():
        split_stats = prepare_split(
            annotations_path,
            split_name,
            categories,
            target_dir,
            strict=args.strict,
        )
        stats[split_name] = split_stats
        print(f"[write] {split_name}: {split_stats['output']} ({split_stats['images']} images)")

    metadata_path = target_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "dataset": args.dataset,
                "annotations_source": str(annotations_path.relative_to(ROOT_DIR)),
                "splits": stats,
            },
            fh,
            indent=2,
        )
    print(f"[write] metadata: {metadata_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
