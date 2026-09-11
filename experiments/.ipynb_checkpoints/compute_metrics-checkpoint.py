#!/usr/bin/env python3
"""
Compute detection metrics (AP, AP@0.5, latency statistics) for both traditional and VLM models.

The script scans the results directories:
    results/raw/vlm/*/per_image/<model>/*.json
    results/raw/traditional/*/per_image/<model>/*.json

It infers the dataset from the stored image paths (supports COCO 2017 val and LVIS v1 unseen),
evaluates detections with COCO-style metrics, and prints latency summaries.

Usage:
    python experiments/compute_metrics.py
    python experiments/compute_metrics.py --dataset coco2017-val
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from pycocotools.coco import COCO  # type: ignore
from pycocotools.cocoeval import COCOeval  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULTS = {
    "vlm": PROJECT_ROOT / "results" / "raw" / "vlm",
    "traditional": PROJECT_ROOT / "results" / "raw" / "traditional",
}

DATASET_CONFIGS = {
    "coco2017-val": {
        "annotation": PROJECT_ROOT / "data" / "raw" / "coco2017" / "annotations" / "instances_val2017.json",
    },
    "lvis-unseen": {
        "annotation": PROJECT_ROOT / "data" / "processed" / "lvis_v1" / "val" / "lvis_v1_val_unseen.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute evaluation metrics from stored per-image results.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS) + ["all"],
        default="all",
        help="Dataset to evaluate. Defaults to all available.",
    )
    parser.add_argument(
        "--vlm-dir",
        type=Path,
        default=DEFAULT_RESULTS["vlm"],
        help="Directory containing VLM results (default: results/raw/vlm).",
    )
    parser.add_argument(
        "--traditional-dir",
        type=Path,
        default=DEFAULT_RESULTS["traditional"],
        help="Directory containing traditional detector results (default: results/raw/traditional).",
    )
    return parser.parse_args()


class ModelPredictions:
    def __init__(self, family: str, dataset: str, model_name: str):
        self.family = family  # "vlm" or "traditional"
        self.dataset = dataset
        self.model_name = model_name
        self.detections: List[Dict] = []
        self.latencies: List[float] = []
        self.failed_labels: set[str] = set()


def infer_dataset_type(image_path: str) -> str:
    lower_path = image_path.lower()
    if "coco2017" in lower_path:
        return "coco2017-val"
    if "lvis" in lower_path:
        return "lvis-unseen"
    raise ValueError(f"Unable to infer dataset type from path: {image_path}")


def extract_boxes(entry: Dict[str, object]) -> Optional[List[float]]:
    box = entry.get("box_2d") or entry.get("bbox") or entry.get("box")
    if box is None:
        return None
    if isinstance(box, dict):
        # Accept legacy dict format
        coords = [box.get(k) for k in ("x1", "y1", "x2", "y2")]
    else:
        coords = list(box)
    try:
        x1, y1, x2, y2 = [float(v) for v in coords]
    except (TypeError, ValueError):
        return None
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return [x1, y1, width, height]


def load_per_image_file(path: Path) -> Tuple[Optional[List[Dict]], Optional[float], Optional[str], Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result_block = data.get("result") if isinstance(data.get("result"), dict) else None
    detections = result_block.get("detections") if result_block else data.get("detections")
    latency = result_block.get("latency_sec") if result_block else data.get("latency_sec")
    raw_response = result_block.get("raw_response") if result_block else data.get("raw_response")
    image_path = data.get("image") or data.get("Image") or ""
    return detections, latency, image_path, data


def collect_predictions(
    family: str,
    base_dir: Path,
) -> Dict[str, Dict[str, ModelPredictions]]:
    """
    Returns mapping dataset -> model_name -> ModelPredictions
    """
    grouped: Dict[str, Dict[str, ModelPredictions]] = defaultdict(dict)
    if not base_dir.exists():
        return grouped

    for dataset_dir in base_dir.iterdir():
        per_image_root = dataset_dir / "per_image"
        if not per_image_root.exists():
            continue
        for model_dir in per_image_root.iterdir():
            if not model_dir.is_dir():
                continue
            json_files = sorted(model_dir.glob("*.json"))
            if not json_files:
                continue
            first_det, _, image_path, _ = load_per_image_file(json_files[0])
            if first_det is None and not image_path:
                continue
            dataset_type = infer_dataset_type(image_path)
            model_name = model_dir.name
            predictions = grouped[dataset_type].get(model_name)
            if not predictions:
                predictions = ModelPredictions(family=family, dataset=dataset_type, model_name=model_name)
                grouped[dataset_type][model_name] = predictions

            for json_path in json_files:
                detections, latency, img_path, raw_data = load_per_image_file(json_path)
                if img_path:
                    dataset_type = infer_dataset_type(img_path)
                image_id = Path(img_path).stem if img_path else Path(raw_data.get("image", "")).stem
                if not image_id:
                    # Attempt fallback from original path format
                    image_id = Path(json_path).stem

                for det in (detections or []):
                    predictions.detections.append(
                        {
                            "image_stem": image_id,
                            "raw": det,
                        }
                    )
                if latency is not None:
                    try:
                        predictions.latencies.append(float(latency))
                    except (TypeError, ValueError):
                        pass
    return grouped


def stem_to_image_id(stem: str, dataset: str, coco: COCO) -> Optional[int]:
    if dataset == "coco2017-val":
        try:
            return int(stem)
        except ValueError:
            return None
    if dataset == "lvis-unseen":
        # LVIS image IDs are integers but file stems are 12-digit COCO ids.
        try:
            image_id = int(stem)
        except ValueError:
            return None
        # Mapping by COCO-style ID (LVIS reuses COCO ids).
        if image_id in coco.imgs:
            return image_id
        # Fallback: search by file name suffix.
        for img_id, img in coco.imgs.items():
            file_name = img.get("file_name") or img.get("coco_url", "").split("/")[-1]
            if file_name and Path(file_name).stem == stem:
                return img_id
        return None
    return None


def prepare_coco_detections(
    predictions: ModelPredictions,
    coco: COCO,
    cat_name_to_id: Dict[str, int],
) -> List[Dict]:
    detections = []
    for entry in predictions.detections:
        det = entry["raw"]
        label = det.get("label") or det.get("class") or det.get("category")
        if not isinstance(label, str):
            continue
        cat_id = cat_name_to_id.get(label.strip().lower())
        if cat_id is None:
            predictions.failed_labels.add(label)
            continue
        bbox = extract_boxes(det)
        if bbox is None:
            continue
        stem = entry["image_stem"]
        image_id = stem_to_image_id(stem, predictions.dataset, coco)
        if image_id is None:
            continue
        score = det.get("score")
        if score is None:
            # Hosted VLMs sometimes omit scores; treat detections as binary.
            score = 1.0 if predictions.family == "vlm" else 0.5
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 1.0 if predictions.family == "vlm" else 0.5
        detections.append(
            {
                "image_id": int(image_id),
                "category_id": int(cat_id),
                "bbox": bbox,
                "score": score,
            }
        )
    return detections


def summarize_latency(latencies: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not latencies:
        return None, None, None
    arr = np.array(latencies)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    return mean, median, p95


def evaluate_model(
    coco: COCO,
    detections: List[Dict],
) -> Tuple[Optional[float], Optional[float]]:
    if not detections:
        return None, None
    coco_dt = coco.loadRes(detections)
    evaluator = COCOeval(coco, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    ap = float(evaluator.stats[0]) if evaluator.stats is not None else None
    ap50 = float(evaluator.stats[1]) if evaluator.stats is not None else None
    return ap, ap50


def main() -> None:
    args = parse_args()

    vlm_predictions = collect_predictions("vlm", args.vlm_dir)
    traditional_predictions = collect_predictions("traditional", args.traditional_dir)

    all_datasets = set(vlm_predictions) | set(traditional_predictions)
    if args.dataset != "all":
        all_datasets = {ds for ds in all_datasets if ds == args.dataset}

    if not all_datasets:
        print("No datasets found with stored results. Nothing to evaluate.")
        return

    for dataset in sorted(all_datasets):
        if dataset not in DATASET_CONFIGS:
            print(f"[warn] Dataset '{dataset}' is not recognized. Skipping.")
            continue

        annotation_path = DATASET_CONFIGS[dataset]["annotation"]
        if not annotation_path.exists():
            print(f"[warn] Annotation file not found for {dataset}: {annotation_path}")
            continue

        coco = COCO(str(annotation_path))
        # Ensure all annotations expose the 'iscrowd' key (some LVIS exports omit it).
        updated = False
        for ann in coco.dataset.get("annotations", []):
            if "iscrowd" not in ann:
                ann["iscrowd"] = 0
                updated = True
        if updated:
            coco.createIndex()
        cat_name_to_id = {cat["name"].strip().lower(): cat["id"] for cat in coco.loadCats(coco.getCatIds())}

        models = {**vlm_predictions.get(dataset, {}), **traditional_predictions.get(dataset, {})}
        if not models:
            print(f"\nDataset: {dataset}\n  No models with stored predictions.")
            continue

        print(f"\nDataset: {dataset}")
        header = f"{'Model':30s} {'Family':10s} {'AP':>8s} {'AP@0.5':>8s} {'lat(mean)':>12s} {'lat(med)':>12s} {'lat(p95)':>12s}"
        print(header)
        print("-" * len(header))

        for model_name, preds in sorted(models.items()):
            detections = prepare_coco_detections(preds, coco, cat_name_to_id)
            ap, ap50 = evaluate_model(coco, detections)
            lat_mean, lat_med, lat_p95 = summarize_latency(preds.latencies)

            def fmt(value: Optional[float]) -> str:
                return f"{value:.4f}" if value is not None else "  n/a  "

            row = f"{model_name:30s} {preds.family:10s} {fmt(ap):>8s} {fmt(ap50):>8s} {fmt(lat_mean):>12s} {fmt(lat_med):>12s} {fmt(lat_p95):>12s}"
            print(row)

            if preds.failed_labels:
                missing = ", ".join(sorted(preds.failed_labels))
                print(f"  [warn] {model_name}: {len(preds.failed_labels)} unmapped labels -> {missing}")


if __name__ == "__main__":
    main()
