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
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from pycocotools.coco import COCO  # type: ignore
from pycocotools.cocoeval import COCOeval  # type: ignore
import contextlib
import io

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
    parser.add_argument(
        "--table",
        action="store_true",
        help="Print a human-readable table to stderr in addition to JSON output.",
    )
    parser.add_argument(
        "--eval-labels-file",
        type=Path,
        help=(
            "Optional dynamic labels JSON with labels_per_image. When provided, "
            "evaluation is restricted to those image IDs and label categories."
        ),
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
        self.image_stems: set[str] = set()


class EvalScope:
    def __init__(
        self,
        image_ids: Set[int],
        cat_ids: Set[int],
        positive_cat_ids_by_image: Dict[int, Set[int]],
        missing_labels: Set[str],
    ):
        self.image_ids = image_ids
        self.cat_ids = cat_ids
        self.positive_cat_ids_by_image = positive_cat_ids_by_image
        self.missing_labels = missing_labels


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
                predictions.image_stems.add(image_id)

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


def label_lookup_key(label: str) -> str:
    return label.strip().lower()


def category_id_for_label(label: str, cat_name_to_id: Dict[str, int]) -> Optional[int]:
    key = label_lookup_key(label)
    if key in cat_name_to_id:
        return cat_name_to_id[key]
    alt_space = key.replace("_", " ")
    if alt_space in cat_name_to_id:
        return cat_name_to_id[alt_space]
    alt_underscore = key.replace(" ", "_")
    return cat_name_to_id.get(alt_underscore)


def load_eval_scope(
    labels_file: Optional[Path],
    dataset: str,
    coco: COCO,
    cat_name_to_id: Dict[str, int],
) -> Optional[EvalScope]:
    if labels_file is None:
        return None
    data = json.loads(labels_file.read_text(encoding="utf-8"))
    labels_per_image = data.get("labels_per_image") if isinstance(data, dict) else None
    if not isinstance(labels_per_image, dict):
        raise ValueError(f"Expected labels_per_image object in {labels_file}")

    image_ids: Set[int] = set()
    cat_ids: Set[int] = set()
    positive_cat_ids_by_image: Dict[int, Set[int]] = defaultdict(set)
    missing_labels: Set[str] = set()

    for image_name, payload in labels_per_image.items():
        image_id = stem_to_image_id(Path(str(image_name)).stem, dataset, coco)
        if image_id is None:
            continue
        image_ids.add(int(image_id))

        if isinstance(payload, dict):
            labels = payload.get("labels") or []
            positives = payload.get("positives") or []
        elif isinstance(payload, list):
            labels = payload
            positives = payload
        else:
            labels = []
            positives = []

        for label in labels:
            if not isinstance(label, str):
                continue
            cat_id = category_id_for_label(label, cat_name_to_id)
            if cat_id is None:
                missing_labels.add(label)
                continue
            cat_ids.add(int(cat_id))

        for label in positives:
            if not isinstance(label, str):
                continue
            cat_id = category_id_for_label(label, cat_name_to_id)
            if cat_id is None:
                missing_labels.add(label)
                continue
            positive_cat_ids_by_image[int(image_id)].add(int(cat_id))

    return EvalScope(
        image_ids=image_ids,
        cat_ids=cat_ids,
        positive_cat_ids_by_image=dict(positive_cat_ids_by_image),
        missing_labels=missing_labels,
    )


def prediction_image_ids(predictions: ModelPredictions, coco: COCO) -> Set[int]:
    image_ids: Set[int] = set()
    for stem in predictions.image_stems:
        image_id = stem_to_image_id(stem, predictions.dataset, coco)
        if image_id is not None:
            image_ids.add(int(image_id))
    return image_ids


def prepare_coco_detections(
    predictions: ModelPredictions,
    coco: COCO,
    cat_name_to_id: Dict[str, int],
    allowed_image_ids: Optional[Set[int]] = None,
    allowed_cat_ids: Optional[Set[int]] = None,
) -> List[Dict]:
    detections = []
    for entry in predictions.detections:
        det = entry["raw"]
        label = det.get("label") or det.get("class") or det.get("category")
        if not isinstance(label, str):
            continue
        cat_id = category_id_for_label(label, cat_name_to_id)
        if cat_id is None:
            predictions.failed_labels.add(label)
            continue
        if allowed_cat_ids is not None and int(cat_id) not in allowed_cat_ids:
            continue
        bbox = extract_boxes(det)
        if bbox is None:
            continue
        stem = entry["image_stem"]
        image_id = stem_to_image_id(stem, predictions.dataset, coco)
        if image_id is None:
            continue
        if allowed_image_ids is not None and int(image_id) not in allowed_image_ids:
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


def compute_label_accuracy(
    coco: COCO,
    detections: List[Dict],
    image_ids: Optional[Set[int]] = None,
    positive_cat_ids_by_image: Optional[Dict[int, Set[int]]] = None,
) -> Optional[float]:
    """Compute per-category binary accuracy: did we predict each ground-truth class at least once?"""

    preds_by_image: Dict[int, Set[int]] = defaultdict(set)
    for det in detections:
        image_id = int(det["image_id"])
        preds_by_image[image_id].add(int(det["category_id"]))

    if positive_cat_ids_by_image is not None:
        total = 0
        correct = 0
        scope_image_ids = image_ids or set(positive_cat_ids_by_image)
        for image_id in scope_image_ids:
            gt_categories = positive_cat_ids_by_image.get(image_id, set())
            if not gt_categories:
                continue
            total += len(gt_categories)
            predicted = preds_by_image.get(image_id, set())
            correct += sum(1 for cat in gt_categories if cat in predicted)
        if total == 0:
            return None
        return correct / total

    images_considered: Set[int] = image_ids or set(preds_by_image)
    if not images_considered:
        return 0.0

    total = 0
    correct = 0
    for image_id in images_considered:
        anns = coco.imgToAnns.get(image_id, [])
        gt_categories = {ann["category_id"] for ann in anns}
        if not gt_categories:
            continue
        total += len(gt_categories)
        predicted = preds_by_image.get(image_id, set())
        correct += sum(1 for cat in gt_categories if cat in predicted)

    if total == 0:
        return None
    return correct / total


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
    image_ids: Optional[Set[int]] = None,
    cat_ids: Optional[Set[int]] = None,
) -> Tuple[Optional[float], Optional[float]]:
    if not detections:
        return 0.0, 0.0
    coco_dt = coco.loadRes(detections)
    evaluator = COCOeval(coco, coco_dt, "bbox")
    if image_ids is not None:
        evaluator.params.imgIds = sorted(image_ids)
    if cat_ids is not None:
        evaluator.params.catIds = sorted(cat_ids)
    with contextlib.redirect_stdout(io.StringIO()):
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

    summary: Dict[str, List[Dict[str, object]]] = {}
    table_rows: List[
        Tuple[
            str,
            str,
            str,
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
        ]
    ] = []

    for dataset in sorted(all_datasets):
        if dataset not in DATASET_CONFIGS:
            continue

        annotation_path = DATASET_CONFIGS[dataset]["annotation"]
        if not annotation_path.exists():
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
        cat_name_to_id = {label_lookup_key(cat["name"]): cat["id"] for cat in coco.loadCats(coco.getCatIds())}
        eval_scope = load_eval_scope(args.eval_labels_file, dataset, coco, cat_name_to_id)

        models = {**vlm_predictions.get(dataset, {}), **traditional_predictions.get(dataset, {})}
        if not models:
            continue

        dataset_entries: List[Dict[str, object]] = []

        for model_name, preds in sorted(models.items()):
            model_image_ids = prediction_image_ids(preds, coco)
            eval_image_ids = eval_scope.image_ids if eval_scope else model_image_ids
            eval_cat_ids = eval_scope.cat_ids if eval_scope else None
            detections = prepare_coco_detections(
                preds,
                coco,
                cat_name_to_id,
                allowed_image_ids=eval_image_ids,
                allowed_cat_ids=eval_cat_ids,
            )
            ap, ap50 = evaluate_model(coco, detections, image_ids=eval_image_ids, cat_ids=eval_cat_ids)
            lat_mean, lat_med, lat_p95 = summarize_latency(preds.latencies)

            label_accuracy = compute_label_accuracy(
                coco,
                detections,
                image_ids=eval_image_ids,
                positive_cat_ids_by_image=eval_scope.positive_cat_ids_by_image if eval_scope else None,
            )

            entry = {
                "model": model_name,
                "family": preds.family,
                "eval_images": len(eval_image_ids),
                "eval_categories": len(eval_cat_ids) if eval_cat_ids is not None else None,
                "detections": len(detections),
                "failed_label_count": len(preds.failed_labels),
                "ap": ap,
                "ap50": ap50,
                "latency_mean": lat_mean,
                "latency_median": lat_med,
                "latency_p95": lat_p95,
                "label_accuracy": label_accuracy,
            }
            dataset_entries.append(entry)
            table_rows.append(
                (
                    dataset,
                    model_name,
                    preds.family,
                    ap,
                    ap50,
                    lat_mean,
                    lat_med,
                    lat_p95,
                    label_accuracy,
                )
            )

        if dataset_entries:
            summary[dataset] = dataset_entries

    print(json.dumps(summary, indent=2))

    if args.table and table_rows:
        def fmt(value: Optional[float]) -> str:
            return f"{value:.4f}" if value is not None else "  n/a  "

        import sys

        print("\nHuman-readable table:", file=sys.stderr)
        header = f"{'Dataset':15s} {'Model':30s} {'Family':10s} {'AP':>8s} {'AP@0.5':>8s} {'lat(mean)':>12s} {'lat(med)':>12s} {'lat(p95)':>12s} {'Acc.':>8s}"
        print(header, file=sys.stderr)
        print("-" * len(header), file=sys.stderr)
        for dataset, model_name, family, ap, ap50, lat_mean, lat_med, lat_p95, label_acc in table_rows:
            row = (
                f"{dataset:15s} "
                f"{model_name:30s} "
                f"{family:10s} "
                f"{fmt(ap):>8s} "
                f"{fmt(ap50):>8s} "
                f"{fmt(lat_mean):>12s} "
                f"{fmt(lat_med):>12s} "
                f"{fmt(lat_p95):>12s} "
                f"{fmt(label_acc):>8s}"
            )
            print(row, file=sys.stderr)


if __name__ == "__main__":
    main()
