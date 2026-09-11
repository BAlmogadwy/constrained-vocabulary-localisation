#!/usr/bin/env python3
"""GT Oracle (Oracle-Box) runner that classifies LVIS ground-truth crops."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.env import load_environment  # noqa: E402

load_environment()

from models.vlm import MODEL_REGISTRY  # noqa: E402


DEFAULT_PROMPT_TEMPLATE = (
    "You are an object recognition assistant.\n\n"
    "You will see a cropped image that contains exactly one object from the following "
    "LVIS categories (unseen split):\n\n{class_list}\n\n"
    "Your task:\n"
    "- Return ONLY the best matching category name from this list.\n"
    "- Do not output explanations or extra text.\n\n"
    'Answer with a single string: "class_name".'
)

ARTICLES = {"a", "an", "the"}


@dataclass
class ImageRecord:
    image_id: int
    file_name: str
    path: Path
    width: int
    height: int
    labels: List[str]
    annotations: List[Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Oracle-Box LVIS experiment.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run only two images (smoke test).",
    )
    return parser.parse_args()


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def derive_filename(image_entry: Dict[str, Any]) -> str:
    if image_entry.get("file_name"):
        return image_entry["file_name"]
    if image_entry.get("coco_url"):
        return Path(image_entry["coco_url"]).name
    if image_entry.get("flickr_url"):
        return Path(image_entry["flickr_url"]).name
    raise ValueError(f"Unable to derive filename for LVIS image entry: {image_entry}")


def load_per_image_labels(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = load_json(path)
    labels_per_image = payload.get("labels_per_image")
    if not isinstance(labels_per_image, dict):
        raise ValueError("labels_per_image must be a mapping from filename to labels.")
    return labels_per_image


def load_optional_synonyms(path: Optional[Path]) -> Dict[str, str]:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Synonyms file not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    else:
        data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("Synonyms file must contain a mapping of alias -> canonical label.")
    normalized: Dict[str, str] = {}
    for alias, canonical in data.items():
        if not isinstance(alias, str) or not isinstance(canonical, str):
            continue
        normalized[alias] = canonical
    return normalized


def build_category_maps(metadata: Dict[str, Any]) -> Tuple[Dict[int, str], Dict[str, str]]:
    categories = metadata.get("categories", [])
    id_to_label: Dict[int, str] = {}
    synonym_lookup: Dict[str, str] = {}
    for category in categories:
        if not isinstance(category, dict):
            continue
        cat_id = category.get("id")
        name = str(category.get("name") or "").strip()
        if not name or cat_id is None:
            continue
        id_to_label[cat_id] = name
        normalized_name = normalize_label(name)
        if normalized_name:
            synonym_lookup.setdefault(normalized_name, name)
        for alias in category.get("synonyms") or []:
            if not isinstance(alias, str):
                continue
            alias_norm = normalize_label(alias)
            if alias_norm:
                synonym_lookup.setdefault(alias_norm, name)
    return id_to_label, synonym_lookup


def merge_synonym_tables(base: Dict[str, str], extra: Dict[str, str]) -> Dict[str, str]:
    merged = dict(base)
    for alias, canonical in extra.items():
        alias_norm = normalize_label(alias)
        if not alias_norm:
            continue
        merged[alias_norm] = canonical
    return merged


def normalize_label(label: str | None) -> str:
    if not label:
        return ""
    lowered = label.strip().lower()
    lowered = lowered.replace("-", " ").replace("_", " ")
    lowered = re.sub(r"[^a-z0-9\s\(\)]", " ", lowered)
    tokens = lowered.split()
    if tokens and tokens[0] in ARTICLES:
        tokens = tokens[1:]
    return " ".join(tokens)


def build_candidate_maps(labels: Iterable[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    normalized: Dict[str, str] = {}
    squeezed: Dict[str, str] = {}
    for label in labels:
        if not isinstance(label, str):
            continue
        norm = normalize_label(label)
        if norm and norm not in normalized:
            normalized[norm] = label
        squeezed_key = norm.replace(" ", "")
        if squeezed_key and squeezed_key not in squeezed:
            squeezed[squeezed_key] = label
    return normalized, squeezed


def format_class_list(labels: List[str]) -> str:
    return "\n".join(f"- {label}" for label in labels)


def crop_region(image: Image.Image, bbox: List[float]) -> Image.Image:
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox [x, y, w, h], received: {bbox}")
    x, y, w, h = bbox
    x1 = max(0, math.floor(x))
    y1 = max(0, math.floor(y))
    x2 = min(image.width, math.ceil(x + w))
    y2 = min(image.height, math.ceil(y + h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop with coordinates {(x1, y1, x2, y2)} from bbox {bbox}")
    return image.crop((x1, y1, x2, y2)).copy()


def slugify_model_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "model"


def _extract_label_from_detections(detections: Any) -> Optional[str]:
    if not isinstance(detections, list):
        return None
    for entry in detections:
        if not isinstance(entry, dict):
            continue
        for key in ("label", "class", "name"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def match_prediction(
    raw_prediction: str | None,
    candidate_map: Dict[str, str],
    candidate_squeezed: Dict[str, str],
    synonym_lookup: Dict[str, str],
) -> Tuple[Optional[str], str]:
    if raw_prediction is None:
        return None, ""
    normalized_pred = normalize_label(raw_prediction)
    if not normalized_pred:
        return None, ""
    match = candidate_map.get(normalized_pred)
    if match:
        return match, normalized_pred
    squeezed = normalized_pred.replace(" ", "")
    match = candidate_squeezed.get(squeezed)
    if match:
        return match, normalized_pred
    alias_target = synonym_lookup.get(normalized_pred)
    if alias_target:
        canonical_norm = normalize_label(alias_target)
        match = candidate_map.get(canonical_norm)
        if match:
            return match, normalized_pred
    return None, normalized_pred


def build_image_records(
    metadata: Dict[str, Any],
    images_dir: Path,
    per_image_labels: Dict[str, Dict[str, Any]],
    limit: Optional[int],
) -> List[ImageRecord]:
    annotations = metadata.get("annotations", [])
    annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        image_id = ann.get("image_id")
        if image_id is None:
            continue
        annotations_by_image.setdefault(image_id, []).append(ann)

    images: List[ImageRecord] = []
    for entry in metadata.get("images", []):
        if not isinstance(entry, dict):
            continue
        image_id = entry.get("id")
        if image_id is None:
            continue
        file_name = derive_filename(entry)
        if file_name not in per_image_labels:
            continue
        label_values = per_image_labels[file_name].get("labels") or []
        if not isinstance(label_values, list):
            raise ValueError(f"Per-image labels for {file_name} must be a list.")
        labels = list(
            dict.fromkeys(
                str(label).strip() for label in label_values if str(label).strip()
            )
        )
        if not labels:
            raise ValueError(f"No candidate labels available for {file_name}.")
        image_path = (images_dir / file_name).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        record = ImageRecord(
            image_id=image_id,
            file_name=file_name,
            path=image_path,
            width=int(entry.get("width") or 0),
            height=int(entry.get("height") or 0),
            labels=labels,
            annotations=annotations_by_image.get(image_id, []),
        )
        if not record.annotations:
            continue
        images.append(record)
        if limit and len(images) >= limit:
            break
    images.sort(key=lambda rec: rec.file_name)
    return images


def load_classifier(model_name: str):
    detector_cls = MODEL_REGISTRY.get(model_name)
    if not detector_cls:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return detector_cls()


def invoke_classifier(
    detector: Any,
    crop_image: Image.Image,
    labels: List[str],
    prompt: str,
    metadata: Dict[str, Any],
    strategy: str,
) -> Tuple[Optional[str], Any]:
    from inspect import signature, Parameter

    kwargs: Dict[str, Any] = {"image": crop_image, "labels": labels, "prompt": prompt}
    if metadata:
        kwargs["metadata"] = metadata
    crop_path = metadata.get("crop_path") if isinstance(metadata, dict) else None
    if crop_path:
        kwargs["image_path"] = crop_path

    def call_with_filtered_kwargs(func):
        sig = signature(func)
        accepts_var_kw = any(param.kind == Parameter.VAR_KEYWORD for param in sig.parameters.values())
        allowed = set(sig.parameters)
        call_kwargs = {}
        for key, value in kwargs.items():
            if key in allowed or accepts_var_kw:
                call_kwargs[key] = value
        return func(**call_kwargs)

    if hasattr(detector, "classify_crop"):
        response = call_with_filtered_kwargs(detector.classify_crop)
    elif hasattr(detector, "classify"):
        response = call_with_filtered_kwargs(detector.classify)
    elif crop_path and hasattr(detector, "predict"):
        prediction = detector.predict(crop_path, labels, strategy=strategy)
        if isinstance(prediction, dict):
            detections = prediction.get("detections", [])
            raw_response = prediction.get("raw_response", prediction)
        else:
            detections = prediction
            raw_response = prediction
        label = _extract_label_from_detections(detections)
        return label, raw_response
    else:
        raise AttributeError(
            f"Model '{type(detector).__name__}' does not implement a crop classification interface."
        )

    if isinstance(response, dict):
        for key in ("prediction", "label", "class_name"):
            if key in response and isinstance(response[key], str):
                return response[key], response
        if "raw_response" in response:
            raw = response["raw_response"]
        else:
            raw = response
        return None, raw
    if response is None:
        return None, None
    if isinstance(response, str):
        return response, response
    return str(response), response


def run_model(
    model_config: Dict[str, Any],
    images: List[ImageRecord],
    id_to_label: Dict[int, str],
    synonym_lookup: Dict[str, str],
    per_image_labels_path: Path,
    metadata_path: Path,
    images_dir: Path,
    prompt_template: str,
    output_dir: Path,
    force: bool,
    strategy: str,
) -> None:
    model_name = model_config["name"]
    prompt_override = model_config.get("prompt_template")
    prompt_template = prompt_override or prompt_template

    detector = load_classifier(model_name)
    slug = slugify_model_name(model_name)
    model_output_dir = output_dir / f"lvis_{slug}"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    per_image_dir = model_output_dir / "per_image"
    per_image_dir.mkdir(parents=True, exist_ok=True)

    output_name = model_config.get("output_name") or f"oracle_box_{slug}.json"
    output_path = model_output_dir / output_name

    model_start = time.perf_counter()
    summary_results: List[Dict[str, Any]] = []
    total_crops = 0
    correct_crops = 0
    total_latency = 0.0

    print(f"Running Oracle-Box for model '{model_name}' on {len(images)} images.")
    with tempfile.TemporaryDirectory(prefix="oracle_box_crops_") as crops_tmp:
        crops_tmp_path = Path(crops_tmp)
        for record in images:
            per_image_path = per_image_dir / f"{Path(record.file_name).stem}.json"
            cached_payload: Dict[str, Any] = {}
            if per_image_path.exists() and not force:
                with per_image_path.open("r", encoding="utf-8") as handle:
                    cached_payload = json.load(handle)

            existing_crops: Dict[int, Dict[str, Any]] = {}
            for crop_entry in cached_payload.get("crops", []) or []:
                ann_id = crop_entry.get("annotation_id")
                if not isinstance(ann_id, int):
                    continue
                if crop_entry.get("error"):
                    continue
                prediction_value = crop_entry.get("prediction")
                if not isinstance(prediction_value, str) or not prediction_value.strip():
                    continue
                existing_crops[ann_id] = crop_entry

            formatted_class_list = format_class_list(record.labels)
            per_image_prompt = prompt_template.replace("{class_list}", formatted_class_list)
            candidate_map, candidate_squeezed = build_candidate_maps(record.labels)

            image_results: List[Dict[str, Any]] = []
            new_crops = 0
            reused_crops = 0
            with Image.open(record.path) as image:
                image = image.convert("RGB")
                for annotation in record.annotations:
                    annotation_id_value = annotation.get("id")
                    if annotation_id_value is None:
                        print(f"[warn] Skipping annotation without ID in {record.file_name}.")
                        continue
                    annotation_id = int(annotation_id_value)
                    bbox = annotation.get("bbox") or []
                    category_id = annotation.get("category_id")
                    gt_label = id_to_label.get(category_id)
                    if gt_label is None:
                        print(
                            f"[warn] Skipping annotation {annotation_id} in {record.file_name}: unknown category {category_id}."
                        )
                        continue

                    if not force and annotation_id in existing_crops:
                        image_results.append(existing_crops[annotation_id])
                        reused_crops += 1
                        continue

                    crop_result: Dict[str, Any] = {
                        "annotation_id": annotation_id,
                        "bbox": bbox,
                        "category_id": category_id,
                        "gt_label": gt_label,
                        "labels": record.labels,
                    }

                    try:
                        crop_image = crop_region(image, bbox)
                    except Exception as exc:  # noqa: BLE001
                        crop_result.update(
                            {
                                "error": f"crop_failed: {exc}",
                                "prediction": None,
                                "normalized_prediction": "",
                                "matched_label": None,
                                "correct": False,
                                "latency_sec": 0.0,
                            }
                        )
                        image_results.append(crop_result)
                        new_crops += 1
                        continue

                    crop_filename = f"{Path(record.file_name).stem}_{annotation_id}.jpg"
                    crop_path = crops_tmp_path / crop_filename
                    try:
                        crop_image.save(crop_path, format="JPEG")
                    except Exception as exc:  # noqa: BLE001
                        crop_result.update(
                            {
                                "error": f"save_failed: {exc}",
                                "prediction": None,
                                "normalized_prediction": "",
                                "matched_label": None,
                                "correct": False,
                                "latency_sec": 0.0,
                            }
                        )
                        image_results.append(crop_result)
                        new_crops += 1
                        continue

                    crop_metadata = {
                        "image_id": record.image_id,
                        "annotation_id": annotation_id,
                        "file_name": record.file_name,
                        "bbox": bbox,
                        "crop_path": str(crop_path),
                    }

                    start = time.perf_counter()
                    try:
                        prediction, raw_response = invoke_classifier(
                            detector=detector,
                            crop_image=crop_image,
                            labels=record.labels,
                            prompt=per_image_prompt,
                            metadata=crop_metadata,
                            strategy=strategy,
                        )
                    except Exception as exc:  # noqa: BLE001
                        latency = time.perf_counter() - start
                        crop_result.update(
                            {
                                "error": f"inference_failed: {exc}",
                                "prediction": None,
                                "normalized_prediction": "",
                                "matched_label": None,
                                "correct": False,
                                "latency_sec": latency,
                                "raw_response": None,
                            }
                        )
                        image_results.append(crop_result)
                        new_crops += 1
                        continue

                    latency = time.perf_counter() - start
                    matched_label, normalized_prediction = match_prediction(
                        prediction,
                        candidate_map,
                        candidate_squeezed,
                        synonym_lookup,
                    )
                    is_correct = bool(matched_label and matched_label == gt_label)
                    crop_result.update(
                        {
                            "prediction": prediction,
                            "normalized_prediction": normalized_prediction,
                            "matched_label": matched_label,
                            "correct": is_correct,
                            "latency_sec": latency,
                            "raw_response": raw_response,
                        }
                    )
                    image_results.append(crop_result)
                    new_crops += 1

            image_results.sort(key=lambda entry: entry.get("annotation_id", 0))
            total_crops += len(image_results)
            correct_crops += sum(1 for entry in image_results if entry.get("correct"))
            total_latency += sum(float(entry.get("latency_sec") or 0.0) for entry in image_results)
            payload = {
                "image": str(record.path),
                "image_id": record.image_id,
                "file_name": record.file_name,
                "num_crops": len(image_results),
                "crops": image_results,
                "prompt": per_image_prompt,
            }
            with per_image_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            summary_results.append(payload)
            print(
                f"[model:{model_name}] {record.file_name}: {len(image_results)} crops "
                f"({new_crops} new, {reused_crops} reused)."
            )

    wall_time = time.perf_counter() - model_start
    accuracy = (correct_crops / total_crops) if total_crops else 0.0
    avg_latency = (total_latency / total_crops) if total_crops else 0.0
    output_payload = {
        "experiment": "oracle-box",
        "model": model_name,
        "num_images": len(summary_results),
        "num_crops": total_crops,
        "num_correct": correct_crops,
        "accuracy": accuracy,
        "avg_latency_sec": avg_latency,
        "summary": {
            "wall_time_sec": wall_time,
            "total_latency_sec": total_latency,
        },
        "dataset": {
            "images_dir": str(images_dir),
            "metadata_file": str(metadata_path),
            "per_image_labels_file": str(per_image_labels_path),
        },
        "results": summary_results,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)
    print(f"Results saved to {output_path}")


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    dataset_cfg = config.get("dataset") or {}
    images_dir_value = dataset_cfg.get("images_dir")
    metadata_value = dataset_cfg.get("metadata_file")
    if not images_dir_value or not metadata_value:
        raise ValueError("dataset.images_dir and dataset.metadata_file must be specified in the config.")
    images_dir = resolve_path(images_dir_value)
    metadata_path = resolve_path(metadata_value)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    limit = dataset_cfg.get("limit")
    if args.test:
        limit = 2

    labels_cfg = config.get("labels") or {}
    per_image_value = labels_cfg.get("per_image_file")
    if not per_image_value:
        raise ValueError("labels.per_image_file must be specified in the config.")
    per_image_labels_path = resolve_path(per_image_value)
    synonyms_file = labels_cfg.get("synonyms_file")
    synonyms_path = resolve_path(synonyms_file) if synonyms_file else None

    metadata = load_json(metadata_path)
    per_image_labels = load_per_image_labels(per_image_labels_path)
    id_to_label, metadata_synonyms = build_category_maps(metadata)
    extra_synonyms = load_optional_synonyms(synonyms_path)
    synonym_lookup = merge_synonym_tables(metadata_synonyms, extra_synonyms)
    images = build_image_records(metadata, images_dir, per_image_labels, limit)
    if not images:
        raise RuntimeError("No LVIS images matched the provided configuration.")

    output_cfg = config.get("output") or {}
    output_dir_value = output_cfg.get("dir")
    if not output_dir_value:
        raise ValueError("output.dir must be specified in the config.")
    output_dir = resolve_path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_cfg = config.get("prompt") or {}
    prompt_template = prompt_cfg.get("template") or DEFAULT_PROMPT_TEMPLATE

    force = bool(config.get("force"))
    models = config.get("models") or []
    if not models:
        raise ValueError("Config does not contain any models.")

    for model_config in models:
        strategy = model_config.get("strategy", "single_query")
        run_model(
            model_config=model_config,
            images=images,
            id_to_label=id_to_label,
            synonym_lookup=synonym_lookup,
            per_image_labels_path=per_image_labels_path,
            metadata_path=metadata_path,
            images_dir=images_dir,
            prompt_template=prompt_template,
            output_dir=output_dir,
            force=force,
            strategy=strategy,
        )


if __name__ == "__main__":
    main()
