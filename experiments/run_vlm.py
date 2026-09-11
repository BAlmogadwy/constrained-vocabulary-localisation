"""
Batch runner that executes VLM detectors based on a YAML config.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.env import load_environment  # noqa: E402

load_environment()

from models.vlm import MODEL_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLM detectors from a YAML config.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run on only two images per dataset (smoke test).",
    )
    return parser.parse_args()


def load_yaml_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_labels(config: Dict) -> Tuple[List[str], Dict[str, List[str]], Optional[str]]:
    labels_section = config.get("labels", {})
    labels: List[str] = list(labels_section.get("values") or [])
    label_file = labels_section.get("file")
    if label_file:
        label_path = (PROJECT_ROOT / label_file).expanduser().resolve()
        if not label_path.exists():
            raise FileNotFoundError(f"Labels file not found: {label_path}")
        with label_path.open("r", encoding="utf-8") as fh:
            labels.extend([line.strip() for line in fh if line.strip()])
    labels = [label.strip() for label in labels if label.strip()]
    labels = list(dict.fromkeys(labels))

    per_image_file = labels_section.get("per_image_file")
    per_image_file_path_str: Optional[str] = None
    per_image_labels: Dict[str, List[str]] = {}
    if per_image_file:
        per_image_path = (PROJECT_ROOT / per_image_file).expanduser().resolve()
        if not per_image_path.exists():
            raise FileNotFoundError(f"Per-image labels file not found: {per_image_path}")
        with per_image_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        per_image_file_path_str = str(per_image_path)
        if isinstance(raw, dict) and "labels_per_image" in raw:
            raw = raw["labels_per_image"]
        if not isinstance(raw, dict):
            raise ValueError("Per-image labels mapping must be a JSON object.")
        for key, value in raw.items():
            if isinstance(value, dict):
                label_list = value.get("labels") or value.get("all") or []
            else:
                label_list = value
            if not isinstance(label_list, list):
                raise ValueError(f"Per-image labels for {key} must be a list.")
            per_image_labels[key] = [label.strip() for label in label_list if str(label).strip()]

    if not labels and not per_image_labels:
        raise ValueError(
            "No labels provided. Populate labels.values/file or supply labels.per_image_file."
        )
    return labels, per_image_labels, per_image_file_path_str


def gather_images(images_dir: Path, limit: int | None = None) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    candidates = sorted(
        path
        for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not candidates:
        raise RuntimeError(f"No image files found in {images_dir}")
    if limit:
        candidates = candidates[:limit]
    return candidates


def should_skip_model_output(output_path: Path, expected_images: int, force: bool) -> bool:
    if force or not output_path.exists():
        return False
    try:
        with output_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    num_images = payload.get("num_images")
    if not isinstance(num_images, int):
        results = payload.get("results")
        num_images = len(results) if isinstance(results, list) else 0
    return num_images >= expected_images


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    dataset_cfg = config["dataset"]
    images_dir = Path(dataset_cfg["images_dir"]).expanduser()
    if not images_dir.is_absolute():
        images_dir = (PROJECT_ROOT / images_dir).resolve()
    limit = dataset_cfg.get("limit")

    labels, per_image_labels, per_image_labels_path = load_labels(config)
    if args.test:
        limit = 2
    image_paths = gather_images(images_dir, limit)

    force = bool(config.get("force"))

    output_cfg = config["output"]
    output_dir = Path(output_cfg["dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    models: Iterable[Dict] = config.get("models", [])
    if not models:
        raise ValueError("Config does not specify any models.")

    for model_config in models:
        model_name = model_config["name"]
        prompt_strategy = model_config.get("prompt_strategy", "single_query")
        output_name = model_config.get("output_name") or f"{model_name}.json"

        output_path = output_dir / output_name
        if should_skip_model_output(output_path, expected_images=len(image_paths), force=force):
            print(f"[skip] {model_name}: {output_path} already exists. Use --force in the config to overwrite.")
            continue

        detector_cls = MODEL_REGISTRY.get(model_name)
        if not detector_cls:
            raise ValueError(f"Unknown model: {model_name}. Available: {sorted(MODEL_REGISTRY)}")

        print(f"Running model: {model_name} with strategy: {prompt_strategy} on {len(image_paths)} images.")
        detector = detector_cls()

        per_image_dir = output_dir / "per_image" / model_name
        per_image_dir.mkdir(parents=True, exist_ok=True)

        model_start = time.perf_counter()
        results = []
        for image_path in image_paths:
            per_image_path = per_image_dir / f"{image_path.stem}.json"
            image_key = image_path.name
            labels_for_image = per_image_labels.get(image_key, labels)
            if not labels_for_image:
                raise ValueError(
                    f"No labels available for image {image_key}. Check the per-image labels file."
                )

            if per_image_path.exists() and not force:
                with per_image_path.open("r", encoding="utf-8") as fh:
                    cached_payload = json.load(fh)
                detections = cached_payload.get("detections", [])
                latency = cached_payload.get("latency_sec")
                raw_response = cached_payload.get("raw_response")
                labels_used = cached_payload.get("labels", labels_for_image)
                results.append(
                    {
                        "image": str(image_path.relative_to(images_dir)),
                        "detections": detections,
                        "latency_sec": latency,
                        "raw_response": raw_response,
                        "labels_used": labels_used,
                    }
                )
                rel_existing = per_image_path.relative_to(output_dir)
                print(
                    f"[skip] {image_path.name}: reused {rel_existing} "
                    f"({len(detections)} detections)"
                )
                continue

            start = time.perf_counter()
            try:
                prediction = detector.predict(
                    str(image_path),
                    labels_for_image,
                    strategy=prompt_strategy,
                )
            except Exception as exc:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                latency = time.perf_counter() - start
                per_image_payload = {
                    "model": model_name,
                    "strategy": prompt_strategy,
                    "image": str(image_path),
                    "labels": labels_for_image,
                    "result": {
                        "detections": [],
                        "latency_sec": latency,
                        "raw_response": {
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    },
                }
                with per_image_path.open("w", encoding="utf-8") as fh:
                    json.dump(per_image_payload, fh, indent=2)
                rel_saved = per_image_path.relative_to(output_dir)
                print(
                    f"[error] {image_path.name} ({latency:.3f}s) -> saved error to {rel_saved}"
                )
                continue
            if isinstance(prediction, dict):
                detections = prediction.get("detections", [])
                raw_response = prediction.get("raw_response")
            else:
                detections = prediction
                raw_response = None
            latency = time.perf_counter() - start
            results.append(
                {
                    "image": str(image_path.relative_to(images_dir)),
                    "detections": detections,
                    "latency_sec": latency,
                    "raw_response": raw_response,
                    "labels_used": labels_for_image,
                }
            )

            per_image_payload = {
                "model": model_name,
                "strategy": prompt_strategy,
                "image": str(image_path),
                "labels": labels_for_image,
                "detections": detections,
                "latency_sec": latency,
                "raw_response": raw_response,
            }
            with per_image_path.open("w", encoding="utf-8") as fh:
                json.dump(per_image_payload, fh, indent=2)

            rel_saved = per_image_path.relative_to(output_dir)
            print(
                f"[done] {image_path.name} ({latency:.3f}s, {len(detections)} detections) -> saved to {rel_saved}"
            )
        wall_time = time.perf_counter() - model_start
        processed_images = len(results)
        if processed_images == 0:
            print(f"No new results generated for {model_name}; skipping aggregate write.")
            continue

        total_latency = sum(entry.get("latency_sec") or 0.0 for entry in results)
        avg_latency = total_latency / processed_images if processed_images else 0.0

        payload = {
            "model": model_name,
            "strategy": prompt_strategy,
            "labels": labels,
            "per_image_labels": bool(per_image_labels),
            "per_image_labels_file": per_image_labels_path,
            "images_dir": str(images_dir),
            "num_images": processed_images,
            "results": results,
            "summary": {
                "wall_time_sec": wall_time,
                "total_latency_sec": total_latency,
                "avg_latency_sec": avg_latency,
            },
        }

        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
