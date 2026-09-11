# Copyright (c) 2024.
# SPDX-License-Identifier: MIT
"""
Batch runner that executes traditional zero-shot detectors based on a YAML config.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from analysis.summarize_traditional_results import (
    load_json as load_result_json,
    render_markdown,
    summarize_file,
)
from experiments.run_traditional_inference import (
    MODEL_REGISTRY,
    as_serializable,
    gather_images,
)


@dataclass
class DatasetConfig:
    images_dir: Path
    limit: Optional[int] = None


@dataclass
class LabelsConfig:
    values: List[str]
    file: Optional[Path] = None
    per_image_file: Optional[Path] = None


@dataclass
class OutputConfig:
    dir: Path


@dataclass
class ModelRunConfig:
    name: str
    device: Optional[str] = None
    output_name: Optional[str] = None
    strategy: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run traditional detectors from a YAML config.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of outputs even if cached results exist.",
    )
    return parser.parse_args()


def load_yaml_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_labels(config: LabelsConfig) -> tuple[List[str], Dict[str, List[str]], Optional[Path]]:
    labels = list(config.values)
    if config.file:
        if not config.file.exists():
            raise FileNotFoundError(f"Labels file not found: {config.file}")
        with config.file.open("r", encoding="utf-8") as fh:
            labels.extend([line.strip() for line in fh if line.strip()])
    labels = [label.strip() for label in labels if label.strip()]
    per_image_labels: Dict[str, List[str]] = {}
    per_image_path: Optional[Path] = config.per_image_file
    if per_image_path:
        if not per_image_path.exists():
            raise FileNotFoundError(f"Per-image labels file not found: {per_image_path}")
        with per_image_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
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
            per_image_labels[key] = [str(item).strip() for item in label_list if str(item).strip()]

    if not labels and not per_image_labels:
        raise ValueError("No labels provided. Populate the labels section/file or per-image mapping.")
    return labels, per_image_labels, per_image_path


def run_model(
    dataset_cfg: DatasetConfig,
    labels: List[str],
    per_image_labels: Dict[str, List[str]],
    per_image_file: Optional[Path],
    model_cfg: ModelRunConfig,
    output_cfg: OutputConfig,
    force: bool,
) -> Path:
    detector_cls = MODEL_REGISTRY.get(model_cfg.name)
    if detector_cls is None:
        raise ValueError(f"Unknown model '{model_cfg.name}'. Choices: {sorted(MODEL_REGISTRY)}")

    detector = detector_cls(device=model_cfg.device)
    images = gather_images(dataset_cfg.images_dir, dataset_cfg.limit)

    output_dir = output_cfg.dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = model_cfg.output_name or f"{model_cfg.name}.json"
    output_path = output_dir / output_name
    strategy_name = model_cfg.strategy or "default"
    per_image_dir = output_dir / "per_image" / model_cfg.name
    per_image_dir.mkdir(parents=True, exist_ok=True)

    print(f"[model] {model_cfg.name} -> {output_path}")
    records = []
    model_start = time.perf_counter()
    for image_path in images:
        per_image_path = per_image_dir / f"{image_path.stem}.json"
        if per_image_path.exists() and not force:
            cached = json.loads(per_image_path.read_text(encoding="utf-8"))
            cached_result = cached.get("result", {})
            detections = cached_result.get("detections", cached.get("detections", []))
            latency = cached_result.get("latency_sec", cached.get("latency_sec"))
            raw_response = cached_result.get("raw_response", cached.get("raw_response"))
            records.append(
                {
                    "image": str(image_path.relative_to(dataset_cfg.images_dir)),
                    "detections": detections,
                    "latency_sec": latency,
                    "raw_response": raw_response,
                    "labels_used": cached.get("labels", labels),
                }
            )
            rel_existing = per_image_path.relative_to(output_dir)
            print(
                f"[skip] {model_cfg.name} :: {image_path.name} "
                f"({len(detections)} detections) -> reused {rel_existing}"
            )
            continue

        start = time.perf_counter()
        labels_for_image = per_image_labels.get(image_path.name, labels)
        if not labels_for_image:
            raise ValueError(f"No labels available for image {image_path.name}.")
        detections = detector.predict(image_path, labels_for_image)
        latency = time.perf_counter() - start
        serializable = as_serializable(detections)
        records.append(
            {
                "image": str(image_path.relative_to(dataset_cfg.images_dir)),
                "detections": serializable,
                "latency_sec": latency,
                "raw_response": None,
                "labels_used": labels_for_image,
            }
        )
        print(f"[done] {model_cfg.name} :: {image_path.name} ({latency:.3f}s, {len(detections)} detections)")
        per_image_payload = {
            "model": model_cfg.name,
            "strategy": strategy_name,
            "image": str(image_path),
            "labels": labels_for_image,
            "result": {
                "detections": serializable,
                "latency_sec": latency,
                "raw_response": None,
            },
        }
        per_image_path.write_text(json.dumps(per_image_payload, indent=2), encoding="utf-8")
    wall_time = time.perf_counter() - model_start
    total_latency = sum(entry.get("latency_sec") or 0.0 for entry in records)
    avg_latency = total_latency / len(records) if records else 0.0

    payload = {
        "model": model_cfg.name,
        "device": model_cfg.device,
        "strategy": strategy_name,
        "labels": labels,
        "per_image_labels": bool(per_image_labels),
        "per_image_labels_file": str(per_image_file) if per_image_file else None,
        "images_dir": str(dataset_cfg.images_dir),
        "num_images": len(records),
        "results": records,
        "summary": {
            "wall_time_sec": wall_time,
            "total_latency_sec": total_latency,
            "avg_latency_sec": avg_latency,
        },
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return output_path


def main() -> None:
    args = parse_args()
    config_dict = load_yaml_config(args.config)

    dataset_cfg = DatasetConfig(
        images_dir=Path(config_dict["dataset"]["images_dir"]),
        limit=config_dict["dataset"].get("limit"),
    )
    dataset_cfg.images_dir = dataset_cfg.images_dir.expanduser().resolve()

    labels_section = config_dict.get("labels", {})
    labels_cfg = LabelsConfig(
        values=labels_section.get("values", []) or [],
        file=Path(labels_section["file"]).expanduser().resolve()
        if labels_section.get("file")
        else None,
        per_image_file=Path(labels_section["per_image_file"]).expanduser().resolve()
        if labels_section.get("per_image_file")
        else None,
    )
    labels, per_image_labels, per_image_path = load_labels(labels_cfg)

    output_cfg = OutputConfig(dir=Path(config_dict["output"]["dir"]).expanduser().resolve())
    force = bool(config_dict.get("force")) or args.force

    model_entries = config_dict.get("models", [])
    if not model_entries:
        raise ValueError("No models specified in configuration.")

    output_paths: List[Path] = []
    for entry in model_entries:
        model_cfg = ModelRunConfig(
            name=entry["name"],
            device=entry.get("device"),
            output_name=entry.get("output_name"),
            strategy=entry.get("strategy"),
        )
        output_path = run_model(
            dataset_cfg,
            labels,
            per_image_labels,
            per_image_path,
            model_cfg,
            output_cfg,
            force=force,
        )
        print(f"[write] {output_path}")
        output_paths.append(output_path)

    if output_paths:
        summaries = [summarize_file(load_result_json(path)) for path in output_paths]
        report = render_markdown(summaries)
        print("\n=== Traditional Model Summary ===\n")
        print(report)
        summary_path = output_cfg.dir / "summary.md"
        summary_path.write_text(report, encoding="utf-8")
        print(f"[write] {summary_path}")


if __name__ == "__main__":
    main()
