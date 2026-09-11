"""
Utility script to run a single-image debug inference using the same pipeline
as the full VLM experiments.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.env import load_environment  # noqa: E402

load_environment()

from models.vlm import MODEL_REGISTRY, MODEL_API_ENV  # noqa: E402
from models.vlm.llava_detector import LlavaDetector  # noqa: E402
from models.vlm.llama_vision_detector import LlamaVisionDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single-image debug inference for one or more VLM detectors."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file used in experiments.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional explicit image path. If omitted, the first image from the config dataset is used.",
    )
    parser.add_argument(
        "--model-name",
        help="Name of the model to debug (must exist in the config). If omitted, runs all API-backed models with credentials configured.",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        help="Optional JSON file with expected detections for validation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the debug detections JSON (only for single-model runs).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run inference even if a cached result already exists on disk.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device for local models (cpu, cuda, cuda:0, etc.). Default auto-selects CUDA when available.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16192,
        help="Maximum tokens to generate for local decoder models (default 16192).",
    )
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict:
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_labels(config: dict) -> List[str]:
    labels_config = config.get("labels", {})
    labels = list(labels_config.get("values", []) or [])
    label_file = labels_config.get("file")
    if label_file:
        with (PROJECT_ROOT / label_file).open("r", encoding="utf-8") as handle:
            labels.extend([line.strip() for line in handle if line.strip()])

    seen = set()
    deduped: List[str] = []
    for label in labels:
        if label not in seen:
            deduped.append(label)
            seen.add(label)
    return deduped


def pick_image(args: argparse.Namespace, config: dict) -> Path:
    if args.image:
        if not args.image.exists():
            raise FileNotFoundError(f"Provided image not found: {args.image}")
        return args.image.resolve()

    dataset_cfg = config.get("dataset", {})
    images_dir = dataset_cfg.get("images_dir")
    if not images_dir:
        raise ValueError("Config missing dataset.images_dir.")

    image_dir_path = (PROJECT_ROOT / images_dir).resolve()
    if not image_dir_path.exists():
        raise FileNotFoundError(f"Images directory not found: {image_dir_path}")

    candidates = []
    for extension in ("*.jpg", "*.jpeg", "*.png"):
        candidates.extend(sorted(image_dir_path.glob(extension)))

    if not candidates:
        raise FileNotFoundError(
            f"No images found in {image_dir_path}. Supported extensions: jpg, jpeg, png."
        )
    return candidates[0]


def select_model_configs(args: argparse.Namespace, config: dict) -> List[dict]:
    models = config.get("models", [])
    if not models:
        raise ValueError("Config contains no models to run.")

    if args.model_name:
        requested_cls = MODEL_REGISTRY.get(args.model_name)
        exact_cfg = next(
            (cfg for cfg in models if cfg.get("name") == args.model_name),
            None,
        )
        if exact_cfg:
            required_env = MODEL_API_ENV.get(args.model_name)
            if required_env and not os.getenv(required_env):
                raise ValueError(
                    f"Model '{args.model_name}' requires environment variable '{required_env}'."
                )
            return [exact_cfg]

        if requested_cls:
            for model_cfg in models:
                cfg_name = model_cfg.get("name")
                if MODEL_REGISTRY.get(cfg_name) is requested_cls:
                    required_env = MODEL_API_ENV.get(cfg_name)
                    if required_env and not os.getenv(required_env):
                        raise ValueError(
                            f"Model '{args.model_name}' requires environment variable '{required_env}'."
                        )
                    aliased_cfg = dict(model_cfg)
                    aliased_cfg["name"] = args.model_name
                    return [aliased_cfg]
        available = ", ".join(model.get("name", "<unnamed>") for model in models)
        raise ValueError(f"Model '{args.model_name}' not found. Available: {available}")

    selected: List[dict] = []
    for model_cfg in models:
        model_name = model_cfg.get("name")
        required_env = MODEL_API_ENV.get(model_name)
        if not required_env:
            print(f"[skip] {model_name}: open-source/local model (run explicitly with --model-name).")
            continue
        if not os.getenv(required_env):
            print(f"[skip] {model_name}: environment variable '{required_env}' not set.")
            continue
        selected.append(model_cfg)

    if not selected:
        raise ValueError(
            "No API-backed models with configured credentials found. "
            "Ensure the relevant API keys are set or provide --model-name."
        )
    return selected


def compare_with_expected(detections: List[Dict[str, Any]], expected_path: Path) -> None:
    with expected_path.open("r", encoding="utf-8") as fh:
        expected = json.load(fh)
    if detections != expected:
        raise AssertionError(
            "Detections differ from expected output.\n"
            f"Expected: {expected}\n"
            f"Received: {detections}"
        )
    print("Detections match the expected output.")


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    labels = resolve_labels(config)
    image_path = pick_image(args, config)
    model_configs = select_model_configs(args, config)

    if args.output and len(model_configs) > 1:
        raise ValueError("--output can only be specified when a single model is selected.")

    resolved_device: Optional[str]
    device_request = args.device.lower() if isinstance(args.device, str) else "auto"
    if device_request == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        try:
            resolved_device = str(torch.device(device_request))
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid device specification '{args.device}'.") from exc
        if torch.device(resolved_device).type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but is unavailable on this system.")

    for model_cfg in model_configs:
        model_name = model_cfg["name"]
        strategy = model_cfg.get("prompt_strategy", "single_query")
        print(
            f"Running debug inference with model '{model_name}' ({strategy}) on {image_path}"
        )

        detector_cls = MODEL_REGISTRY.get(model_name)
        if not detector_cls:
            raise ValueError(f"Unknown model: {model_name}")

        default_dir = PROJECT_ROOT / "results" / "debug"
        default_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output
        if not output_path:
            output_filename = f"{model_name.replace('-', '_')}_{strategy}_{image_path.stem}.json"
            output_path = default_dir / output_filename

        if output_path.exists() and not args.force:
            print(f"[skip] Existing debug output found at {output_path}. Use --force to regenerate.")
            if args.expected:
                with output_path.open("r", encoding="utf-8") as fh:
                    cached = json.load(fh)
                cached_dets = cached.get("result", {}).get("detections", [])
                compare_with_expected(cached_dets, args.expected)
            continue

        detector_kwargs: Dict[str, Any] = {}
        if isinstance(detector_cls, type):
            if issubclass(detector_cls, LlavaDetector):
                detector_kwargs["forced_device"] = resolved_device
                detector_kwargs["max_new_tokens"] = args.max_new_tokens
            elif issubclass(detector_cls, LlamaVisionDetector):
                detector_kwargs["forced_device"] = resolved_device
                detector_kwargs["max_new_tokens"] = args.max_new_tokens
        detector = detector_cls(**detector_kwargs)
        if not detector_kwargs and hasattr(detector, "max_new_tokens"):
            try:
                detector.max_new_tokens = args.max_new_tokens
            except Exception:  # pragma: no cover - best effort
                pass
        start = time.perf_counter()
        prediction = detector.predict(
            str(image_path),
            labels,
            strategy=strategy,
        )
        latency = time.perf_counter() - start

        if isinstance(prediction, dict):
            detections = prediction.get("detections", [])
            raw_response = prediction.get("raw_response")
        else:
            detections = prediction
            raw_response = None

        payload = {
            "model": model_name,
            "strategy": strategy,
            "labels": labels,
            "image": str(image_path),
            "result": {
                "detections": detections,
                "latency_sec": latency,
                "raw_response": raw_response,
            },
        }

        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        print(f"Debug detections saved to {output_path}")

        if args.expected:
            compare_with_expected(payload["result"]["detections"], args.expected)


if __name__ == "__main__":
    main()
