"""Run temperature/prompt ablations for VLM detectors on a cached LVIS subset."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.env import load_environment  # noqa: E402

load_environment()

from experiments.run_vlm import gather_images, load_labels, load_yaml_config  # noqa: E402
from models.vlm import MODEL_REGISTRY  # noqa: E402

DEFAULT_SUBSET_CACHE = PROJECT_ROOT / "data/subsets/lvis/ablation_100_ids.json"
DEFAULT_TEMPERATURES = [0.0, 0.2, 0.5]
DEFAULT_PROMPT_VARIANTS = {
    "baseline": {"prompt_strategy": "single_query"},
    "prompt_b": {"prompt_strategy": "iterative"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablation runner that reuses the VLM experiment stack but constrains the dataset "
            "to a cached LVIS subset and iterates over temperature/prompt variants."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to the ablation config file.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Process only two images from the cached subset (smoke test).",
    )
    parser.add_argument(
        "--resample-subset",
        action="store_true",
        help="Force regeneration of the cached subset before running the ablation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run models even if cached per-image outputs already exist.",
    )
    return parser.parse_args()


def _resolve_path(path_value: str | Path, default_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (default_root / path).resolve()
    return path


def ensure_subset(
    images_dir: Path,
    cache_path: Path,
    subset_size: int,
    seed: int,
    resample: bool = False,
) -> List[Path]:
    cache_path = cache_path.expanduser()
    if not cache_path.is_absolute():
        cache_path = (PROJECT_ROOT / cache_path).resolve()

    def materialize(entries: Sequence[Dict[str, str | int]]) -> List[Path]:
        materialized: List[Path] = []
        for entry in entries:
            rel = entry.get("relative_path")
            filename = entry.get("filename")
            candidate: Optional[Path] = None
            if isinstance(rel, str) and rel:
                candidate = (images_dir / rel)
            elif isinstance(filename, str) and filename:
                candidate = (images_dir / filename)
            if candidate is None or not candidate.exists():
                raise FileNotFoundError(
                    f"Cached subset references missing image: {rel or filename}."
                )
            materialized.append(candidate)
        return materialized

    regenerate_reason: Optional[str] = None
    if cache_path.exists() and not resample:
        with cache_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        entries = cached.get("images")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Subset cache {cache_path} is malformed.")
        try:
            return materialize(entries)
        except FileNotFoundError as exc:
            regenerate_reason = str(exc)

    if regenerate_reason:
        print(
            f"[warn] Subset cache {cache_path} references missing images; regenerating. "
            f"Reason: {regenerate_reason}"
        )

    images = gather_images(images_dir)
    if len(images) < subset_size:
        raise ValueError(
            f"Requested subset of {subset_size} images but only {len(images)} candidates exist in {images_dir}."
        )
    rng = random.Random(seed)
    selection = rng.sample(images, subset_size)
    missing = [str(path) for path in selection if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The following sampled images are missing on disk: " + ", ".join(missing)
        )
    entries = [
        {
            "id": path.stem,
            "filename": path.name,
            "relative_path": path.relative_to(images_dir).as_posix(),
        }
        for path in selection
    ]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images_dir": str(images_dir),
        "subset_size": subset_size,
        "seed": seed,
        "images": entries,
    }
    with cache_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return materialize(entries)


def normalize_prompt_variants(configured: Optional[Dict[str, Dict[str, str]]]) -> Dict[str, Dict[str, str]]:
    if not configured:
        return DEFAULT_PROMPT_VARIANTS
    normalized: Dict[str, Dict[str, str]] = {}
    for name, value in configured.items():
        if value is None:
            normalized[name] = {}
        elif isinstance(value, dict):
            normalized[name] = value
        else:
            normalized[name] = {"prompt_strategy": str(value)}
    missing = [key for key in ("baseline", "prompt_b") if key not in normalized]
    if missing:
        raise ValueError(
            "Ablation prompt_variants must include entries for: " + ", ".join(missing)
        )
    return normalized


def format_temperature_folder(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def configure_generation(detector: object, temperature: float) -> None:
    """Best-effort hook to adjust HF generation configs per ablation temp."""
    do_sample = temperature > 0.0
    candidates: List[object] = []

    direct_cfg = getattr(detector, "generation_config", None)
    if direct_cfg is not None:
        candidates.append(direct_cfg)

    model = getattr(detector, "model", None)
    if model is not None:
        model_cfg = getattr(model, "generation_config", None)
        if model_cfg is not None:
            candidates.append(model_cfg)

    seen: set[int] = set()
    for cfg in candidates:
        cfg_id = id(cfg)
        if cfg_id in seen:
            continue
        seen.add(cfg_id)
        try:
            setattr(cfg, "do_sample", do_sample)
        except Exception:
            pass
        try:
            setattr(cfg, "temperature", max(temperature, 0.0))
        except Exception:
            pass

    # Some detectors might read these attributes directly.
    try:
        setattr(detector, "do_sample", do_sample)
    except Exception:
        pass
    try:
        setattr(detector, "temperature", max(temperature, 0.0))
    except Exception:
        pass


def run_model_variant(
    *,
    model_config: Dict,
    model_name: str,
    strategy: str,
    images_dir: Path,
    image_paths: Sequence[Path],
    labels: List[str],
    per_image_labels: Dict[str, List[str]],
    per_image_labels_path: Optional[str],
    output_dir: Path,
    prompt_variant: str,
    temperature: float,
    force: bool,
) -> None:
    detector_cls = MODEL_REGISTRY.get(model_name)
    if not detector_cls:
        raise ValueError(f"Unknown model: {model_name}. Available: {sorted(MODEL_REGISTRY)}")

    variant_dir = output_dir / prompt_variant / format_temperature_folder(temperature) / model_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    output_filename = model_config.get("output_name") or f"{model_name}.json"
    aggregate_path = variant_dir / output_filename
    if aggregate_path.exists() and not force:
        print(
            f"[skip] {model_name} ({prompt_variant}@{temperature}): "
            f"{aggregate_path} already exists. Use --force or force: true to recompute."
        )
        return

    print(
        f"Running {model_name} | strategy={strategy} | prompt={prompt_variant} | temp={temperature} "
        f"on {len(image_paths)} images."
    )
    detector = detector_cls()
    configure_generation(detector, temperature)

    results: List[Dict] = []
    start_time = time.perf_counter()
    for image_path in image_paths:
        per_image_path = variant_dir / f"{image_path.stem}.json"
        image_key = image_path.name
        labels_for_image = per_image_labels.get(image_key, labels)
        if not labels_for_image:
            raise ValueError(
                f"No labels resolved for {image_key}; check labels file/per-image mapping."
            )

        use_cached_result = False
        if per_image_path.exists() and not force:
            try:
                with per_image_path.open("r", encoding="utf-8") as fh:
                    cached_payload = json.load(fh)
            except json.JSONDecodeError as exc:
                print(
                    f"[warn] {image_path.name}: cached payload {per_image_path} is unreadable "
                    f"({exc}). Recomputing and overwriting."
                )
                try:
                    per_image_path.unlink()
                except OSError:
                    pass
            except OSError as exc:
                print(
                    f"[warn] {image_path.name}: failed to load cached payload {per_image_path} "
                    f"({exc}). Recomputing."
                )
            else:
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
                rel_path = per_image_path.relative_to(output_dir)
                print(f"[skip] {image_path.name}: reused {rel_path} ({len(detections)} detections)")
                use_cached_result = True

        if use_cached_result:
            continue

        inference_start = time.perf_counter()
        try:
            prediction = detector.predict(
                str(image_path),
                labels_for_image,
                strategy=strategy,
            )
        except Exception as exc:  # noqa: BLE001
            latency = time.perf_counter() - inference_start
            per_image_payload = {
                "model": model_name,
                "strategy": strategy,
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
            rel_path = per_image_path.relative_to(output_dir)
            print(f"[error] {image_path.name} ({latency:.3f}s) -> saved error to {rel_path}")
            continue

        if isinstance(prediction, dict):
            detections = prediction.get("detections", [])
            raw_response = prediction.get("raw_response")
        else:
            detections = prediction
            raw_response = None
        latency = time.perf_counter() - inference_start
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
            "strategy": strategy,
            "image": str(image_path),
            "labels": labels_for_image,
            "detections": detections,
            "latency_sec": latency,
            "raw_response": raw_response,
        }
        with per_image_path.open("w", encoding="utf-8") as fh:
            json.dump(per_image_payload, fh, indent=2)
        rel_path = per_image_path.relative_to(output_dir)
        print(
            f"[done] {image_path.name} ({latency:.3f}s, {len(detections)} detections) -> saved to {rel_path}"
        )

    wall_time = time.perf_counter() - start_time
    if not results:
        print(f"No results generated for {model_name} ({prompt_variant}@{temperature}). Skipping write.")
        return

    total_latency = sum(entry.get("latency_sec") or 0.0 for entry in results)
    avg_latency = total_latency / len(results) if results else 0.0

    payload = {
        "model": model_name,
        "strategy": strategy,
        "labels": labels,
        "per_image_labels": bool(per_image_labels),
        "per_image_labels_file": per_image_labels_path,
        "images_dir": str(images_dir),
        "num_images": len(results),
        "results": results,
        "summary": {
            "wall_time_sec": wall_time,
            "total_latency_sec": total_latency,
            "avg_latency_sec": avg_latency,
        },
    }

    with aggregate_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Results saved to {aggregate_path}")


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    dataset_cfg = config.get("dataset") or {}
    if not dataset_cfg:
        raise ValueError("Config missing dataset settings.")
    images_dir_value = dataset_cfg.get("images_dir")
    if not images_dir_value:
        raise ValueError("dataset.images_dir is required.")
    images_dir = _resolve_path(images_dir_value, PROJECT_ROOT)

    ablation_cfg = config.get("ablation", {})
    subset_cache = ablation_cfg.get("subset_cache") or DEFAULT_SUBSET_CACHE
    subset_size = int(ablation_cfg.get("subset_size", 100))
    subset_seed = int(ablation_cfg.get("subset_seed", 1337))
    prompt_variants_cfg = normalize_prompt_variants(ablation_cfg.get("prompt_variants"))
    temperature_values = ablation_cfg.get("temperatures", DEFAULT_TEMPERATURES)
    if not isinstance(temperature_values, Iterable) or isinstance(temperature_values, (str, bytes)):
        raise ValueError("ablation.temperatures must be a list of numbers.")
    temperatures = [float(value) for value in temperature_values]

    labels, per_image_labels, per_image_labels_path = load_labels(config)
    force = bool(config.get("force")) or args.force

    subset_paths = ensure_subset(
        images_dir,
        Path(subset_cache),
        subset_size=subset_size,
        seed=subset_seed,
        resample=args.resample_subset,
    )
    if args.test:
        subset_paths = subset_paths[:2]
    if not subset_paths:
        raise RuntimeError("No images available for the ablation run.")

    output_cfg = config.get("output") or {}
    output_dir_value = output_cfg.get("dir") or "results/ablation"
    output_dir = _resolve_path(output_dir_value, PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    models: Iterable[Dict] = config.get("models", [])
    if not models:
        raise ValueError("Config does not list any models to run.")

    for model_config in models:
        model_name = model_config.get("name")
        if not model_name:
            raise ValueError("Each model entry must include a name.")
        base_strategy = model_config.get("prompt_strategy", "single_query")

        for prompt_name, variant_settings in prompt_variants_cfg.items():
            strategy = variant_settings.get("prompt_strategy", base_strategy)
            for temperature in temperatures:
                run_model_variant(
                    model_config=model_config,
                    model_name=model_name,
                    strategy=strategy,
                    images_dir=images_dir,
                    image_paths=subset_paths,
                    labels=labels,
                    per_image_labels=per_image_labels,
                    per_image_labels_path=per_image_labels_path,
                    output_dir=output_dir,
                    prompt_variant=prompt_name,
                    temperature=temperature,
                    force=force,
                )


if __name__ == "__main__":
    main()
