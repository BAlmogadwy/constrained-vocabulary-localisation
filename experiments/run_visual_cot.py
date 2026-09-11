"""Run a visual chain-of-thought (iterative visual refinement) experiment for VLM detectors."""

from __future__ import annotations

import argparse
import inspect
import json
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageColor, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.env import load_environment  # noqa: E402

load_environment()

from experiments.run_vlm import gather_images, load_labels, load_yaml_config  # noqa: E402
from models.vlm import MODEL_REGISTRY  # noqa: E402

DEFAULT_SUBSET_CACHE = PROJECT_ROOT / "data/subsets/lvis/ablation_100_ids.json"
DEFAULT_FALLBACK_IMAGE_DIRS = [PROJECT_ROOT / "data/raw/coco2017/val2017"]
DEFAULT_TEMPERATURES = [0.0, 0.2, 0.5]
DEFAULT_PROMPT_VARIANTS = {
    "baseline": {"prompt_strategy": "single_query"},
    "prompt_b": {"prompt_strategy": "iterative"},
}

REFINEMENT_STEPS = 5
BASELINE_PROMPT_NOTE = "Baseline inference using the configured prompt strategy."

NormalizedBox = Tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 5-step visual chain-of-thought refinement loop using the ablation dataset configuration."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to the YAML config file.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Process only two images from the cached subset (smoke test).",
    )
    parser.add_argument(
        "--resample-subset",
        action="store_true",
        help="Force regeneration of the cached subset before running the experiment.",
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

    def _backfill_image_if_needed(target: Path, fragment: Path, sources: Sequence[Path]) -> Optional[Path]:
        for root in sources:
            if not root:
                continue
            candidate = (root / fragment).resolve()
            if candidate.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, target)
                return target
        return None

    def materialize(
        entries: Sequence[Dict[str, str | int]],
        fallback_roots: Sequence[Path],
    ) -> List[Path]:
        materialized: List[Path] = []
        for entry in entries:
            rel = entry.get("relative_path")
            filename = entry.get("filename")
            candidate: Optional[Path] = None
            fragment_str = rel if isinstance(rel, str) and rel else filename
            if not isinstance(fragment_str, str) or not fragment_str:
                raise ValueError("Cached subset entry is missing filename/relative_path.")
            fragment = Path(fragment_str)
            candidate = images_dir / fragment
            if not candidate.exists():
                recovered = _backfill_image_if_needed(candidate, fragment, fallback_roots)
                if not recovered or not recovered.exists():
                    raise FileNotFoundError(
                        f"Cached subset references missing image: {fragment_str}."
                    )
                candidate = recovered
            materialized.append(candidate)
        return materialized

    def compute_fallback_roots(preferred: Optional[str]) -> List[Path]:
        fallback_roots: List[Path] = []
        if preferred:
            preferred_path = Path(preferred).expanduser()
            if preferred_path.exists() and preferred_path.resolve() != images_dir.resolve():
                fallback_roots.append(preferred_path)
        for default_dir in DEFAULT_FALLBACK_IMAGE_DIRS:
            resolved = default_dir.expanduser().resolve()
            if resolved.exists() and resolved not in fallback_roots:
                fallback_roots.append(resolved)
        return fallback_roots

    if cache_path.exists() and not resample:
        with cache_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        entries = cached.get("images")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Subset cache {cache_path} is malformed.")
        cached_images_dir = cached.get("images_dir") if isinstance(cached, dict) else None
        fallback_roots = compute_fallback_roots(str(cached_images_dir) if cached_images_dir else None)
        return materialize(entries, fallback_roots)

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
    fallback_roots = compute_fallback_roots(str(images_dir))
    return materialize(entries, fallback_roots)


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

    try:
        setattr(detector, "do_sample", do_sample)
    except Exception:
        pass
    try:
        setattr(detector, "temperature", max(temperature, 0.0))
    except Exception:
        pass


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _coerce_box_coordinates(
    value: Any,
    default_order: str,
) -> Optional[Tuple[List[float], str]]:
    try:
        if isinstance(value, dict):
            key_orders: List[Tuple[Tuple[str, ...], str]] = [
                (("ymin", "xmin", "ymax", "xmax"), "yxyx"),
                (("xmin", "ymin", "xmax", "ymax"), "xyxy"),
                (("x1", "y1", "x2", "y2"), "xyxy"),
                (("y1", "x1", "y2", "x2"), "yxyx"),
                (("x", "y", "width", "height"), "xywh"),
            ]
            for keys, order in key_orders:
                if all(key in value for key in keys):
                    coords = [float(value[key]) for key in keys]
                    return coords, order
            return None
        coords = [float(num) for num in value]
    except (TypeError, ValueError):
        return None
    if len(coords) != 4:
        return None
    return coords, default_order


def _normalize_box(
    coords: Sequence[float],
    order: str,
    width: int,
    height: int,
    already_normalized: bool,
) -> Optional[NormalizedBox]:
    if width <= 0 or height <= 0:
        return None
    if order == "xywh":
        x, y, w, h = coords
        coords = (x, y, x + w, y + h)
        order = "xyxy"
    if order == "xyxy":
        x1, y1, x2, y2 = coords
    elif order == "yxyx":
        y1, x1, y2, x2 = coords
    else:
        return None
    if not already_normalized:
        x1 /= width
        x2 /= width
        y1 /= height
        y2 /= height
    x1_norm = clamp(min(x1, x2))
    x2_norm = clamp(max(x1, x2))
    y1_norm = clamp(min(y1, y2))
    y2_norm = clamp(max(y1, y2))
    return (y1_norm, x1_norm, y2_norm, x2_norm)


def normalize_detections_to_boxes(
    detections: Sequence[Any],
    image_size: Tuple[int, int],
) -> Tuple[List[NormalizedBox], List[Optional[str]]]:
    width, height = image_size
    normalized: List[NormalizedBox] = []
    labels: List[Optional[str]] = []
    if width <= 0 or height <= 0:
        return normalized, labels
    for entry in detections:
        if not isinstance(entry, dict):
            continue
        candidates = [
            ("box_normalized_yxyx", "yxyx", True),
            ("box_normalized_xyxy", "xyxy", True),
            ("box_normalized", "xyxy", True),
            ("box_norm", "xyxy", True),
            ("box", "xyxy", False),
            ("box_2d", "xyxy", False),
            ("bbox", "xywh", False),
        ]
        normalized_box: Optional[NormalizedBox] = None
        for key, default_order, is_normalized in candidates:
            if key not in entry:
                continue
            coords_order = _coerce_box_coordinates(entry[key], default_order)
            if coords_order is None:
                continue
            coords, order = coords_order
            normalized_box = _normalize_box(coords, order, width, height, is_normalized)
            if normalized_box:
                break
        if normalized_box:
            normalized.append(normalized_box)
            labels.append(entry.get("label"))
    return normalized, labels


class BoundingBoxAnnotator:
    """Draw normalized (ymin, xmin, ymax, xmax) boxes onto an image copy."""

    def __init__(self, color: str = "red", width: int = 3, fill_alpha: int = 60) -> None:
        self.color = color
        self.width = max(1, width)
        self.fill_alpha = max(0, min(fill_alpha, 255))
        self.color_rgb = ImageColor.getrgb(color)
        self.font = ImageFont.load_default()

    def draw(
        self,
        image_path: Path,
        normalized_boxes: Sequence[NormalizedBox],
        output_path: Path,
        labels: Optional[Sequence[Optional[str]]] = None,
    ) -> Path:
        if not normalized_boxes:
            raise ValueError("No boxes supplied for annotation.")
        with Image.open(image_path) as base_image:
            canvas = base_image.convert("RGBA")
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        width_px, height_px = canvas.size
        for idx, box in enumerate(normalized_boxes):
            y_min, x_min, y_max, x_max = box
            x1 = clamp(x_min) * width_px
            x2 = clamp(x_max) * width_px
            y1 = clamp(y_min) * height_px
            y2 = clamp(y_max) * height_px
            fill = (*self.color_rgb, self.fill_alpha) if self.fill_alpha else None
            draw_overlay.rectangle(
                [x1, y1, x2, y2],
                outline=(*self.color_rgb, 255),
                width=self.width,
                fill=fill,
            )
            if labels and idx < len(labels):
                text = labels[idx]
                if text:
                    draw_overlay.text(
                        (x1 + 4, max(0, y1 + 4)),
                        str(text),
                        font=self.font,
                        fill=(255, 255, 255, 255),
                    )
        blended = Image.alpha_composite(canvas, overlay).convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        blended.save(output_path)
        return output_path


def compose_side_by_side(
    original_path: Path,
    annotated_path: Path,
    output_path: Path,
) -> Path:
    with Image.open(original_path) as left_image:
        left = left_image.convert("RGB")
    with Image.open(annotated_path) as right_image:
        right = right_image.convert("RGB")
    spacing = 12
    height = max(left.height, right.height)
    width = left.width + right.width + spacing
    canvas = Image.new("RGB", (width, height), color=(16, 16, 16))
    left_y = (height - left.height) // 2
    right_y = (height - right.height) // 2
    canvas.paste(left, (0, left_y))
    canvas.paste(right, (left.width + spacing, right_y))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 10), "Original", fill=(240, 240, 240), font=font)
    draw.text((left.width + spacing + 10, 10), "Refinement", fill=(240, 240, 240), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def supports_multi_image_input(detector: object) -> bool:
    flag = getattr(detector, "supports_multi_image_input", None)
    if flag is not None:
        return bool(flag)
    predict_fn = getattr(detector, "predict", None)
    if not callable(predict_fn):
        return False
    try:
        signature = inspect.signature(predict_fn)
    except (TypeError, ValueError):
        return False
    parameters = list(signature.parameters.values())
    if not parameters:
        return False
    first_param = parameters[0]
    annotation = first_param.annotation
    if annotation is not inspect._empty:
        origin = getattr(annotation, "__origin__", None)
        if origin in (list, tuple, Sequence):
            return True
        if annotation in (List[str], Sequence[str], Tuple[str, ...]):
            return True
    param_name = first_param.name
    return param_name in {"image_paths", "images"}


def get_image_size(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as handle:
        width, height = handle.size
    return width, height


def build_refinement_prompt(labels: List[str], had_boxes: bool, overlay_hint: bool = True) -> str:
    labels_text = ", ".join(labels)
    if had_boxes:
        prompt = (
            "I have drawn your predictions from the previous step in red. "
            "Visually inspect the alignment. Are the boxes accurate? If not, provide corrected "
            f"coordinates for: {labels_text}. Return JSON only."
        )
    else:
        prompt = (
            "You found no objects previously. Look closer and provide bounding boxes for any of the "
            f"following labels: {labels_text}. Return JSON only."
        )
    if overlay_hint:
        prompt += " The left panel is the original image; the right panel shows your previous boxes in red."
    return prompt


def summarize_chain(chain: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for entry in chain:
        detections = entry.get("detections") or []
        summary.append(
            {
                "step": entry.get("step"),
                "stage": entry.get("stage"),
                "annotated_image": entry.get("annotated_image"),
                "latency_sec": entry.get("latency_sec"),
                "num_detections": len(detections) if isinstance(detections, list) else 0,
            }
        )
    return summary


def build_result_entry(
    *,
    image_relative: str,
    labels_used: List[str],
    final_detections: List[Dict[str, Any]],
    total_latency: Optional[float],
    chain: Sequence[Dict[str, Any]],
    per_image_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    rel_path = per_image_path.relative_to(output_dir)
    return {
        "image": image_relative,
        "labels_used": labels_used,
        "final_detections": final_detections,
        "final_latency_sec": total_latency,
        "chain_file": str(rel_path),
        "chain_summary": summarize_chain(chain),
    }


def run_visual_cot_variant(
    *,
    model_config: Dict[str, Any],
    model_name: str,
    strategy: str,
    fallback_strategy: str,
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
    debug_dir_root = variant_dir / "debug_images"
    debug_dir_root.mkdir(parents=True, exist_ok=True)

    output_filename = model_config.get("output_name") or f"{model_name}.json"
    aggregate_path = variant_dir / output_filename
    if aggregate_path.exists() and not force:
        print(
            f"[skip] {model_name} ({prompt_variant}@{temperature}): "
            f"{aggregate_path} already exists. Use --force or force: true to recompute."
        )
        return

    detector = detector_cls()
    configure_generation(detector, temperature)
    allow_multi_image = supports_multi_image_input(detector)
    annotator = BoundingBoxAnnotator(color="red", width=3)

    results: List[Dict[str, Any]] = []
    start_time = time.perf_counter()
    for image_path in image_paths:
        per_image_path = variant_dir / f"{image_path.stem}.json"
        image_key = image_path.name
        labels_for_image = per_image_labels.get(image_key, labels)
        if not labels_for_image:
            raise ValueError(
                f"No labels resolved for {image_key}; check labels file/per-image mapping."
            )

        if per_image_path.exists() and not force:
            with per_image_path.open("r", encoding="utf-8") as fh:
                cached_payload = json.load(fh)
            final_dets = cached_payload.get("final_detections") or cached_payload.get("detections", [])
            total_latency = cached_payload.get("final_latency_sec") or cached_payload.get("latency_sec")
            chain_entries = cached_payload.get("refinement_chain") or []
            results.append(
                build_result_entry(
                    image_relative=str(image_path.relative_to(images_dir)),
                    labels_used=cached_payload.get("labels", labels_for_image),
                    final_detections=final_dets,
                    total_latency=total_latency,
                    chain=chain_entries,
                    per_image_path=per_image_path,
                    output_dir=output_dir,
                )
            )
            rel_path = per_image_path.relative_to(output_dir)
            print(f"[skip] {image_path.name}: reused {rel_path} (chain length={len(chain_entries)})")
            continue

        try:
            image_size = get_image_size(image_path)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to read {image_path}: {exc}") from exc

        prev_norm_boxes: List[NormalizedBox] = []
        prev_box_labels: List[str] = []
        chain_entries: List[Dict[str, Any]] = []
        total_latency = 0.0
        current_strategy = strategy
        per_image_debug_dir = debug_dir_root / image_path.stem
        per_image_debug_dir.mkdir(parents=True, exist_ok=True)

        for step_idx in range(REFINEMENT_STEPS):
            stage = "baseline" if step_idx == 0 else "refinement"
            had_boxes = bool(prev_norm_boxes)
            prompt_text = (
                BASELINE_PROMPT_NOTE
                if step_idx == 0
                else build_refinement_prompt(labels_for_image, had_boxes, overlay_hint=had_boxes)
            )
            if step_idx == 0:
                labels_for_call = list(labels_for_image)
            else:
                labels_for_call = list(labels_for_image)

            annotated_image_path: Optional[Path] = None
            model_inputs: List[str]
            if step_idx == 0:
                model_inputs = [str(image_path)]
            elif prev_norm_boxes:
                annotated_image_path = per_image_debug_dir / f"step_{step_idx}.png"
                annotator.draw(
                    image_path,
                    prev_norm_boxes,
                    annotated_image_path,
                    labels=prev_box_labels,
                )
                if allow_multi_image:
                    model_inputs = [str(image_path), str(annotated_image_path)]
                else:
                    paired_path = per_image_debug_dir / f"step_{step_idx}_panel.png"
                    compose_side_by_side(image_path, annotated_image_path, paired_path)
                    model_inputs = [str(paired_path)]
            else:
                model_inputs = [str(image_path)]

            inference_start = time.perf_counter()
            attempt_strategy = current_strategy
            prediction: Any = None
            raw_response: Any = None
            detections: List[Dict[str, Any]] = []
            error_payload: Optional[Dict[str, Any]] = None

            while True:
                try:
                    model_input = model_inputs if (allow_multi_image and len(model_inputs) > 1) else model_inputs[0]
                    prediction = detector.predict(
                        model_input,
                        labels_for_call,
                        strategy=attempt_strategy,
                    )
                    if attempt_strategy != current_strategy:
                        current_strategy = attempt_strategy
                    break
                except ValueError as exc:
                    msg = str(exc).lower()
                    if "not supported" in msg and fallback_strategy and attempt_strategy != fallback_strategy:
                        print(
                            f"[warn] {model_name}: strategy '{attempt_strategy}' unsupported. "
                            f"Falling back to '{fallback_strategy}'."
                        )
                        attempt_strategy = fallback_strategy
                        continue
                    latency = time.perf_counter() - inference_start
                    error_payload = {"error": str(exc), "traceback": traceback.format_exc()}
                    raw_response = error_payload
                    break
                except Exception as exc:  # noqa: BLE001
                    latency = time.perf_counter() - inference_start
                    error_payload = {"error": str(exc), "traceback": traceback.format_exc()}
                    raw_response = error_payload
                    break

            if prediction is not None:
                if isinstance(prediction, dict):
                    detections = prediction.get("detections", []) or []
                    raw_response = prediction.get("raw_response")
                else:
                    detections = prediction or []
                    raw_response = None
            else:
                detections = []

            latency = time.perf_counter() - inference_start
            total_latency += latency

            if not isinstance(detections, list):
                detections = []
            norm_boxes, norm_labels = normalize_detections_to_boxes(detections, image_size)
            prev_norm_boxes = norm_boxes
            prev_box_labels = [str(label) if label is not None else "" for label in norm_labels]

            chain_entry = {
                "step": step_idx,
                "stage": stage,
                "prompt": prompt_text,
                "labels_sent": labels_for_call,
                "input_images": model_inputs,
                "annotated_image": str(annotated_image_path) if annotated_image_path else (str(image_path) if step_idx == 0 else None),
                "detections": detections,
                "latency_sec": latency,
                "raw_response": raw_response,
                "status": "error" if error_payload else "ok",
                "normalized_boxes": [list(box) for box in norm_boxes],
            }
            chain_entries.append(chain_entry)

        final_detections = chain_entries[-1]["detections"] if chain_entries else []
        per_image_payload = {
            "model": model_name,
            "strategy": current_strategy,
            "experiment": "visual_cot",
            "prompt_variant": prompt_variant,
            "temperature": temperature,
            "image": str(image_path),
            "image_relative": str(image_path.relative_to(images_dir)),
            "labels": labels_for_image,
            "metadata": {
                "image": str(image_path),
                "image_relative": str(image_path.relative_to(images_dir)),
                "image_size": {"width": image_size[0], "height": image_size[1]},
                "labels": labels_for_image,
                "ground_truth_labels": per_image_labels.get(image_key, []),
            },
            "detections": final_detections,
            "latency_sec": total_latency,
            "final_detections": final_detections,
            "final_latency_sec": total_latency,
            "refinement_chain": chain_entries,
        }
        with per_image_path.open("w", encoding="utf-8") as fh:
            json.dump(per_image_payload, fh, indent=2)
        rel_path = per_image_path.relative_to(output_dir)
        print(
            f"[done] {image_path.name}: steps={len(chain_entries)} final_boxes={len(final_detections)} "
            f"total={total_latency:.3f}s -> saved to {rel_path}"
        )

        results.append(
            build_result_entry(
                image_relative=str(image_path.relative_to(images_dir)),
                labels_used=labels_for_image,
                final_detections=final_detections,
                total_latency=total_latency,
                chain=chain_entries,
                per_image_path=per_image_path,
                output_dir=output_dir,
            )
        )

    wall_time = time.perf_counter() - start_time
    if not results:
        print(
            f"No results generated for {model_name} ({prompt_variant}@{temperature}). Skipping write."
        )
        return

    total_latency_sum = sum(float(entry.get("final_latency_sec") or 0.0) for entry in results)
    avg_latency = total_latency_sum / len(results) if results else 0.0

    payload = {
        "model": model_name,
        "strategy": strategy,
        "prompt_variant": prompt_variant,
        "temperature": temperature,
        "labels": labels,
        "per_image_labels": bool(per_image_labels),
        "per_image_labels_file": per_image_labels_path,
        "images_dir": str(images_dir),
        "num_images": len(results),
        "refinement_steps": REFINEMENT_STEPS,
        "multi_image_supported": allow_multi_image,
        "results": results,
        "summary": {
            "wall_time_sec": wall_time,
            "total_latency_sec": total_latency_sum,
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
        raise RuntimeError("No images available for the visual chain-of-thought run.")

    output_cfg = config.get("output") or {}
    output_dir_value = output_cfg.get("dir") or "results/visual_cot"
    output_dir = _resolve_path(output_dir_value, PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    models: Iterable[Dict[str, Any]] = config.get("models", [])
    if not models:
        raise ValueError("Config does not list any models to run.")

    for model_config in models:
        model_name = model_config.get("name")
        if not model_name:
            raise ValueError("Each model entry must include a name.")
        base_strategy = model_config.get("prompt_strategy", "single_query")

        for prompt_name, variant_settings in prompt_variants_cfg.items():
            strategy_to_use = variant_settings.get("prompt_strategy", base_strategy)
            for temperature in temperatures:
                run_visual_cot_variant(
                    model_config=model_config,
                    model_name=model_name,
                    strategy=strategy_to_use,
                    fallback_strategy=base_strategy,
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
