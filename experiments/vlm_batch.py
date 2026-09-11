"""Batch inference helpers for hosted VLM providers.

This module intentionally supports only providers with discounted batch APIs:
OpenAI, Anthropic, and Gemini. OpenRouter and DashScope/Qwen remain on the
regular synchronous runner.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image

from experiments.run_vlm import (
    PROJECT_ROOT,
    gather_images,
    load_labels,
    load_yaml_config,
    should_skip_model_output,
)
from models.vlm.parsing import normalize_detections_to_image, parse_detections_from_text


@dataclass(frozen=True)
class BatchModelSpec:
    provider: str
    model_env: str
    default_model: str
    max_tokens_env: str
    default_max_tokens: int


BATCH_MODEL_SPECS: Dict[str, BatchModelSpec] = {
    "gpt-5.5": BatchModelSpec(
        provider="openai",
        model_env="OPENAI_GPT55_MODEL",
        default_model="gpt-5.5",
        max_tokens_env="OPENAI_MAX_OUTPUT_TOKENS",
        default_max_tokens=2048,
    ),
    "gpt-5.4-mini": BatchModelSpec(
        provider="openai",
        model_env="OPENAI_GPT54_MINI_MODEL",
        default_model="gpt-5.4-mini",
        max_tokens_env="OPENAI_MAX_OUTPUT_TOKENS",
        default_max_tokens=2048,
    ),
    "claude-opus-4.8": BatchModelSpec(
        provider="anthropic",
        model_env="ANTHROPIC_CLAUDE_OPUS_48_MODEL",
        default_model="claude-opus-4-8",
        max_tokens_env="ANTHROPIC_MAX_TOKENS",
        default_max_tokens=1024,
    ),
    "claude-haiku-4.5": BatchModelSpec(
        provider="anthropic",
        model_env="ANTHROPIC_CLAUDE_HAIKU_45_MODEL",
        default_model="claude-haiku-4-5",
        max_tokens_env="ANTHROPIC_MAX_TOKENS",
        default_max_tokens=1024,
    ),
    "gemini-3.1-pro": BatchModelSpec(
        provider="gemini",
        model_env="GEMINI_31_PRO_MODEL",
        default_model="gemini-3.1-pro-preview",
        max_tokens_env="GEMINI_MAX_OUTPUT_TOKENS",
        default_max_tokens=2048,
    ),
    "gemini-3.5-flash": BatchModelSpec(
        provider="gemini",
        model_env="GEMINI_35_FLASH_MODEL",
        default_model="gemini-3.5-flash",
        max_tokens_env="GEMINI_MAX_OUTPUT_TOKENS",
        default_max_tokens=2048,
    ),
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def _custom_id(model_name: str, image_stem: str) -> str:
    suffix = f"__{image_stem}"
    max_prefix_len = 64 - len(suffix)
    return f"{_safe_name(model_name)[:max_prefix_len]}{suffix}"


def _prompt(classes: List[str]) -> str:
    return (
        "Detect the following objects in the image: "
        f"{', '.join(classes)}. For each detected object, return JSON in the form "
        '[{"box_2d": [x1, y1, x2, y2], "label": "class_name", "confidence": score}, ...]. '
        "Use integer pixel coordinates relative to the input image. Return only JSON."
    )


def _mime_type(image_path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(image_path))
    return guessed or "image/jpeg"


def _encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def _data_url(image_path: Path) -> str:
    return f"data:{_mime_type(image_path)};base64,{_encode_image(image_path)}"


def _model_id(spec: BatchModelSpec) -> str:
    return os.getenv(spec.model_env, spec.default_model)


def _max_tokens(spec: BatchModelSpec) -> int:
    return int(os.getenv(spec.max_tokens_env, str(spec.default_max_tokens)))


def build_openai_batch_line(
    custom_id: str,
    spec: BatchModelSpec,
    image_path: Path,
    labels: List[str],
) -> Dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": _model_id(spec),
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": _prompt(labels)},
                        {"type": "input_image", "image_url": _data_url(image_path)},
                    ],
                }
            ],
            "max_output_tokens": _max_tokens(spec),
        },
    }


def build_anthropic_batch_request(
    custom_id: str,
    spec: BatchModelSpec,
    image_path: Path,
    labels: List[str],
) -> Dict[str, Any]:
    return {
        "custom_id": custom_id,
        "params": {
            "model": _model_id(spec),
            "max_tokens": _max_tokens(spec),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": _mime_type(image_path),
                                "data": _encode_image(image_path),
                            },
                        },
                        {"type": "text", "text": _prompt(labels)},
                    ],
                }
            ],
        },
    }


def build_gemini_batch_line(
    custom_id: str,
    spec: BatchModelSpec,
    image_path: Path,
    labels: List[str],
) -> Dict[str, Any]:
    return {
        "key": custom_id,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": _prompt(labels)},
                        {
                            "inline_data": {
                                "mime_type": _mime_type(image_path),
                                "data": _encode_image(image_path),
                            }
                        },
                    ],
                }
            ],
            "generation_config": {
                "max_output_tokens": _max_tokens(spec),
                "temperature": 0.2,
            },
        },
    }


def _resolve_images_dir(config: Dict[str, Any]) -> Path:
    images_dir = Path(config["dataset"]["images_dir"]).expanduser()
    if not images_dir.is_absolute():
        images_dir = (PROJECT_ROOT / images_dir).resolve()
    return images_dir


def _resolve_output_dir(config: Dict[str, Any]) -> Path:
    output_dir = Path(config["output"]["dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    return output_dir


def prepare_batches(
    config_path: Path,
    batch_name: Optional[str] = None,
    test: bool = False,
) -> Dict[str, Any]:
    config = load_yaml_config(config_path)
    images_dir = _resolve_images_dir(config)
    limit = 2 if test else config["dataset"].get("limit")
    image_paths = gather_images(images_dir, limit)
    labels, per_image_labels, per_image_labels_path = load_labels(config)
    output_dir = _resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    if batch_name is None:
        batch_name = time.strftime("%Y%m%d_%H%M%S")
    batch_root = output_dir / "batches" / batch_name
    batch_root.mkdir(parents=True, exist_ok=True)

    force = bool(config.get("force"))
    prepared_models: List[str] = []
    skipped_models: List[Dict[str, str]] = []

    for model_config in config.get("models", []):
        model_name = model_config["name"]
        prompt_strategy = model_config.get("prompt_strategy", "single_query")
        if prompt_strategy != "single_query":
            skipped_models.append({"model": model_name, "reason": "prompt_strategy_not_supported"})
            continue
        spec = BATCH_MODEL_SPECS.get(model_name)
        if spec is None:
            skipped_models.append(
                {"model": model_name, "reason": "provider_not_supported_for_batch"}
            )
            continue

        output_name = model_config.get("output_name") or f"{model_name}.json"
        output_path = output_dir / output_name
        if should_skip_model_output(output_path, expected_images=len(image_paths), force=force):
            skipped_models.append({"model": model_name, "reason": "aggregate_already_complete"})
            continue

        provider_dir = batch_root / spec.provider / model_name
        provider_dir.mkdir(parents=True, exist_ok=True)

        manifest: Dict[str, Any] = {
            "provider": spec.provider,
            "model": model_name,
            "model_id": _model_id(spec),
            "strategy": prompt_strategy,
            "labels": labels,
            "output_name": output_name,
            "images_dir": str(images_dir),
            "output_dir": str(output_dir),
            "per_image_labels": bool(per_image_labels),
            "per_image_labels_file": per_image_labels_path,
            "all_images": [],
            "requests": {},
            "batch": {},
        }
        openai_lines: List[Dict[str, Any]] = []
        anthropic_requests: List[Dict[str, Any]] = []
        gemini_lines: List[Dict[str, Any]] = []

        for image_path in image_paths:
            image_key = image_path.name
            labels_for_image = per_image_labels.get(image_key, labels)
            if not labels_for_image:
                raise ValueError(f"No labels available for image {image_key}.")

            per_image_path = output_dir / "per_image" / model_name / f"{image_path.stem}.json"
            manifest["all_images"].append(
                {
                    "image": str(image_path),
                    "image_relative": str(image_path.relative_to(images_dir)),
                    "labels": labels_for_image,
                    "per_image_path": str(per_image_path),
                }
            )
            if per_image_path.exists() and not force:
                continue

            custom_id = _custom_id(model_name, image_path.stem)
            manifest["requests"][custom_id] = {
                "image": str(image_path),
                "image_relative": str(image_path.relative_to(images_dir)),
                "labels": labels_for_image,
                "per_image_path": str(per_image_path),
            }

            if spec.provider == "openai":
                openai_lines.append(
                    build_openai_batch_line(custom_id, spec, image_path, labels_for_image)
                )
            elif spec.provider == "anthropic":
                anthropic_requests.append(
                    build_anthropic_batch_request(custom_id, spec, image_path, labels_for_image)
                )
            elif spec.provider == "gemini":
                gemini_lines.append(
                    build_gemini_batch_line(custom_id, spec, image_path, labels_for_image)
                )

        if not manifest["requests"]:
            skipped_models.append({"model": model_name, "reason": "all_per_image_cached"})
            continue

        if openai_lines:
            _write_jsonl(provider_dir / "input.jsonl", openai_lines)
        if anthropic_requests:
            (provider_dir / "requests.json").write_text(
                json.dumps({"requests": anthropic_requests}, indent=2),
                encoding="utf-8",
            )
        if gemini_lines:
            _write_jsonl(provider_dir / "input.jsonl", gemini_lines)

        (provider_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        prepared_models.append(model_name)

    summary = {
        "batch_root": str(batch_root),
        "prepared_models": prepared_models,
        "skipped_models": skipped_models,
    }
    (batch_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return str(value)


def extract_openai_response_text(response_body: Dict[str, Any]) -> str:
    output_text = response_body.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    pieces: List[str] = []
    for item in response_body.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict):
                text = content.get("text")
                if isinstance(text, str):
                    pieces.append(text)
    return "\n".join(pieces)


def extract_anthropic_message_text(message: Dict[str, Any]) -> str:
    pieces: List[str] = []
    for content in message.get("content", []) or []:
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                pieces.append(text)
    return "\n".join(pieces)


def extract_gemini_response_text(response: Dict[str, Any]) -> str:
    pieces: List[str] = []
    for candidate in response.get("candidates", []) or []:
        content = candidate.get("content") if isinstance(candidate, dict) else None
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []) or []:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    pieces.append(text)
    return "\n".join(pieces)


def collect_batch_results(batch_root: Path) -> Dict[str, Any]:
    batch_root = Path(batch_root)
    collected_models: List[str] = []
    skipped_models: List[Dict[str, str]] = []

    for manifest_path in sorted(batch_root.glob("*/*/manifest.json")):
        model_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        output_path = model_dir / "output.jsonl"
        if not output_path.exists():
            skipped_models.append(
                {"model": manifest["model"], "reason": "output_jsonl_missing"}
            )
            continue

        result_by_custom_id = _load_provider_results(manifest["provider"], output_path)
        output_dir = Path(manifest["output_dir"])
        images_dir = Path(manifest["images_dir"])

        for custom_id, request_meta in manifest["requests"].items():
            result_payload = result_by_custom_id.get(custom_id)
            if result_payload is None:
                continue
            image_path = Path(request_meta["image"])
            with Image.open(image_path) as image:
                image_size = image.size
            raw_text = result_payload["text"]
            coordinate_mode = _coordinate_mode_for_provider(manifest["provider"])
            parse_image_size = None if coordinate_mode.startswith("normalized_1000") else image_size
            detections = parse_detections_from_text(raw_text, image_size=parse_image_size)
            per_image_path = Path(request_meta["per_image_path"])
            per_image_path.parent.mkdir(parents=True, exist_ok=True)
            per_image_payload = {
                "model": manifest["model"],
                "strategy": manifest["strategy"],
                "image": str(image_path),
                "labels": request_meta["labels"],
                "detections": detections,
                "latency_sec": None,
                "raw_response": result_payload["raw_response"],
                "batch": {
                    "provider": manifest["provider"],
                    "batch_id": _batch_reference(manifest),
                    "custom_id": custom_id,
                },
            }
            per_image_path.write_text(json.dumps(per_image_payload, indent=2), encoding="utf-8")

        _normalize_per_image_cache_boxes(manifest)
        aggregate_results = _aggregate_from_per_image_cache(manifest)
        if not aggregate_results:
            skipped_models.append({"model": manifest["model"], "reason": "no_results_collected"})
            continue

        aggregate_payload = {
            "model": manifest["model"],
            "strategy": manifest["strategy"],
            "labels": manifest["labels"],
            "per_image_labels": manifest["per_image_labels"],
            "per_image_labels_file": manifest["per_image_labels_file"],
            "images_dir": str(images_dir),
            "num_images": len(aggregate_results),
            "results": aggregate_results,
            "summary": {
                "wall_time_sec": None,
                "total_latency_sec": None,
                "avg_latency_sec": None,
                "batch_provider": manifest["provider"],
                "batch_id": _batch_reference(manifest),
            },
        }
        (output_dir / manifest["output_name"]).write_text(
            json.dumps(aggregate_payload, indent=2),
            encoding="utf-8",
        )
        collected_models.append(manifest["model"])

    return {"collected_models": collected_models, "skipped_models": skipped_models}


def _coordinate_mode_for_provider(provider: str) -> str:
    if provider == "gemini":
        return "normalized_1000_yxyx"
    return "pixel"


def _normalize_per_image_cache_boxes(manifest: Dict[str, Any]) -> None:
    coordinate_mode = _coordinate_mode_for_provider(manifest["provider"])
    all_images = manifest.get("all_images") or manifest.get("requests", {}).values()
    for image_meta in all_images:
        per_image_path = Path(image_meta["per_image_path"])
        if not per_image_path.exists():
            continue

        image_path = Path(image_meta["image"])
        with Image.open(image_path) as image:
            image_size = image.size

        cached_payload = json.loads(per_image_path.read_text(encoding="utf-8"))
        detections = normalize_detections_to_image(
            cached_payload.get("detections", []),
            image_size=image_size,
            coordinate_mode=coordinate_mode,
        )
        allowed_labels = set(cached_payload.get("labels") or image_meta.get("labels", []))
        cached_payload["detections"] = [
            detection
            for detection in detections
            if detection.get("label") in allowed_labels
        ]
        per_image_path.write_text(
            json.dumps(cached_payload, indent=2),
            encoding="utf-8",
        )


def _aggregate_from_per_image_cache(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    all_images = manifest.get("all_images") or manifest.get("requests", {}).values()
    for image_meta in all_images:
        per_image_path = Path(image_meta["per_image_path"])
        if not per_image_path.exists():
            continue
        cached_payload = json.loads(per_image_path.read_text(encoding="utf-8"))
        results.append(
            {
                "image": image_meta["image_relative"],
                "detections": cached_payload.get("detections", []),
                "latency_sec": cached_payload.get("latency_sec"),
                "raw_response": cached_payload.get("raw_response"),
                "labels_used": cached_payload.get("labels", image_meta.get("labels", [])),
                "batch_custom_id": cached_payload.get("batch", {}).get("custom_id"),
            }
        )
    return results


def _batch_reference(manifest: Dict[str, Any]) -> Optional[str]:
    batch = manifest.get("batch", {})
    return batch.get("id") or batch.get("name")


def _batch_is_complete(manifest: Dict[str, Any]) -> bool:
    batch = manifest.get("batch", {})
    provider = manifest.get("provider")
    if provider == "openai":
        return batch.get("status") == "completed" and bool(batch.get("output_file_id"))
    if provider == "anthropic":
        return batch.get("processing_status") == "ended"
    if provider == "gemini":
        return batch.get("state") == "JOB_STATE_SUCCEEDED"
    return False


def submit_batches(
    batch_root: Path,
    provider_submitter=None,
) -> Dict[str, Any]:
    batch_root = Path(batch_root)
    submitted_models: List[str] = []
    skipped_models: List[Dict[str, str]] = []
    submitter = provider_submitter or _submit_provider_batch

    for manifest_path in sorted(batch_root.glob("*/*/manifest.json")):
        model_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _batch_reference(manifest):
            skipped_models.append({"model": manifest["model"], "reason": "already_submitted"})
            continue
        batch_info = _json_safe(submitter(manifest["provider"], model_dir, manifest))
        if "id" not in batch_info and batch_info.get("name"):
            batch_info["id"] = batch_info["name"]
        manifest["batch"] = {
            **manifest.get("batch", {}),
            **batch_info,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        submitted_models.append(manifest["model"])

    return {"submitted_models": submitted_models, "skipped_models": skipped_models}


def status_batches(batch_root: Path, provider_status_getter=None) -> Dict[str, Any]:
    batch_root = Path(batch_root)
    getter = provider_status_getter or _get_provider_batch_status
    statuses: List[Dict[str, Any]] = []

    for manifest_path in sorted(batch_root.glob("*/*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        batch_id = _batch_reference(manifest)
        if not batch_id:
            statuses.append({"model": manifest["model"], "status": "not_submitted"})
            continue
        status = _json_safe(getter(manifest["provider"], batch_id, manifest))
        manifest["batch"] = {**manifest.get("batch", {}), **status}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        statuses.append({**status, "model": manifest["model"]})
    return {"statuses": statuses}


def download_batch_outputs(batch_root: Path, provider_downloader=None) -> Dict[str, Any]:
    batch_root = Path(batch_root)
    downloader = provider_downloader or _download_provider_batch_output
    downloaded_models: List[str] = []
    skipped_models: List[Dict[str, str]] = []

    for manifest_path in sorted(batch_root.glob("*/*/manifest.json")):
        model_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (model_dir / "output.jsonl").exists():
            skipped_models.append({"model": manifest["model"], "reason": "output_already_exists"})
            continue
        batch_id = _batch_reference(manifest)
        if not batch_id:
            skipped_models.append({"model": manifest["model"], "reason": "not_submitted"})
            continue
        if not _batch_is_complete(manifest):
            skipped_models.append({"model": manifest["model"], "reason": "batch_not_complete"})
            continue
        output_text = downloader(manifest["provider"], model_dir, manifest)
        (model_dir / "output.jsonl").write_text(output_text, encoding="utf-8")
        downloaded_models.append(manifest["model"])
    return {"downloaded_models": downloaded_models, "skipped_models": skipped_models}


def _load_provider_results(provider: str, output_path: Path) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        custom_id = _result_custom_id(provider, payload)
        if not custom_id:
            continue
        text, raw_response = _result_text_and_raw(provider, payload)
        results[custom_id] = {"text": text, "raw_response": raw_response}
    return results


def _result_custom_id(provider: str, payload: Dict[str, Any]) -> Optional[str]:
    if provider == "gemini":
        return payload.get("key") or payload.get("custom_id")
    return payload.get("custom_id")


def _result_text_and_raw(provider: str, payload: Dict[str, Any]) -> tuple[str, Any]:
    if provider == "openai":
        body = (payload.get("response") or {}).get("body") or {}
        return extract_openai_response_text(body), payload
    if provider == "anthropic":
        result = payload.get("result") or {}
        message = result.get("message") or result.get("response") or payload.get("message") or {}
        return extract_anthropic_message_text(message), payload
    if provider == "gemini":
        response = payload.get("response") or payload.get("GenerateContentResponse") or payload
        return extract_gemini_response_text(response), payload
    return "", payload


def _submit_provider_batch(provider: str, model_dir: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    if provider == "openai":
        return _submit_openai_batch(model_dir)
    if provider == "anthropic":
        return _submit_anthropic_batch(model_dir)
    if provider == "gemini":
        return _submit_gemini_batch(model_dir, manifest)
    raise ValueError(f"Unsupported batch provider: {provider}")


def _get_provider_batch_status(
    provider: str,
    batch_id: str,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    if provider == "openai":
        return _get_openai_batch_status(batch_id)
    if provider == "anthropic":
        return _get_anthropic_batch_status(batch_id)
    if provider == "gemini":
        return _get_gemini_batch_status(batch_id)
    raise ValueError(f"Unsupported batch provider: {provider}")


def _download_provider_batch_output(
    provider: str,
    model_dir: Path,
    manifest: Dict[str, Any],
) -> str:
    if provider == "openai":
        return _download_openai_batch_output(manifest)
    if provider == "anthropic":
        return _download_anthropic_batch_output(manifest)
    if provider == "gemini":
        return _download_gemini_batch_output(manifest)
    raise ValueError(f"Unsupported batch provider: {provider}")


def _submit_openai_batch(model_dir: Path) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with (model_dir / "input.jsonl").open("rb") as fh:
        input_file = client.files.create(file=fh, purpose="batch")
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    return _serialize_batch_object(batch)


def _get_openai_batch_status(batch_id: str) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _serialize_batch_object(client.batches.retrieve(batch_id))


def _download_openai_batch_output(manifest: Dict[str, Any]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    output_file_id = manifest.get("batch", {}).get("output_file_id")
    if not output_file_id:
        batch = client.batches.retrieve(manifest["batch"]["id"])
        output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise RuntimeError("OpenAI batch has no output_file_id yet.")
    content = client.files.content(output_file_id)
    if hasattr(content, "text"):
        return content.text
    if hasattr(content, "read"):
        raw = content.read()
        return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    return str(content)


def _submit_anthropic_batch(model_dir: Path) -> Dict[str, Any]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    payload = json.loads((model_dir / "requests.json").read_text(encoding="utf-8"))
    batch = client.messages.batches.create(requests=payload["requests"])
    return _serialize_batch_object(batch)


def _get_anthropic_batch_status(batch_id: str) -> Dict[str, Any]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _serialize_batch_object(client.messages.batches.retrieve(batch_id))


def _download_anthropic_batch_output(manifest: Dict[str, Any]) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    rows = []
    for item in client.messages.batches.results(manifest["batch"]["id"]):
        rows.append(json.dumps(_serialize_batch_object(item), ensure_ascii=False))
    return "\n".join(rows) + "\n"


def _submit_gemini_batch(model_dir: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    genai = _import_google_genai()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    uploaded = client.files.upload(
        file=str(model_dir / "input.jsonl"),
        config={"mime_type": "application/jsonl"},
    )
    batch = client.batches.create(
        model=manifest["model_id"],
        src=uploaded.name,
        config={"display_name": f"{manifest['model']}-lvis-batch"},
    )
    return _serialize_batch_object(batch)


def _get_gemini_batch_status(batch_id: str) -> Dict[str, Any]:
    genai = _import_google_genai()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _serialize_batch_object(client.batches.get(name=batch_id))


def _download_gemini_batch_output(manifest: Dict[str, Any]) -> str:
    genai = _import_google_genai()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    batch_ref = _batch_reference(manifest)
    if not batch_ref:
        raise RuntimeError("Gemini batch has no id/name yet.")
    batch = client.batches.get(name=batch_ref)
    batch_payload = _serialize_batch_object(batch)
    dest = batch_payload.get("dest") or batch_payload.get("destination") or {}
    file_name = dest.get("file_name") or dest.get("fileName")
    if not file_name:
        raise RuntimeError("Gemini batch has no output file yet.")
    content = client.files.download(file=file_name)
    if isinstance(content, bytes):
        return content.decode("utf-8")
    return str(content)


def _import_google_genai():
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "Gemini Batch requires the 'google-genai' package. "
            "Install it with 'pip install google-genai'."
        ) from exc
    return genai


def _serialize_batch_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return _json_safe(value)
    for attr in ("model_dump", "to_dict", "dict"):
        if hasattr(value, attr):
            try:
                dumped = getattr(value, attr)()
            except Exception:  # pragma: no cover
                continue
            if isinstance(dumped, dict):
                return _json_safe(dumped)
    if hasattr(value, "__dict__"):
        return _json_safe(dict(value.__dict__))
    return {"value": str(value)}
