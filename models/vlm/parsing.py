"""Shared parsing helpers for VLM object detection responses."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


JSON_KEYS_WITH_DETECTIONS = ("detections", "objects", "results", "predictions")


def strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if "```" not in stripped:
        return stripped
    lines = []
    for line in stripped.splitlines():
        if line.strip().startswith("```"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _decode_json_candidates(text: str) -> Iterable[Any]:
    decoder = json.JSONDecoder()
    cleaned = strip_markdown_fences(text)
    for index, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        yield value


def _coerce_label(entry: Dict[str, Any]) -> Optional[str]:
    label = (
        entry.get("label")
        or entry.get("class_name")
        or entry.get("class")
        or entry.get("category")
        or entry.get("name")
    )
    if label is None:
        return None
    label = str(label).strip()
    return label or None


def _coerce_box(raw_box: Any) -> Optional[List[float]]:
    if raw_box is None:
        return None
    if isinstance(raw_box, dict):
        values = [raw_box.get(key) for key in ("x1", "y1", "x2", "y2")]
    elif isinstance(raw_box, Sequence) and not isinstance(raw_box, (str, bytes)):
        values = list(raw_box)
    else:
        return None
    if len(values) != 4:
        return None
    try:
        coords = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return None
    return coords


def normalize_detection(entry: Any, image_size: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None
    label = _coerce_label(entry)
    if not label:
        return None
    box = _coerce_box(
        entry.get("box_2d")
        or entry.get("bbox_2d")
        or entry.get("bbox")
        or entry.get("box")
        or entry.get("coordinates")
    )
    if box is None:
        return None
    if image_size:
        box = _scale_normalized_box_if_needed(box, image_size)
    detection: Dict[str, Any] = {"box_2d": box, "label": label}
    score = entry.get("score", entry.get("confidence"))
    if score is not None:
        try:
            parsed_score = float(score)
        except (TypeError, ValueError):
            pass
        else:
            if parsed_score <= 0:
                return None
            detection["score"] = parsed_score
    return detection


def normalize_detections_to_image(
    detections: Iterable[Dict[str, Any]],
    image_size: Tuple[int, int],
    coordinate_mode: str = "pixel",
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        box = _coerce_box(detection.get("box_2d") or detection.get("bbox") or detection.get("box"))
        if box is None:
            continue
        transformed = _transform_box_for_mode(box, image_size, coordinate_mode)
        clipped = _clip_box_to_image(transformed, image_size)
        if clipped is None:
            continue
        normalized_detection = dict(detection)
        normalized_detection["box_2d"] = clipped
        normalized.append(normalized_detection)
    return normalized


def _scale_normalized_box_if_needed(box: List[float], image_size: Tuple[int, int]) -> List[float]:
    width, height = image_size
    if width <= 0 or height <= 0:
        return box
    if all(0.0 <= value <= 1.0 for value in box):
        x1, y1, x2, y2 = box
        return [x1 * width, y1 * height, x2 * width, y2 * height]
    return box


def _transform_box_for_mode(
    box: List[float],
    image_size: Tuple[int, int],
    coordinate_mode: str,
) -> List[float]:
    width, height = image_size
    if coordinate_mode == "normalized_1000_xyxy":
        x1, y1, x2, y2 = box
        return [x1 * width / 1000.0, y1 * height / 1000.0, x2 * width / 1000.0, y2 * height / 1000.0]
    if coordinate_mode == "normalized_1000_yxyx":
        y1, x1, y2, x2 = box
        return [x1 * width / 1000.0, y1 * height / 1000.0, x2 * width / 1000.0, y2 * height / 1000.0]
    return box


def _clip_box_to_image(box: List[float], image_size: Tuple[int, int]) -> Optional[List[float]]:
    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = box
    clipped = [
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def _extract_detection_items(payload: Any) -> List[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in JSON_KEYS_WITH_DETECTIONS:
            items = payload.get(key)
            if isinstance(items, list):
                return items
    return []


def parse_detections_from_text(
    text: Optional[str],
    image_size: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    if not text:
        return []
    fallback_detections: List[Dict[str, Any]] = []
    for candidate in _decode_json_candidates(text):
        single_detection = normalize_detection(candidate, image_size=image_size)
        if single_detection is not None:
            fallback_detections.append(single_detection)
            continue
        items = _extract_detection_items(candidate)
        detections = [
            detection
            for detection in (normalize_detection(item, image_size=image_size) for item in items)
            if detection is not None
        ]
        if detections or (isinstance(candidate, list) and items == []):
            return detections
    return fallback_detections


def extract_openai_compatible_text(response_json: Dict[str, Any]) -> str:
    try:
        content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    pieces.append(text)
            elif isinstance(item, str):
                pieces.append(item)
        return "\n".join(pieces)
    return ""


def parse_detections_from_openai_response(
    response_json: Dict[str, Any],
    image_size: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    return parse_detections_from_text(
        extract_openai_compatible_text(response_json),
        image_size=image_size,
    )
