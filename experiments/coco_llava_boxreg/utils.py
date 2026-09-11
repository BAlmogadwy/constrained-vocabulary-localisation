from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoProcessor

try:
    from transformers import LlavaNextProcessor  # type: ignore
except ImportError:  # pragma: no cover - fallback for older transformers builds
    LlavaNextProcessor = None  # type: ignore


DEFAULT_MODEL_NAME = os.getenv("LLAVA_MODEL_ID", "llava-hf/llava-1.5-7b-hf")


def resolve_model_source(model_id: Optional[str]) -> str:
    """
    Resolve the model identifier/path.

    Priority order:
      1. Explicit argument (if provided)
      2. LLAVA_MODEL_ID env var or DEFAULT_MODEL_NAME
      3. LLAVA_LOCAL_PATH env var if set (has precedence when present)
    """
    candidate = model_id or DEFAULT_MODEL_NAME
    local_override = os.getenv("LLAVA_LOCAL_PATH")
    if local_override:
        return str(Path(local_override).expanduser().resolve())
    candidate_path = Path(candidate).expanduser()
    if candidate_path.exists():
        return str(candidate_path.resolve())
    return candidate


def load_image_processor(model_source: str):
    """
    Load the image processor that matches the requested LLaVA variant.
    """
    model_str = str(model_source)
    is_v16 = "v1.6" in model_str or "llava-next" in model_str
    if is_v16 and LlavaNextProcessor is not None:
        return LlavaNextProcessor.from_pretrained(model_source)
    return AutoProcessor.from_pretrained(model_source)


def default_coco_paths(split: str) -> Tuple[Path, Path]:
    split = split.strip().lower()
    if split not in {"train", "train2017", "val", "val2017"}:
        raise ValueError(f"Unsupported COCO split '{split}'. Expected train/train2017/val/val2017.")
    base = Path("data/raw/coco2017")
    images_dir = base / ("train2017" if "train" in split else "val2017")
    ann_file = base / "annotations" / (
        "instances_train2017.json" if "train" in split else "instances_val2017.json"
    )
    return images_dir, ann_file


def select_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clamp_normalized_boxes(boxes: torch.Tensor) -> torch.Tensor:
    return boxes.clamp(0.0, 1.0)


def denormalize_boxes(boxes: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized xyxy boxes into pixel coordinates.
    image_sizes is expected to be shaped (N, 2) with (width, height).
    """
    widths = image_sizes[:, 0:1]
    heights = image_sizes[:, 1:2]
    scale = torch.cat([widths, heights, widths, heights], dim=1)
    return boxes * scale


def box_iou_xyxy(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU for aligned box pairs in normalized xyxy format.
    """
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_w = torch.clamp(x2 - x1, min=0.0)
    inter_h = torch.clamp(y2 - y1, min=0.0)
    intersection = inter_w * inter_h

    pred_area = torch.clamp(pred_boxes[:, 2] - pred_boxes[:, 0], min=0.0) * torch.clamp(
        pred_boxes[:, 3] - pred_boxes[:, 1],
        min=0.0,
    )
    target_area = torch.clamp(target_boxes[:, 2] - target_boxes[:, 0], min=0.0) * torch.clamp(
        target_boxes[:, 3] - target_boxes[:, 1],
        min=0.0,
    )
    union = pred_area + target_area - intersection
    union = torch.clamp(union, min=1e-6)
    return intersection / union
