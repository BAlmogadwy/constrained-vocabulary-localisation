# Copyright (c) 2024.
# SPDX-License-Identifier: MIT
"""YOLO-World detector wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None

from .base import DetectionResult, TraditionalDetector


DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "yolov8x-world.pt"


@dataclass
class YOLOWorldConfig:
    weights_path: Path = DEFAULT_WEIGHTS
    confidence: float = 0.001  # score-floor audit (was 0.25)
    iou: float = 0.45
    use_half: bool = False


class YOLOWorldDetector(TraditionalDetector):
    model_name = "yolo-world"

    def __init__(
        self,
        config: YOLOWorldConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or YOLOWorldConfig()
        super().__init__(device=device)

    def _load_model(self):
        if YOLO is None:
            raise ImportError(
                "ultralytics package is required for YOLO-World inference. "
                "Install it with `pip install ultralytics`."
            )
        weights_path = self.config.weights_path
        if not weights_path.exists():
            raise FileNotFoundError(
                f"YOLO-World weights not found at {weights_path}. "
                "Run `python models/traditional/download_weights.py --models yolo_world` first."
            )
        model = YOLO(str(weights_path))
        model, resolved_device = self._ensure_device(model, preferred=self.device)
        self.runtime_device = resolved_device
        if self.config.use_half and torch and resolved_device and str(resolved_device).startswith("cuda"):
            model.model.half()
        return model

    def predict(self, image_path: str | Path, labels: Iterable[str]) -> List[DetectionResult]:
        prompts = self._preprocess_labels(labels)
        image_path = Path(image_path)
        if prompts:
            # YOLO-World expects dynamic categories to be registered through set_classes
            self.model.set_classes(list(prompts))
        results = self.model.predict(
            source=str(image_path),
            conf=self.config.confidence,
            iou=self.config.iou,
            verbose=False,
        )

        detections: List[DetectionResult] = []
        for result in results:
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.tolist()
            confs = boxes.conf.tolist()
            classes = boxes.cls.tolist()
            for box, score, cls_idx in zip(xyxy, confs, classes):
                cls_int = int(cls_idx)
                if isinstance(names, dict):
                    label = names.get(cls_int, str(cls_int))
                else:
                    try:
                        label = names[cls_int]
                    except (TypeError, IndexError):
                        label = str(cls_int)
                detections.append(
                    DetectionResult(
                        label=label,
                        score=float(score),
                        box=self._to_numpy(box),
                    )
                )
        return detections
