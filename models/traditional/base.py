# Copyright (c) 2024.
# SPDX-License-Identifier: MIT
"""Base utilities for traditional detector wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import warnings

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class DetectionResult:
    """Container for a single detection prediction."""

    label: str
    score: float
    box: Sequence[float]  # [xmin, ymin, xmax, ymax] in pixels


class TraditionalDetector:
    """Abstract base class for all traditional detectors used in the benchmark."""

    model_name: str = "abstract-detector"

    def __init__(self, device: str | None = None) -> None:
        self.device = device
        self.runtime_device: str | None = None
        self.model = self._load_model()
        if self.runtime_device is None:
            self.runtime_device = self.device

    def _load_model(self):
        """Subclasses must implement model loading."""
        raise NotImplementedError

    def _load_image(self, image_path: Path) -> Image.Image:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return Image.open(image_path).convert("RGB")

    def _preprocess_labels(self, labels: Iterable[str]) -> List[str]:
        return [label.strip() for label in labels if label.strip()]

    def predict(self, image_path: str | Path, labels: Iterable[str]) -> List[DetectionResult]:
        """Run inference on an image, returning a list of detection results."""
        raise NotImplementedError

    @staticmethod
    def _to_numpy(box: Sequence[float]) -> Sequence[float]:
        arr = np.asarray(box, dtype=float)
        if arr.shape != (4,):
            raise ValueError(f"Expected bounding box with 4 coordinates, got {arr}")
        return arr.tolist()

    def _select_device(self, preferred: str | None = None) -> str:
        if preferred:
            return preferred
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _ensure_device(self, module, preferred: str | None = None) -> Tuple[object, str]:
        device = self._select_device(preferred)
        if torch is None or not hasattr(module, "to"):
            return module, "cpu"

        try:
            module = module.to(device)
            resolved = device
        except RuntimeError as exc:
            if device.startswith("cuda") and "forward compatibility" in str(exc).lower():
                warnings.warn(
                    f"{self.model_name}: CUDA device '{device}' unavailable ({exc}). Falling back to CPU.",
                    RuntimeWarning,
                )
                module = module.to("cpu")
                resolved = "cpu"
            else:
                raise
        return module, resolved
