# Copyright (c) 2024.
# SPDX-License-Identifier: MIT
"""OWL-ViT detector wrapper using Hugging Face Transformers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from .base import DetectionResult, TraditionalDetector


DEFAULT_MODEL_ID = "google/owlvit-base-patch32"


@dataclass
class OWLVitConfig:
    model_id: str = DEFAULT_MODEL_ID
    score_threshold: float = 0.001  # score-floor audit (was 0.2)
    max_detections: int | None = None


class OWLVitDetector(TraditionalDetector):
    model_name = "owl-vit"

    def __init__(
        self,
        config: OWLVitConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or OWLVitConfig()
        self.processor: OwlViTProcessor | None = None
        self.runtime_device: str | None = None
        super().__init__(device=device)

    def _load_model(self):
        processor = OwlViTProcessor.from_pretrained(self.config.model_id)
        model = OwlViTForObjectDetection.from_pretrained(self.config.model_id)
        model, resolved_device = self._ensure_device(model, preferred=self.device)
        self.processor = processor
        self.runtime_device = resolved_device
        return model

    def predict(self, image_path: str | Path, labels: Iterable[str]) -> List[DetectionResult]:
        prompts = self._preprocess_labels(labels)
        if not prompts:
            raise ValueError("OWL-ViT requires at least one text label for inference.")
        if self.processor is None or self.runtime_device is None:
            raise RuntimeError("OWL-ViT model is not initialized correctly.")

        image = self._load_image(Path(image_path))
        text_queries = [[label] for label in prompts]
        inputs = self.processor(text=text_queries, images=image, return_tensors="pt").to(self.runtime_device)
        outputs = self.model(**inputs)  # type: ignore[operator]

        target_sizes = torch.tensor([image.size[::-1]], device=self.runtime_device)
        processed = self.processor.post_process_object_detection(
            outputs,
            threshold=self.config.score_threshold,
            target_sizes=target_sizes,
        )[0]

        boxes = processed["boxes"].detach().cpu().tolist()
        scores = processed["scores"].detach().cpu().tolist()
        labels_out = processed["labels"].detach().cpu().tolist()

        detections: List[DetectionResult] = []
        for idx, (box, score, label_idx) in enumerate(zip(boxes, scores, labels_out)):
            if score < self.config.score_threshold:
                continue
            if self.config.max_detections and len(detections) >= self.config.max_detections:
                break
            label_int = int(label_idx)
            label = prompts[label_int] if label_int < len(prompts) else f"label_{label_int}"
            detections.append(DetectionResult(label=label, score=float(score), box=self._to_numpy(box)))
        return detections
