# Copyright (c) 2024.
# SPDX-License-Identifier: MIT
"""Grounding DINO detector wrapper using Hugging Face Transformers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from .base import DetectionResult, TraditionalDetector


MODEL_ID = "IDEA-Research/grounding-dino-base"


@dataclass
class GroundingDINOConfig:
    model_id: str = MODEL_ID
    box_threshold: float = 0.001  # score-floor audit (was 0.25); text_threshold kept at 0.25
    text_threshold: float = 0.25
    max_detections: int | None = None


class GroundingDINODetector(TraditionalDetector):
    model_name = "grounding-dino"

    def __init__(
        self,
        config: GroundingDINOConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or GroundingDINOConfig()
        self.processor: AutoProcessor | None = None
        self.runtime_device: str | None = None
        super().__init__(device=device)

    def _load_model(self):
        model = GroundingDinoForObjectDetection.from_pretrained(self.config.model_id)
        processor = AutoProcessor.from_pretrained(self.config.model_id)
        model, resolved_device = self._ensure_device(model, preferred=self.device)
        self.processor = processor
        self.runtime_device = resolved_device
        return model

    def predict(self, image_path: str | Path, labels: Iterable[str]) -> List[DetectionResult]:
        image = self._load_image(Path(image_path))
        prompts = self._preprocess_labels(labels)
        if not prompts:
            raise ValueError("GroundingDINO requires at least one text label for inference.")

        text_prompt = ". ".join(prompts)
        if self.processor is None or self.runtime_device is None:
            raise RuntimeError("GroundingDINO model is not initialized correctly.")
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.runtime_device)
        outputs = self.model(**inputs)  # type: ignore[operator]

        target_sizes = torch.tensor([image.size[::-1]], device=self.runtime_device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.get("input_ids"),
            threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=target_sizes,
        )[0]

        detections: List[DetectionResult] = []
        boxes = results["boxes"].detach().cpu().tolist()
        scores = results["scores"].detach().cpu().tolist()
        labels_out = results.get("labels") or results.get("text_labels") or []
        text_labels = results.get("text_labels") or []

        def normalize(text: str) -> str:
            return "".join(ch for ch in text.lower() if ch.isalnum())

        for idx, (box, score) in enumerate(zip(boxes, scores)):
            if self.config.max_detections and len(detections) >= self.config.max_detections:
                break
            raw_label = None
            if isinstance(labels_out, torch.Tensor):
                label_idx = int(labels_out[idx])
                raw_label = prompts[label_idx] if label_idx < len(prompts) else str(label_idx)
            elif isinstance(labels_out, list) and idx < len(labels_out):
                candidate = labels_out[idx]
                if isinstance(candidate, (int, float)):
                    candidate_idx = int(candidate)
                    raw_label = prompts[candidate_idx] if candidate_idx < len(prompts) else str(candidate_idx)
                elif isinstance(candidate, str):
                    raw_label = candidate
            if raw_label is None and idx < len(text_labels):
                raw_label = text_labels[idx]

            label = raw_label or "object"
            matched = next((p for p in prompts if normalize(p) == normalize(label)), None)
            detections.append(
                DetectionResult(label=matched or label, score=float(score), box=self._to_numpy(box))
            )
        return detections
