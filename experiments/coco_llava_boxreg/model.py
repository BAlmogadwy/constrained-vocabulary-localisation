from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:
    from transformers import LlavaForConditionalGeneration  # type: ignore
except ImportError:  # pragma: no cover - defensive fallback
    LlavaForConditionalGeneration = None  # type: ignore

try:
    from transformers import LlavaNextForConditionalGeneration  # type: ignore
except ImportError:  # pragma: no cover - defensive fallback
    LlavaNextForConditionalGeneration = None  # type: ignore

from .utils import resolve_model_source


class CocoLLaVABBoxRegressor(nn.Module):
    """
    Lightweight bounding-box (and optional classification) head over frozen LLaVA visual features.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        head_hidden_dim: int = 512,
        num_classes: int = 80,
        freeze_backbone: bool = True,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.model_source = resolve_model_source(model_id)
        self.freeze_backbone = freeze_backbone
        self.num_classes = num_classes
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_dtype = torch_dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)

        vision_module = self._load_vision_tower()
        self.vision_tower = vision_module.to(self.device, dtype=self.vision_dtype)
        if self.freeze_backbone:
            for param in self.vision_tower.parameters():
                param.requires_grad = False
            self.vision_tower.eval()
        else:
            self.vision_tower.train()

        embed_dim = getattr(self.vision_tower.config, "hidden_size", None)
        if embed_dim is None:
            embed_dim = getattr(self.vision_tower.config, "vision_config", {}).get("hidden_size")  # type: ignore[attr-defined]
        if embed_dim is None:
            raise RuntimeError("Unable to determine vision embedding dimension from the LLaVA vision tower.")
        self.embed_dim = int(embed_dim)

        self.box_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, 4),
        )
        if num_classes > 0:
            self.cls_head = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, head_hidden_dim),
                nn.GELU(),
                nn.Linear(head_hidden_dim, num_classes),
            )
        else:
            self.cls_head = None

    def _load_vision_tower(self) -> nn.Module:
        if LlavaForConditionalGeneration is None:
            raise ImportError(
                "transformers with LLaVA support is required for CocoLLaVABBoxRegressor. "
                "Please install transformers>=4.38."
            )

        model_str = str(self.model_source)
        is_v16 = "v1.6" in model_str or "llava-next" in model_str

        def _resolve_module(llava_model: Any) -> nn.Module:
            if hasattr(llava_model, "get_vision_tower"):
                tower = llava_model.get_vision_tower()
            else:
                tower = getattr(llava_model, "vision_tower", None)
            if tower is None:
                raise RuntimeError("LLaVA model does not expose a vision tower.")
            # Some wrappers store the actual vision module inside .vision_tower
            if hasattr(tower, "vision_tower"):
                vision_module = tower.vision_tower
            else:
                vision_module = tower
            return vision_module

        if is_v16 and LlavaNextForConditionalGeneration is not None:
            backbone = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_source,
                torch_dtype=self.vision_dtype,
                device_map=None,
            )
        else:
            backbone = LlavaForConditionalGeneration.from_pretrained(
                self.model_source,
                torch_dtype=self.vision_dtype,
                device_map=None,
            )
        vision_module = _resolve_module(backbone)
        # Free memory from the full generative stack.
        del backbone
        return vision_module

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        inputs = pixel_values.to(self.device, dtype=self.vision_dtype)
        with torch.set_grad_enabled(not self.freeze_backbone):
            outputs = self.vision_tower(pixel_values=inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return pooled.to(torch.float32)

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encode(pixel_values)
        box_logits = self.box_head(features)
        pred_boxes = torch.sigmoid(box_logits)
        outputs: Dict[str, torch.Tensor] = {"pred_boxes": pred_boxes}
        if self.cls_head is not None:
            outputs["logits"] = self.cls_head(features)
        return outputs
