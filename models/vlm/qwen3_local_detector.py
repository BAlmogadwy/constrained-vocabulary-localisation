import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor

try:
    from transformers import AutoModelForImageTextToText  # type: ignore
except ImportError:
    AutoModelForImageTextToText = None  # type: ignore

try:
    from transformers import AutoModelForVision2Seq  # type: ignore
except ImportError:
    AutoModelForVision2Seq = None  # type: ignore

from transformers import AutoModelForCausalLM


class Qwen3LocalDetector:
    """Local Qwen3-VL detector using Transformers."""

    def __init__(self) -> None:
        model_id = os.getenv("QWEN3_LOCAL_MODEL", "qwen/Qwen3-VL-8B-Instruct")
        local_path = os.getenv("QWEN3_LOCAL_PATH")
        if local_path:
            self.model_source: Any = Path(local_path).expanduser().resolve()
        else:
            candidate = Path(model_id)
            self.model_source = candidate.expanduser().resolve() if candidate.exists() else model_id

        self.device = os.getenv("QWEN3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            self.dtype = torch.float16
            self.device_map: Optional[str] = "auto"
        else:
            self.dtype = torch.float32
            self.device_map = None

        self.max_new_tokens = int(os.getenv("QWEN3_MAX_NEW_TOKENS", "8000"))

        print(f"Loading Qwen3 model {self.model_source} on {self.device} (dtype={self.dtype}).")
        self.processor = AutoProcessor.from_pretrained(self.model_source)
        model_cls = self._select_model_class()
        load_kwargs = {"device_map": self.device_map}
        if self.device_map is None:
            load_kwargs["torch_dtype"] = self.dtype
        else:
            load_kwargs["torch_dtype"] = self.dtype
        try:
            self.model = model_cls.from_pretrained(self.model_source, **load_kwargs)
        except TypeError:
            alt_kwargs = load_kwargs.copy()
            if "torch_dtype" in alt_kwargs:
                alt_kwargs["dtype"] = alt_kwargs.pop("torch_dtype")
            self.model = model_cls.from_pretrained(self.model_source, **alt_kwargs)

        if self.device_map is None:
            self.model.to(self.device)

    def _select_model_class(self):
        config = AutoConfig.from_pretrained(self.model_source)
        model_type = getattr(config, "model_type", "")
        if model_type in {"qwen3_vl"}:
            if AutoModelForImageTextToText is not None:
                return AutoModelForImageTextToText
            if AutoModelForVision2Seq is not None:
                return AutoModelForVision2Seq
            raise RuntimeError(
                "Transformers installation does not provide AutoModelForImageTextToText "
                "or AutoModelForVision2Seq required for Qwen3-VL models."
            )
        return AutoModelForCausalLM

    def predict(
        self,
        image_path: str,
        classes: List[str],
        strategy: str = "single_query",
    ) -> Dict[str, Any]:
        if strategy != "single_query":
            raise ValueError(f"Strategy '{strategy}' not supported for Qwen3LocalDetector.")

        image = Image.open(image_path).convert("RGB")
        prompt = (
            "You are an object detection assistant. "
            "Given the image and the list of target classes, return detections as JSON.\n"
            f"Target classes: {', '.join(classes)}.\n"
            'Output format: [{"box_2d":[x1,y1,x2,y2],"label":"class_name"}...]\n'
            "Use pixel coordinates relative to the input image. Return JSON only."
        )
        raw_text = self._generate(image, prompt)
        detections = self._parse_detections(raw_text)
        return {
            "detections": detections,
            "raw_response": raw_text,
        }

    def _generate(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt_text = None
        if hasattr(self.processor, "apply_chat_template"):
            try:
                prompt_text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt_text = None
        if prompt_text is None:
            prompt_text = f"<image>\n{prompt}\nAssistant:"

        try:
            inputs = self.processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
            )
        except TypeError:
            inputs = self.processor(
                text=[prompt_text],
                images=image,
                return_tensors="pt",
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return decoded.strip()

    @staticmethod
    def _parse_detections(text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        cleaned = Qwen3LocalDetector._strip_markdown(text)
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            return []
        block = match.group(0).replace("\\\"", '"').replace("\n", "").replace("\t", "")
        try:
            data = json.loads(block)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(block)
                if isinstance(data, list):
                    return data
            except (SyntaxError, ValueError):
                pass

        detections: List[Dict[str, Any]] = []
        seen = set()
        pattern = re.compile(
            r"\[\s*([-0-9.,\s]+?)\s*\]\s*,\s*\"label\"\s*:\s*\"([^\"]+)\"",
            re.DOTALL,
        )
        for coords_str, label in pattern.findall(block):
            try:
                coords = [float(value.strip()) for value in coords_str.split(",") if value.strip()]
            except ValueError:
                continue
            if len(coords) != 4:
                continue
            key = (label, tuple(coords))
            if key in seen:
                continue
            seen.add(key)
            detections.append({"box_2d": coords, "label": label})
        return detections

    @staticmethod
    def _strip_markdown(text: str) -> str:
        stripped = text.strip()
        if "```" in stripped:
            lines = [ln for ln in stripped.splitlines() if not ln.strip().startswith("```")]
            stripped = "\n".join(lines).strip()
        if "ASSISTANT:" in stripped:
            stripped = stripped.split("ASSISTANT:", 1)[-1].strip()
        if "[" in stripped:
            stripped = stripped[stripped.index("[") :]
        return stripped
