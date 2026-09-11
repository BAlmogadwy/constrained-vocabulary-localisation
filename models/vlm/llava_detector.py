import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
try:
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
except ImportError:  # older transformers
    LlavaNextProcessor = None  # type: ignore
    LlavaNextForConditionalGeneration = None  # type: ignore
import PIL.Image


class LlavaDetector:
    """LLaVA detector for zero-shot object detection (local inference)."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        forced_device: Optional[str] = None,
        max_new_tokens: int = 16192,
    ) -> None:
        resolved_model = model_id or os.getenv("LLAVA_MODEL_ID", "llava-hf/llava-1.5-7b-hf")
        local_override = os.getenv("LLAVA_LOCAL_PATH")
        if local_override:
            model_source: Any = Path(local_override).expanduser().resolve()
        else:
            candidate_path = Path(resolved_model)
            model_source = candidate_path.expanduser().resolve() if candidate_path.exists() else resolved_model

        requested_device = forced_device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = str(requested_device)
        use_cuda = device.startswith("cuda")
        if use_cuda and device == "cuda":
            torch_dtype = torch.float16
            device_map: Optional[str] = "auto"
        else:
            torch_dtype = torch.float16 if use_cuda else torch.float32
            device_map = None

        print(f"Loading LLaVA model {model_source} on {device} (dtype={torch_dtype}).")

        model_str = str(model_source)
        is_v16 = "v1.6" in model_str or "llava-next" in model_str

        if is_v16 and LlavaNextProcessor is not None and LlavaNextForConditionalGeneration is not None:
            self.processor = LlavaNextProcessor.from_pretrained(model_source)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_source,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        else:
            if is_v16 and LlavaNextProcessor is None:
                raise ImportError(
                    "LlavaNextProcessor is unavailable. Please upgrade transformers (>=4.41) "
                    "or install a version that provides LlavaNext support."
                )
            self.processor = AutoProcessor.from_pretrained(model_source)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_source,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        self.device = device
        self.max_new_tokens = max_new_tokens
        if device_map is None:
            self.model.to(self.device)

    def predict(
        self,
        image_path: str,
        classes: List[str],
        strategy: str = "single_query",
    ) -> Dict[str, Any]:
        if strategy == "single_query":
            return self._predict_single_query(image_path, classes)
        if strategy == "iterative":
            return self._predict_iterative(image_path, classes)
        raise ValueError(f"Unknown strategy: {strategy}")

    def _generate(
        self,
        prompt: str,
        image: PIL.Image.Image,
        max_tokens: Optional[int] = None,
    ) -> str:
        tokens_to_use = max_tokens or self.max_new_tokens
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=tokens_to_use,
            do_sample=False,
        )
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response

    def _predict_single_query(self, image_path: str, classes: List[str]) -> Dict[str, Any]:
        print(f"Predicting objects in {image_path} with classes: {classes} using LLaVA (single query).")
        image = PIL.Image.open(image_path).convert("RGB")
        prompt = (
            "USER: <image>\n"
            "Detect the following objects in the image: "
            f"{', '.join(classes)}. "
            "For each detected object, provide JSON entries in the form "
            '[{"box_2d": [x1, y1, x2, y2], "label": "class_name"}]. '
            "Return only JSON. ASSISTANT:"
        )
        raw_text = self._generate(prompt, image)
        detections = self._parse_detections(raw_text)
        return {
            "detections": detections,
            "raw_response": raw_text,
        }

    def _predict_iterative(self, image_path: str, classes: List[str]) -> Dict[str, Any]:
        print(f"Predicting objects in {image_path} with classes: {classes} using LLaVA (iterative).")
        image = PIL.Image.open(image_path).convert("RGB")

        id_prompt = "USER: <image>\nList all objects you can detect in this image. Respond with comma-separated names only. ASSISTANT:"
        id_text = self._generate(id_prompt, image)
        names_section = id_text.split("ASSISTANT:")[-1] if "ASSISTANT:" in id_text else id_text
        object_names = [name.strip() for name in names_section.split(",") if name.strip()]

        detections: List[Dict[str, Any]] = []
        raw_calls: Dict[str, Any] = {
            "identification": id_text,
            "localization": {},
        }

        for name in object_names:
            if name not in classes:
                continue
            loc_prompt = (
                "USER: <image>\n"
                f"Where is the '{name}' in the image? Provide the bounding box as JSON "
                'in the form {"box_2d": [x1, y1, x2, y2], "label": "name"}. '
                "Return only JSON. ASSISTANT:"
            )
            loc_text = self._generate(loc_prompt, image)
            raw_calls["localization"][name] = loc_text
            parsed = self._parse_detections(loc_text)
            detections.extend(det for det in parsed if det.get("label") == name)

        return {
            "detections": detections,
            "raw_response": raw_calls,
        }

    @staticmethod
    def _parse_detections(text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        cleaned = LlavaDetector._strip_markdown(text)
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            return []
        block = match.group(0)
        block = block.replace('\"', '"')
        block = block.replace("\n", "")
        block = block.replace("\t", "")
        try:
            data = json.loads(block)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError as err:
            try:
                data = ast.literal_eval(block)
                if isinstance(data, list):
                    return data
            except (SyntaxError, ValueError):
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
                if detections:
                    return detections
            print(f"Error parsing LLaVA response: {err}")
        return []

    @staticmethod
    def _strip_markdown(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = []
            for line in stripped.splitlines():
                if line.strip().startswith("```"):
                    continue
                lines.append(line)
            stripped = "\n".join(lines).strip()
        if "ASSISTANT:" in stripped:
            stripped = stripped.split("ASSISTANT:", 1)[-1].strip()
        if "[" in stripped:
            stripped = stripped[stripped.index("[") :]
        return stripped


class LlavaMistralDetector(LlavaDetector):
    """LLaVA v1.6 Mistral 7B detector with optional CPU fallback."""

    def __init__(
        self,
        forced_device: Optional[str] = None,
        max_new_tokens: int = 16192,
    ) -> None:
        model_env = os.getenv("LLAVA_V16_MODEL_ID", "llava-hf/llava-v1.6-mistral-7b-hf")
        super().__init__(
            model_id=model_env,
            forced_device=forced_device,
            max_new_tokens=max_new_tokens,
        )
