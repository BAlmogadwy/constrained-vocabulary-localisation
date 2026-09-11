import ast
import json
import os
import re
from typing import Any, Dict, List, Optional

import PIL.Image
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import BatchEncoding

try:  # Transformers < 4.45
    from transformers import AutoModelForVision2Seq  # type: ignore
except ImportError:  # pragma: no cover
    AutoModelForVision2Seq = None  # type: ignore

try:  # Transformers >= 4.45
    from transformers import AutoModelForImageTextToText  # type: ignore
except ImportError:  # pragma: no cover
    AutoModelForImageTextToText = None  # type: ignore


class LlamaVisionDetector:
    """Meta LLaMA 3.2 Vision 11B detector for zero-shot object detection."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        forced_device: Optional[str] = None,
        max_new_tokens: int = 1024,
    ) -> None:
        model_env = os.getenv("LLAMA_VISION_MODEL", "meta-llama/Llama-3.2-11B-Vision-Instruct")
        self.model_id = model_id or model_env

        requested_device = forced_device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            target_device = torch.device(requested_device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid device specification '{requested_device}'") from exc
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is False.")

        self.device = str(target_device)
        if target_device.type == "cuda":
            bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            self.dtype = torch.bfloat16 if bf16_supported else torch.float16
        else:
            self.dtype = torch.float32
        self.max_new_tokens = min(max_new_tokens, 4096)

        print(f"Loading LLaMA Vision model {self.model_id} on {self.device} (dtype={self.dtype}).")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        self.pad_token_id: Optional[int] = None
        self.eos_token_ids: List[int] = []
        if self.tokenizer is not None:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None and eos_id is not None:
                try:
                    self.tokenizer.pad_token_id = eos_id
                    pad_id = eos_id
                except AttributeError:
                    pad_id = eos_id
            if pad_id is not None:
                self.pad_token_id = pad_id
            if eos_id is not None:
                self.eos_token_ids.append(eos_id)
            try:
                eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if isinstance(eot_id, int) and eot_id not in self.eos_token_ids:
                    self.eos_token_ids.append(eot_id)
            except Exception:
                pass

        try:
            config = AutoConfig.from_pretrained(self.model_id, local_files_only=False)
        except Exception:  # pragma: no cover - offline fallback
            config = None

        def iter_candidate_loaders():
            if config is not None and getattr(config, "vision_config", None) is not None:
                if AutoModelForImageTextToText is not None:
                    yield AutoModelForImageTextToText
                if AutoModelForVision2Seq is not None:
                    yield AutoModelForVision2Seq
            if AutoModelForImageTextToText is not None:
                yield AutoModelForImageTextToText
            if AutoModelForVision2Seq is not None:
                yield AutoModelForVision2Seq
            yield AutoModelForCausalLM

        self.model = None
        last_err: Optional[Exception] = None
        for loader in iter_candidate_loaders():
            try:
                self.model = loader.from_pretrained(
                    self.model_id,
                    dtype=self.dtype,
                    device_map=None,
                )
                break
            except TypeError as exc:
                last_err = exc
                try:
                    self.model = loader.from_pretrained(
                        self.model_id,
                        torch_dtype=self.dtype,
                        device_map=None,
                    )
                    break
                except Exception as inner_exc:  # noqa: BLE001
                    last_err = inner_exc
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue

        if self.model is None:
            raise RuntimeError(f"Unable to load LLaMA Vision model {self.model_id}.") from last_err

        self.model.to(target_device)
        print("LLaMA Vision model loaded successfully.")

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

    def _move_to_device(self, obj: Any) -> Any:
        if isinstance(obj, (BatchEncoding, BatchFeature)):
            return {key: self._move_to_device(value) for key, value in obj.items() if value is not None}
        if isinstance(obj, dict):
            return {key: self._move_to_device(value) for key, value in obj.items() if value is not None}
        if isinstance(obj, torch.Tensor):
            target_dtype = self.dtype if torch.is_floating_point(obj) else obj.dtype
            return obj.to(self.device, dtype=target_dtype)
        return obj

    def _prepare_inputs(self, prompt: str, image: PIL.Image.Image) -> Dict[str, Any]:
        system_prompt = (
            "You are an object detection assistant. "
            "Return JSON only in the format "
            '[{"box_2d":[x1,y1,x2,y2],"label":"class_name"}] using integer pixel coordinates.'
        )
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        chat_payload: Any = None
        chat_text: Optional[str] = None

        if hasattr(self.processor, "apply_chat_template"):
            try:
                chat_payload = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except TypeError:
                try:
                    chat_payload = self.processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_tensors="pt",
                    )
                except Exception:  # noqa: BLE001
                    chat_payload = None
            except Exception:  # noqa: BLE001
                chat_payload = None

        if isinstance(chat_payload, (BatchEncoding, BatchFeature, dict)):
            inputs = self._move_to_device(chat_payload)
        elif isinstance(chat_payload, torch.Tensor):
            inputs = {"input_ids": self._move_to_device(chat_payload)}
        elif isinstance(chat_payload, str):
            chat_text = chat_payload
            inputs = {}
        else:
            inputs = {}

        if isinstance(inputs, dict) and "pixel_values" not in inputs:
            inputs = {}

        if not inputs:
            if chat_text is None:
                try:
                    chat_text = self.processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                except Exception:
                    chat_text = (
                        "<|start_header_id|>user<|end_header_id|>\n"
                        f"{prompt}\n<|image|>\n"
                        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                    )
            try:
                raw_inputs = self.processor(
                    text=[chat_text],
                    images=[image],
                    return_tensors="pt",
                )
            except TypeError:
                raw_inputs = self.processor(
                    text=[chat_text],
                    images=image,
                    return_tensors="pt",
                )
            inputs = self._move_to_device(dict(raw_inputs))

        return inputs

    def _generate(
        self,
        prompt: str,
        image: PIL.Image.Image,
        max_tokens: Optional[int] = None,
        sampling: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        inputs = self._prepare_inputs(prompt, image)
        tokens = max_tokens or self.max_new_tokens
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": tokens,
        }
        if sampling:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": max(0.0, temperature),
                    "top_p": min(max(top_p, 0.0), 1.0),
                }
            )
        else:
            generation_kwargs.update(
                {
                    "do_sample": False,
                    "temperature": 0.0,
                    "top_p": 1.0,
                }
            )
        if self.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = self.pad_token_id
        if self.eos_token_ids:
            generation_kwargs["eos_token_id"] = self.eos_token_ids if len(self.eos_token_ids) > 1 else self.eos_token_ids[0]
        generation = self.model.generate(
            **inputs,
            **generation_kwargs,
        )
        response = self.processor.batch_decode(generation, skip_special_tokens=True)[0]
        return response.strip()

    def _predict_single_query(self, image_path: str, classes: List[str]) -> Dict[str, Any]:
        print(f"Predicting objects in {image_path} with classes: {classes} using LLaMA 3.2 Vision (single query).")
        unique_classes = list(dict.fromkeys(classes))
        raw_image = PIL.Image.open(image_path).convert("RGB")
        width, height = raw_image.size
        size_hint = (
            f"The image resolution is {width}x{height} pixels. "
            "Bounding boxes must use this coordinate space with x1<x2 and y1<y2."
        )

        detections: List[Dict[str, Any]] = []
        raw_calls: Dict[str, Any] = {}

        for name in unique_classes:
            prompt = (
                f"{size_hint} Return JSON bounding boxes for the object "
                f"'{name}' if present in the image. "
                'Use the format [{"box_2d":[x1,y1,x2,y2],"label":"class_name"}]. '
                "Use integer pixel coordinates. Avoid placeholder boxes such as [0,0,0,0]. "
                "If the object is absent, return []. "
                "Do not include explanations."
            )
            response = self._generate(prompt, raw_image, max_tokens=512)
            trimmed = LlamaVisionDetector._trim_to_assistant(response.strip())
            raw_calls[name] = trimmed or response
            parsed = self._parse_detections(trimmed, default_label=name)
            if not parsed:
                norm_prompt = (
                    f"{size_hint} Provide the bounding box for '{name}' using normalized coordinates (values between 0 and 1). "
                    'Return JSON in the format [{"box_norm":[x1,y1,x2,y2],"label":"class_name"}]. '
                    "Ensure x1<x2 and y1<y2. Avoid placeholder values like 0 when the object is absent; instead, return []. "
                    "Do not include explanations."
                )
                norm_response = self._generate(
                    norm_prompt,
                    raw_image,
                    max_tokens=512,
                    sampling=True,
                    temperature=0.2,
                    top_p=0.8,
                )
                norm_trimmed = LlamaVisionDetector._trim_to_assistant(norm_response.strip())
                raw_calls[f"{name}_norm"] = norm_trimmed or norm_response
                parsed = self._parse_normalized_detections(norm_trimmed, width, height, default_label=name)
            if parsed:
                detections.extend(parsed)

        if not detections:
            fallback_prompt = (
                f"{size_hint} Inspect the entire image carefully. Detect every instance of the following classes: "
                f"{', '.join(unique_classes)}. "
                'Return JSON only in the form [{"box_2d":[x1,y1,x2,y2],"label":"class_name"}]. '
                "Coordinates must match the visible object extents (top-left and bottom-right corners). "
                "Provide one entry per visible instance and use the exact class names listed. "
                "Only return [] if none of the requested classes are in the scene."
            )
            fallback_response = self._generate(
                fallback_prompt,
                raw_image,
                max_tokens=1024,
                sampling=True,
                temperature=0.2,
                top_p=0.8,
            )
            fallback_trimmed = LlamaVisionDetector._trim_to_assistant(fallback_response.strip())
            raw_calls["_fallback"] = fallback_trimmed or fallback_response
            fallback_parsed = self._parse_detections(fallback_trimmed)
            seen: set[tuple[str, tuple[float, float, float, float]]] = set()
            for det in fallback_parsed:
                label = det.get("label")
                if not label and len(unique_classes) == 1:
                    label = unique_classes[0]
                if label not in unique_classes:
                    continue
                coords = det.get("box_2d")
                if not isinstance(coords, list) or len(coords) != 4:
                    continue
                try:
                    coords_f = [float(value) for value in coords]
                except (TypeError, ValueError):
                    continue
                key = (label, tuple(coords_f))
                if key in seen:
                    continue
                seen.add(key)
                coords_px = [int(round(value)) for value in coords_f]
                if coords_px[2] <= coords_px[0] or coords_px[3] <= coords_px[1]:
                    continue
                detections.append({"box_2d": coords_px, "label": label})

        return {
            "detections": detections,
            "raw_response": raw_calls,
        }

    def _predict_iterative(self, image_path: str, classes: List[str]) -> Dict[str, Any]:
        print(f"Predicting objects in {image_path} with classes: {classes} using LLaMA 3.2 Vision (iterative).")
        raw_image = PIL.Image.open(image_path).convert("RGB")

        id_prompt = (
            "List all distinct target objects visible in this image as comma-separated names. "
            f"Focus on the following classes: {', '.join(classes)}."
        )
        id_response = self._generate(id_prompt, raw_image, max_tokens=512)
        object_names = [entry.strip() for entry in id_response.split(",") if entry.strip()]

        detections: List[Dict[str, Any]] = []
        raw_calls: Dict[str, Any] = {"identification": id_response, "localization": {}}
        for name in object_names:
            if name not in classes:
                continue
            loc_prompt = (
                f"Provide the bounding box for the object labeled '{name}' as JSON "
                'in the form {"box_2d": [x1, y1, x2, y2], "label": "name"} using pixel coordinates. '
                "Return JSON only."
            )
            loc_response = self._generate(loc_prompt, raw_image, max_tokens=512)
            raw_calls["localization"][name] = loc_response
            detections.extend(det for det in self._parse_detections(loc_response, default_label=name) if det.get("label") == name)
        return {
            "detections": detections,
            "raw_response": raw_calls,
        }

    @staticmethod
    def _parse_detections(response: str, default_label: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            cleaned = response.strip()
            cleaned = LlamaVisionDetector._trim_to_assistant(cleaned)
            json_block = LlamaVisionDetector._extract_json_block(cleaned)
            if not json_block:
                return []
            detections = LlamaVisionDetector._coerce_to_list(json_block, default_label)
            return detections
        except (json.JSONDecodeError, re.error, ValueError) as exc:
            print(f"Error parsing response: {exc}")
        return []

    @staticmethod
    def _trim_to_assistant(text: str) -> str:
        lower = text.lower()
        if "assistant" in lower:
            idx = lower.rfind("assistant")
            return text[idx + len("assistant") :].lstrip(": \n")
        return text

    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        start = text.find("[")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None

    @staticmethod
    def _coerce_to_list(block: str, default_label: Optional[str]) -> List[Dict[str, Any]]:
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            data = ast.literal_eval(block)
        if not isinstance(data, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            label = entry.get("label") or default_label
            if label is None:
                continue
            coords = entry.get("box_2d") or entry.get("bbox") or entry.get("box")
            if isinstance(coords, list) and len(coords) == 4:
                try:
                    coords = [float(value) for value in coords]
                except (TypeError, ValueError):
                    continue
                if coords[2] <= coords[0] or coords[3] <= coords[1]:
                    continue
                coords_int = [int(round(value)) for value in coords]
                if coords_int[2] <= coords_int[0] or coords_int[3] <= coords_int[1]:
                    continue
                normalized.append({"box_2d": coords_int, "label": label})
        return normalized

    @staticmethod
    def _parse_normalized_detections(
        response: str,
        width: int,
        height: int,
        default_label: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        cleaned = response.strip()
        cleaned = LlamaVisionDetector._trim_to_assistant(cleaned)
        json_block = LlamaVisionDetector._extract_json_block(cleaned)
        if not json_block:
            return []
        try:
            data = json.loads(json_block)
        except json.JSONDecodeError:
            data = ast.literal_eval(json_block)
        if not isinstance(data, list):
            return []

        results: List[Dict[str, Any]] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            label = entry.get("label") or default_label
            if label is None:
                continue
            coords = entry.get("box_norm") or entry.get("box_normalized") or entry.get("box_2d")
            if not (isinstance(coords, list) and len(coords) == 4):
                continue
            try:
                coords_f = [float(value) for value in coords]
            except (TypeError, ValueError):
                continue
            if not all(0.0 <= value <= 1.0 for value in coords_f):
                continue
            if coords_f[2] <= coords_f[0] or coords_f[3] <= coords_f[1]:
                continue
            x1 = max(0.0, min(float(width), coords_f[0] * width))
            y1 = max(0.0, min(float(height), coords_f[1] * height))
            x2 = max(0.0, min(float(width), coords_f[2] * width))
            y2 = max(0.0, min(float(height), coords_f[3] * height))
            if x2 <= x1 or y2 <= y1:
                continue
            coords_px = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
            if coords_px[2] <= coords_px[0] or coords_px[3] <= coords_px[1]:
                continue
            results.append({"box_2d": coords_px, "label": label})
        return results
