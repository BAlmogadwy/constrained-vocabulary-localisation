import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import BatchEncoding

try:
    from transformers import AutoModelForImageTextToText  # type: ignore
except ImportError:  # pragma: no cover
    AutoModelForImageTextToText = None  # type: ignore


def _select_model_cls(model_id: str):
    config = AutoConfig.from_pretrained(model_id)
    model_type = getattr(config, "model_type", "")
    multimodal = {"paligemma2", "paligemma"}
    if model_type in multimodal:
        if AutoModelForImageTextToText is not None:
            return AutoModelForImageTextToText
        if AutoModelForVision2Seq is not None:
            return AutoModelForVision2Seq
        raise RuntimeError(
            "This transformers build lacks AutoModelForImageTextToText / AutoModelForVision2Seq needed for PaLI-Gemma models."
        )
    return AutoModelForCausalLM


def _move_to_device(obj: Any, device: str, dtype: Optional[torch.dtype] = None) -> Any:
    if isinstance(obj, (BatchEncoding, BatchFeature)):
        return {k: _move_to_device(v, device, dtype) for k, v in obj.items() if v is not None}
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device, dtype) for k, v in obj.items() if v is not None}
    if isinstance(obj, torch.Tensor):
        if dtype is not None and obj.dtype in (torch.float16, torch.float32, torch.bfloat16):
            return obj.to(device=device, dtype=dtype)
        return obj.to(device=device)
    return obj


class PaliGemmaLocalDetector:
    """Local PaLI-Gemma 2 10B detector using transformers."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        forced_device: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> None:
        env_model = os.getenv("PALIGEMMA_LOCAL_MODEL", "google/paligemma2-10b-mix-448")
        path_override = os.getenv("PALIGEMMA_LOCAL_PATH")
        if path_override:
            self.model_id = str(Path(path_override).expanduser().resolve())
        else:
            self.model_id = model_id or env_model

        requested_device = forced_device or os.getenv("PALIGEMMA_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        try:
            target_device = torch.device(requested_device)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f"Invalid device specification '{requested_device}'") from exc
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

        self.device = str(target_device)
        self.dtype = torch.float16 if target_device.type == "cuda" else torch.float32
        default_tokens = int(os.getenv("PALIGEMMA_MAX_NEW_TOKENS", "512"))
        self.max_new_tokens = max_new_tokens or default_tokens

        print(f"Loading PaLI-Gemma model {self.model_id} on {self.device} (dtype={self.dtype}).")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load PaLI-Gemma processor. Ensure you have accepted the model license and executed "
                "'huggingface-cli login --token <HF_TOKEN>' on this machine."
            ) from exc

        model_cls = _select_model_cls(self.model_id)
        try:
            self.model = model_cls.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                device_map="auto" if target_device.type == "cuda" else None,
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to load PaLI-Gemma weights. Confirm the checkpoint is available locally or your HF token "
                "has access to google/paligemma2-10b-mix-448."
            ) from exc

        if target_device.type != "cuda":
            self.model.to(target_device)

        tokenizer = getattr(self.processor, "tokenizer", None)
        self.pad_token_id: Optional[int] = None
        self.eos_token_ids: List[int] = []
        if tokenizer is not None:
            pad_id = getattr(tokenizer, "pad_token_id", None)
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if pad_id is None and eos_id is not None:
                try:
                    tokenizer.pad_token_id = eos_id
                    pad_id = eos_id
                except AttributeError:
                    pad_id = eos_id
            self.pad_token_id = pad_id
            if eos_id is not None:
                self.eos_token_ids.append(eos_id)
            try:
                eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if isinstance(eot_id, int) and eot_id not in self.eos_token_ids:
                    self.eos_token_ids.append(eot_id)
            except Exception:
                pass

        print("PaLI-Gemma model ready.")

    def predict(
        self,
        image_path: str,
        classes: List[str],
        strategy: str = "single_query",
    ) -> Dict[str, Any]:
        if strategy != "single_query":
            raise ValueError(f"Strategy '{strategy}' not supported for PaLI-Gemma detector.")

        image = Image.open(image_path).convert("RGB")
        prompt = (
            "Detect the following objects in the image: "
            f"{', '.join(classes)}. "
            "Return JSON only in the format "
            '[{"box_2d":[x1,y1,x2,y2],"label":"class_name"}] using integer pixel coordinates.'
        )
        attempts: List[Dict[str, Any]] = []

        raw_text = self._generate(prompt, image, tokens=self.max_new_tokens)
        attempts.append({"prompt": prompt, "output": raw_text})
        detections = self._parse_detections(raw_text)

        if not detections:
            strict_prompt = (
                prompt
                + " Your entire reply must be a valid JSON array that starts with '[' and ends with ']'. "
                "If no objects are detected, respond with []. Do not include any words outside the JSON."
            )
            strict_text = self._generate(strict_prompt, image, tokens=self.max_new_tokens)
            attempts.append({"prompt": strict_prompt, "output": strict_text})
            detections = self._parse_detections(strict_text)
            raw_payload: Any = {
                "attempts": attempts,
                "final_output": strict_text,
            }
        else:
            raw_payload = raw_text

        return {
            "detections": detections,
            "raw_response": raw_payload,
        }

    def _generate(self, prompt: str, image: Image.Image, tokens: int) -> str:
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an object detection assistant. "
                            "Return JSON only in the format "
                            '[{"box_2d":[x1,y1,x2,y2],"label":"class_name"}] using integer pixel coordinates.'
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        inputs = None
        if hasattr(self.processor, "apply_chat_template"):
            try:
                chat_text = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                inputs = self.processor(
                    text=chat_text,
                    images=image,
                    return_tensors="pt",
                )
            except Exception as exc:
                print(f"⚠️ PaLI-Gemma chat template fallback triggered: {exc}")
                inputs = None

        if inputs is None:
            fallback_prompt = (
                "<start_of_turn>system\n"
                "You are an object detection assistant. Return JSON only in the format "
                '[{\"box_2d\":[x1,y1,x2,y2],\"label\":\"class_name\"}] using integer pixel coordinates.'
                "\n<end_of_turn>\n"
                "<start_of_turn>user\n"
                "<image>\n"
                f"{prompt}\n"
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
            )
            try:
                inputs = self.processor(
                    text=[fallback_prompt],
                    images=[image],
                    return_tensors="pt",
                )
            except TypeError:
                inputs = self.processor(
                    text=fallback_prompt,
                    images=image,
                    return_tensors="pt",
                )

        inputs = _move_to_device(inputs, self.device, self.dtype)

        generation_kwargs: Dict[str, Any] = dict(inputs)
        generation_kwargs["max_new_tokens"] = tokens
        generation_kwargs["do_sample"] = False
        if self.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = self.pad_token_id
        if self.eos_token_ids:
            generation_kwargs["eos_token_id"] = (
                self.eos_token_ids if len(self.eos_token_ids) > 1 else self.eos_token_ids[0]
            )

        outputs = self.model.generate(**generation_kwargs)
        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return decoded.strip()

    @staticmethod
    def _parse_detections(raw_text: str) -> List[Dict[str, Any]]:
        if not raw_text:
            return []
        cleaned = PaliGemmaLocalDetector._strip_markup(raw_text)
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if not match:
            return []
        block = match.group(0)
        try:
            data = json.loads(block)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            try:
                data = eval(block, {"__builtins__": None}, {})  # noqa: PGH001,S102
                if isinstance(data, list):
                    return data
            except Exception:
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
    def _strip_markup(text: str) -> str:
        stripped = text.strip()
        if "```" in stripped:
            lines = [ln for ln in stripped.splitlines() if not ln.strip().startswith("```")]
            stripped = "\n".join(lines).strip()
        if "assistant" in stripped.lower():
            parts = stripped.split("assistant", 1)
            stripped = parts[-1].strip()
        if "[" in stripped:
            stripped = stripped[stripped.index("[") :]
        return stripped
