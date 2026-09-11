import base64
import json
import os
import re
from typing import Any, Dict, List

from .parsing import parse_detections_from_text

try:
    from openai import OpenAI  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore
    _OPENAI_IMPORT_ERROR = exc
else:
    _OPENAI_IMPORT_ERROR = None


class GPT5Detector:
    """GPT-5 vision-capable detector using the OpenAI Responses API."""

    model_env = "OPENAI_VISION_MODEL"
    default_model = "gpt-5"

    def __init__(self) -> None:
        if OpenAI is None:  # pragma: no cover - optional dependency guard
            raise ModuleNotFoundError(
                "The 'openai' package is required to use GPT-5 detectors. "
                "Install it via 'pip install openai'."
            ) from _OPENAI_IMPORT_ERROR
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv(self.model_env, self.default_model)
        self.max_output_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "2048"))

    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def predict(
        self,
        image_path: str,
        classes: List[str],
        strategy: str = "single_query",
    ) -> List[Dict]:
        if strategy == "single_query":
            return self._predict_single_query(image_path, classes)
        if strategy == "iterative":
            return self._predict_iterative(image_path, classes)
        raise ValueError(f"Unknown strategy: {strategy}")

    def _predict_single_query(self, image_path: str, classes: List[str]) -> List[Dict]:
        base64_image = self._encode_image(image_path)
        prompt = (
            "Detect the following objects in the image: "
            f"{', '.join(classes)}. For each detected object, return JSON in the form "
            '[{"box_2d": [x1, y1, x2, y2], "label": "class_name", "confidence": score}, ...]. '
            "Use pixel coordinates relative to the input image."
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            ],
            max_output_tokens=self.max_output_tokens,
        )
        output_text = self._extract_output_text(response)
        detections = self._parse_detection_response(output_text)
        return {
            "detections": detections,
            "raw_response": self._serialize_response(response),
        }

    def _predict_iterative(self, image_path: str, classes: List[str]) -> List[Dict]:
        base64_image = self._encode_image(image_path)

        id_response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "List every object you can recognize in this image. "
                                "Respond with comma-separated names."
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            ],
            max_output_tokens=256,
        )
        object_names = self._parse_object_list(self._extract_output_text(id_response))

        detections: List[Dict] = []
        raw_calls: Dict[str, List[Any]] = {
            "identification": self._serialize_response(id_response),
            "localization": [],
        }
        for name in object_names:
            if name not in classes:
                continue
            loc_response = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    f"Provide the bounding box for '{name}' in the format "
                                    "[x1, y1, x2, y2] with optional confidence score."
                                ),
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        ],
                    }
                ],
                max_output_tokens=256,
            )
            parsed = self._parse_detection_response(self._extract_output_text(loc_response))
            detections.extend(det for det in parsed if det.get("label") == name)
            raw_calls["localization"].append(self._serialize_response(loc_response))
        return {
            "detections": detections,
            "raw_response": raw_calls,
        }

    @staticmethod
    def _extract_output_text(response) -> str | None:
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text
        pieces: List[str] = []
        for item in getattr(response, "output", []) or []:
            if item.get("type") != "message":
                continue
            for content in item.get("content", []) or []:
                text = content.get("text")
                if text:
                    pieces.append(text)
        return "\n".join(pieces) if pieces else None

    @staticmethod
    def _parse_object_list(response_text: str | None) -> List[str]:
        if not response_text:
            return []
        return [item.strip() for item in response_text.split(",") if item.strip()]

    @staticmethod
    def _parse_detection_response(response_text: str | None) -> List[Dict]:
        return parse_detections_from_text(response_text)

    @staticmethod
    def _serialize_response(response):
        if hasattr(response, "model_dump"):
            try:
                return response.model_dump()
            except Exception:  # pragma: no cover - defensive
                pass
        if hasattr(response, "to_dict"):
            try:
                return response.to_dict()
            except Exception:  # pragma: no cover - defensive
                pass
        return response


class GPT55Detector(GPT5Detector):
    """GPT-5.5 detector using the OpenAI Responses API."""

    model_env = "OPENAI_GPT55_MODEL"
    default_model = "gpt-5.5"


class GPT54MiniDetector(GPT5Detector):
    """GPT-5.4-mini detector using the OpenAI Responses API."""

    model_env = "OPENAI_GPT54_MINI_MODEL"
    default_model = "gpt-5.4-mini"
