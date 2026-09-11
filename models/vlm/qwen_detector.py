import base64
import json
import os
import re
from typing import Any, Dict, List

import requests

from .parsing import extract_openai_compatible_text, parse_detections_from_text


class QwenDetector:
    """Qwen2.5-VL-72B detector using the DashScope compatible API."""

    api_key_env = "QWEN_API_KEY"
    api_base_env = "QWEN_API_BASE"
    model_env = "QWEN_MODEL"
    default_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    default_model = "qwen2.5-vl-72b-instruct"

    def __init__(self) -> None:
        self.api_key = os.getenv(self.api_key_env) or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(f"{self.api_key_env} (or DASHSCOPE_API_KEY) environment variable not set.")
        self.api_base = os.getenv(
            self.api_base_env,
            self.default_api_base,
        )
        self.model = os.getenv(self.model_env, self.default_model)

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.api_base.rstrip('/')}/chat/completions"
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()

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

    def _predict_single_query(self, image_path: str, classes: List[str]) -> Dict[str, Any]:
        base64_image = self._encode_image(image_path)
        prompt = (
            "Detect the following objects in the image: "
            f"{', '.join(classes)}. For each detected object, return JSON in the form "
            '[{"box_2d": [x1, y1, x2, y2], "label": "class_name", "confidence": score}, ...]. '
            "Use pixel coordinates relative to the input image."
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            "temperature": 0.2,
            "max_tokens": 800,
        }
        response_json = self._request(payload)
        detections = self._parse_detection_response(response_json)
        return {
            "detections": detections,
            "raw_response": response_json,
        }

    def _predict_iterative(self, image_path: str, classes: List[str]) -> Dict[str, Any]:
        base64_image = self._encode_image(image_path)
        id_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "List every object you can recognize in this image. "
                                "Respond with comma-separated object names."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            "temperature": 0.2,
            "max_tokens": 200,
        }
        id_response = self._request(id_payload)
        object_names = self._parse_object_list(id_response)

        detections: List[Dict[str, Any]] = []
        raw_localization: Dict[str, Any] = {}
        for name in object_names:
            if name not in classes:
                continue
            loc_payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"Provide the bounding box for '{name}' in the format "
                                    "[x1, y1, x2, y2] (pixel coordinates)."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 200,
            }
            loc_response = self._request(loc_payload)
            parsed = self._parse_detection_response(loc_response)
            detections.extend(det for det in parsed if det.get("label") == name)
            raw_localization[name] = {
                "request": loc_payload,
                "response": loc_response,
            }
        return {
            "detections": detections,
            "raw_response": {
                "identification": id_response,
                "localization": raw_localization,
            },
        }

    @staticmethod
    def _parse_object_list(response_json: Dict[str, Any]) -> List[str]:
        try:
            content = response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return []
        if not content:
            return []
        return [item.strip() for item in content.split(",") if item.strip()]

    @staticmethod
    def _parse_detection_response(response_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        return parse_detections_from_text(extract_openai_compatible_text(response_json))
