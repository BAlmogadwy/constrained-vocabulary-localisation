"""Generic detector for OpenAI-compatible multimodal chat endpoints."""

from __future__ import annotations

import base64
import os
import time
from typing import Any, Dict, List

import requests
from PIL import Image

from .parsing import (
    extract_openai_compatible_text,
    normalize_detections_to_image,
    parse_detections_from_text,
)


class OpenAICompatibleVLMDetector:
    provider_name = "openai-compatible"
    api_key_env = "OPENAI_COMPATIBLE_API_KEY"
    api_base_env = "OPENAI_COMPATIBLE_API_BASE"
    model_env = "OPENAI_COMPATIBLE_MODEL"
    timeout_env = "OPENAI_COMPATIBLE_TIMEOUT"
    max_tokens_env = "OPENAI_COMPATIBLE_MAX_TOKENS"
    temperature_env = "OPENAI_COMPATIBLE_TEMPERATURE"
    max_retries_env = "OPENAI_COMPATIBLE_MAX_RETRIES"
    retry_delay_env = "OPENAI_COMPATIBLE_RETRY_DELAY"
    default_api_base = "https://api.openai.com/v1"
    default_model = ""
    default_timeout = 120
    default_max_tokens = 800
    default_temperature = 0.2
    default_max_retries = 3
    default_retry_delay = 60.0
    coordinate_mode = "pixel"
    system_prompt = ""
    response_format: Dict[str, Any] | None = None

    def __init__(self) -> None:
        self.api_key = os.getenv(self.api_key_env)
        if not self.api_key:
            raise ValueError(f"{self.api_key_env} environment variable not set.")
        self.api_base = os.getenv(self.api_base_env, self.default_api_base).rstrip("/")
        self.model = os.getenv(self.model_env, self.default_model)
        if not self.model:
            raise ValueError(f"{self.model_env} environment variable not set and no default model configured.")
        self.timeout = int(os.getenv(self.timeout_env, str(self.default_timeout)))
        self.max_tokens = int(os.getenv(self.max_tokens_env, str(self.default_max_tokens)))
        self.temperature = float(os.getenv(self.temperature_env, str(self.default_temperature)))
        self.max_retries = int(os.getenv(self.max_retries_env, str(self.default_max_retries)))
        self.retry_delay = float(os.getenv(self.retry_delay_env, str(self.default_retry_delay)))

    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def _prompt(classes: List[str]) -> str:
        return (
            "Detect the following objects in the image: "
            f"{', '.join(classes)}. For each detected object, return JSON in the form "
            '[{"box_2d": [x1, y1, x2, y2], "label": "class_name", "confidence": score}, ...]. '
            "Use integer pixel coordinates relative to the input image. Return only JSON."
        )

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(self.max_retries + 1):
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code == 429 and attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after is not None else self.retry_delay
                except ValueError:
                    delay = self.retry_delay
                time.sleep(max(delay, 0.0))
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError("unreachable")

    def _single_query_payload(self, image_path: str, classes: List[str]) -> Dict[str, Any]:
        base64_image = self._encode_image(image_path)
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._prompt(classes)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        )
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.response_format:
            payload["response_format"] = self.response_format
        return payload

    def predict(
        self,
        image_path: str,
        classes: List[str],
        strategy: str = "single_query",
    ) -> Dict[str, Any]:
        if strategy != "single_query":
            raise ValueError(f"Strategy '{strategy}' is not supported by {self.provider_name}.")
        payload = self._single_query_payload(image_path, classes)
        response_json = self._request(payload)
        with Image.open(image_path) as image:
            image_size = image.size
        raw_text = extract_openai_compatible_text(response_json)
        parse_image_size = None if self.coordinate_mode.startswith("normalized_1000") else image_size
        detections = parse_detections_from_text(raw_text, image_size=parse_image_size)
        detections = normalize_detections_to_image(
            detections,
            image_size=image_size,
            coordinate_mode=self.coordinate_mode,
        )
        allowed_labels = set(classes)
        detections = [
            detection
            for detection in detections
            if detection.get("label") in allowed_labels
        ]
        return {
            "detections": detections,
            "raw_response": response_json,
        }


class OpenRouterQwen25VL72BDetector(OpenAICompatibleVLMDetector):
    provider_name = "openrouter"
    api_key_env = "OPENROUTER_API_KEY"
    api_base_env = "OPENROUTER_API_BASE"
    model_env = "OPENROUTER_QWEN25_VL_72B_MODEL"
    default_api_base = "https://openrouter.ai/api/v1"
    default_model = "qwen/qwen2.5-vl-72b-instruct"


class OpenRouterPixtralLargeDetector(OpenAICompatibleVLMDetector):
    provider_name = "openrouter"
    api_key_env = "OPENROUTER_API_KEY"
    api_base_env = "OPENROUTER_API_BASE"
    model_env = "OPENROUTER_PIXTRAL_LARGE_MODEL"
    default_api_base = "https://openrouter.ai/api/v1"
    default_model = "mistralai/pixtral-large-2411"


class OpenRouterMistralLargeDetector(OpenAICompatibleVLMDetector):
    provider_name = "openrouter"
    api_key_env = "OPENROUTER_API_KEY"
    api_base_env = "OPENROUTER_API_BASE"
    model_env = "OPENROUTER_MISTRAL_LARGE_MODEL"
    default_api_base = "https://openrouter.ai/api/v1"
    default_model = "mistralai/mistral-large-2512"
    coordinate_mode = "normalized_1000_xyxy"


class OpenRouterQwen3VL235BDetector(OpenAICompatibleVLMDetector):
    provider_name = "openrouter"
    api_key_env = "OPENROUTER_API_KEY"
    api_base_env = "OPENROUTER_API_BASE"
    model_env = "OPENROUTER_QWEN3_VL_235B_MODEL"
    default_api_base = "https://openrouter.ai/api/v1"
    default_model = "qwen/qwen3-vl-235b-a22b-instruct"
    coordinate_mode = "normalized_1000_xyxy"


class OpenRouterLlama4MaverickDetector(OpenAICompatibleVLMDetector):
    provider_name = "openrouter"
    api_key_env = "OPENROUTER_API_KEY"
    api_base_env = "OPENROUTER_API_BASE"
    model_env = "OPENROUTER_LLAMA4_MAVERICK_MODEL"
    default_api_base = "https://openrouter.ai/api/v1"
    default_model = "meta-llama/llama-4-maverick"


class OpenRouterLlama32Vision11BDetector(OpenAICompatibleVLMDetector):
    provider_name = "openrouter"
    api_key_env = "OPENROUTER_API_KEY"
    api_base_env = "OPENROUTER_API_BASE"
    model_env = "OPENROUTER_LLAMA32_VISION_11B_MODEL"
    default_api_base = "https://openrouter.ai/api/v1"
    default_model = "meta-llama/llama-3.2-11b-vision-instruct"
    default_temperature = 0.0
    response_format = {"type": "json_object"}
    system_prompt = (
        "You are an object detection engine. Return only valid JSON. "
        "Do not describe the image. Do not include markdown. "
        "Do not invent objects or zero-area placeholder boxes. "
        'If no listed object is visible, return {"detections": []}.'
    )

    @staticmethod
    def _prompt(classes: List[str]) -> str:
        return (
            "Detect only visible instances of these target classes: "
            f"{', '.join(classes)}.\n"
            "Return exactly one JSON object with this schema:\n"
            '{"detections": [{"box_2d": [x1, y1, x2, y2], '
            '"label": "one of the target class names", "confidence": 0.0}]}\n'
            "Rules: use integer pixel coordinates relative to the input image; "
            "x2 must be greater than x1 and y2 must be greater than y1; "
            "labels must exactly match one of the target classes; "
            'if none of the target classes are visible, return {"detections": []}; '
            "return JSON only."
        )


class OpenRouterGemma327BDetector(OpenAICompatibleVLMDetector):
    provider_name = "openrouter"
    api_key_env = "OPENROUTER_API_KEY"
    api_base_env = "OPENROUTER_API_BASE"
    model_env = "OPENROUTER_GEMMA3_27B_MODEL"
    default_api_base = "https://openrouter.ai/api/v1"
    default_model = "google/gemma-3-27b-it"
    coordinate_mode = "normalized_1000_xyxy"


class DashScopeQwenVLMaxDetector(OpenAICompatibleVLMDetector):
    provider_name = "dashscope"
    api_key_env = "QWEN_API_KEY"
    api_base_env = "QWEN_API_BASE"
    model_env = "QWEN_VL_MAX_MODEL"
    default_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    default_model = "qwen-vl-max"


class MistralPixtralLargeDetector(OpenAICompatibleVLMDetector):
    provider_name = "mistral"
    api_key_env = "MISTRAL_API_KEY"
    api_base_env = "MISTRAL_API_BASE"
    model_env = "MISTRAL_PIXTRAL_LARGE_MODEL"
    default_api_base = "https://api.mistral.ai/v1"
    default_model = "pixtral-large-latest"
