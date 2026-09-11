import json
import os
import re
from typing import Any, Dict, List

import PIL.Image
from .parsing import parse_detections_from_text

try:
    import google.generativeai as genai  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    genai = None  # type: ignore
    _GENAI_IMPORT_ERROR = exc
else:
    _GENAI_IMPORT_ERROR = None


class GeminiFlashDetector:
    """Gemini 2.5 Flash detector for zero-shot object detection."""

    model_env = "GEMINI_VISION_MODEL"
    default_model = "gemini-2.5-flash"

    def __init__(self) -> None:
        if genai is None:  # pragma: no cover - optional dependency guard
            raise ModuleNotFoundError(
                "The 'google-generativeai' package is required for Gemini detectors. "
                "Install it via 'pip install google-generativeai'."
            ) from _GENAI_IMPORT_ERROR
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=self.api_key)
        model_name = os.getenv(self.model_env, self.default_model)
        self.model = genai.GenerativeModel(model_name)

    def predict(
        self,
        image_path: str,
        classes: List[str],
        strategy: str = "single_query",
    ) -> Dict[str, Any]:
        """Predict objects in an image using Gemini 2.5 Flash with a given strategy."""
        if strategy == "single_query":
            return self._predict_single_query(image_path, classes)
        if strategy == "iterative":
            return self._predict_iterative(image_path, classes)
        raise ValueError(f"Unknown strategy: {strategy}")

    def _predict_single_query(self, image_path: str, classes: List[str]) -> Dict[str, Any]:
        print(f"Predicting objects in {image_path} with classes: {classes} using Gemini 2.5 Flash (single query).")
        img = PIL.Image.open(image_path)
        prompt = (
            "Detect the following objects in the image: "
            f"{', '.join(classes)}. For each detected object, provide its name and bounding box in JSON format "
            '[{"box_2d": [x1, y1, x2, y2], "label": "class_name"}]. '
            "Return only JSON; no additional explanation."
        )
        response = self.model.generate_content([prompt, img], stream=True)
        response.resolve()
        raw_text = getattr(response, "text", "") or ""
        print(raw_text)
        detections = self._parse_detections(raw_text)
        return {
            "detections": detections,
            "raw_response": raw_text or self._serialize_response(response),
        }

    def _predict_iterative(self, image_path: str, classes: List[str]) -> Dict[str, Any]:
        print(f"Predicting objects in {image_path} with classes: {classes} using Gemini 2.5 Flash (iterative).")
        img = PIL.Image.open(image_path)

        id_prompt = (
            "List all the objects you can detect in this image. Respond with comma-separated object names only."
        )
        id_response = self.model.generate_content([id_prompt, img], stream=True)
        id_response.resolve()
        id_text = getattr(id_response, "text", "") or ""
        print("Identification response:", id_text)

        object_names: List[str] = []
        if id_text:
            object_names = [name.strip() for name in id_text.split(",") if name.strip()]

        detections: List[Dict[str, Any]] = []
        raw_calls: Dict[str, Any] = {
            "identification": id_text,
            "localization": {},
        }

        for name in object_names:
            if name not in classes:
                continue

            loc_prompt = (
                f"Where is the '{name}' in the image? Provide the bounding box coordinates in the format "
                "[x1, y1, x2, y2] as JSON only."
            )
            loc_response = self.model.generate_content([loc_prompt, img], stream=True)
            loc_response.resolve()
            loc_text = getattr(loc_response, "text", "") or ""
            print(f"Localization response for '{name}':", loc_text)
            raw_calls["localization"][name] = loc_text or self._serialize_response(loc_response)

            parsed = self._parse_detections(loc_text)
            detections.extend(det for det in parsed if det.get("label") == name or det.get("label") in classes)

        return {
            "detections": detections,
            "raw_response": raw_calls,
        }

    @staticmethod
    def _parse_detections(response_text: str) -> List[Dict[str, Any]]:
        return parse_detections_from_text(response_text)

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = []
            for line in stripped.splitlines():
                if line.strip().startswith("```"):
                    continue
                lines.append(line)
            stripped = "\n".join(lines).strip()
        return stripped

    @staticmethod
    def _serialize_response(response: Any) -> Any:
        for attr in ("to_dict", "model_dump"):
            if hasattr(response, attr):
                try:
                    return getattr(response, attr)()
                except Exception:  # pragma: no cover
                    continue
        return str(response)


class Gemini31ProDetector(GeminiFlashDetector):
    """Gemini 3.1 Pro detector."""

    model_env = "GEMINI_31_PRO_MODEL"
    default_model = "gemini-3.1-pro-preview"


class Gemini35FlashDetector(GeminiFlashDetector):
    """Gemini 3.5 Flash detector."""

    model_env = "GEMINI_35_FLASH_MODEL"
    default_model = "gemini-3.5-flash"
