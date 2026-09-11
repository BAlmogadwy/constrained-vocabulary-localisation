import json
import os
import re
from typing import Any, Dict, List

import PIL.Image

try:
    import google.generativeai as genai  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    genai = None  # type: ignore
    _GENAI_IMPORT_ERROR = exc
else:
    _GENAI_IMPORT_ERROR = None


class GemmaDetector:
    """Gemma 3 multimodal detector using the Google Generative AI API."""

    def __init__(self) -> None:
        if genai is None:  # pragma: no cover - optional dependency guard
            raise ModuleNotFoundError(
                "The 'google-generativeai' package is required for Gemma detectors. "
                "Install it via 'pip install google-generativeai'."
            ) from _GENAI_IMPORT_ERROR
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        genai.configure(api_key=self.api_key)
        model_name = os.getenv("GEMMA_MODEL", "gemma-3-27b-it")
        self.model = genai.GenerativeModel(model_name)

    def predict(
        self,
        image_path: str,
        classes: List[str],
        strategy: str = "single_query",
    ) -> List[Dict[str, Any]]:
        if strategy == "single_query":
            return self._predict_single_query(image_path, classes)
        if strategy == "iterative":
            return self._predict_iterative(image_path, classes)
        raise ValueError(f"Unknown strategy: {strategy}")

    def _predict_single_query(self, image_path: str, classes: List[str]) -> Dict[str, Any]:
        print(f"Predicting objects in {image_path} with classes: {classes} using Gemma 3 (single query).")
        image = PIL.Image.open(image_path)
        prompt = (
            "Detect the following objects in the image: "
            f"{', '.join(classes)}. For each detected object, provide JSON entries "
            '[{"box_2d": [x1, y1, x2, y2], "label": "class_name"}].'
        )
        response = self.model.generate_content([prompt, image], stream=True)
        response.resolve()
        text = getattr(response, "text", "") or ""
        detections = self._parse_detections(text)
        raw_payload = text or self._serialize_response(response)
        return {
            "detections": detections,
            "raw_response": raw_payload,
        }

    def _predict_iterative(self, image_path: str, classes: List[str]) -> Dict[str, Any]:
        print(f"Predicting objects in {image_path} with classes: {classes} using Gemma 3 (iterative).")
        image = PIL.Image.open(image_path)

        id_prompt = "List all the objects you can identify in this image. Respond with comma-separated names."
        id_response = self.model.generate_content([id_prompt, image], stream=True)
        id_response.resolve()
        id_text = getattr(id_response, "text", "") or ""
        object_names = [name.strip() for name in id_text.split(",") if name.strip()]

        detections: List[Dict[str, Any]] = []
        raw_localization: Dict[str, Any] = {}
        for name in object_names:
            if name not in classes:
                continue
            loc_prompt = (
                f"Where is the '{name}' located? Provide the bounding box in [x1, y1, x2, y2] format "
                "with pixel coordinates."
            )
            loc_response = self.model.generate_content([loc_prompt, image], stream=True)
            loc_response.resolve()
            loc_text = getattr(loc_response, "text", "") or ""
            raw_localization[name] = loc_text or self._serialize_response(loc_response)
            detections.extend(det for det in self._parse_detections(loc_text) if det.get("label") == name)
        return {
            "detections": detections,
            "raw_response": {
                "identification": id_text or self._serialize_response(id_response),
                "localization": raw_localization,
            },
        }

    @staticmethod
    def _parse_detections(response_text: str) -> List[Dict[str, Any]]:
        print(response_text)
        try:
            json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if not json_match:
                return []
            detections = json.loads(json_match.group(0))
            if isinstance(detections, list):
                return detections
        except (json.JSONDecodeError, re.error) as exc:
            print(f"Error parsing response: {exc}")
        return []

    @staticmethod
    def _serialize_response(response: Any) -> Any:
        for attr in ("to_dict", "model_dump"):
            if hasattr(response, attr):
                try:
                    return getattr(response, attr)()
                except Exception:  # pragma: no cover
                    continue
        return str(response)
