import os
import base64
import json
import re
from typing import List, Dict, Any

from .parsing import parse_detections_from_text

try:
    import anthropic  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    anthropic = None  # type: ignore
    _ANTHROPIC_IMPORT_ERROR = exc
else:
    _ANTHROPIC_IMPORT_ERROR = None

class ClaudeHaikuDetector:
    """Claude Haiku 4.5 detector for zero-shot object detection."""

    model_env = "ANTHROPIC_VISION_MODEL"
    default_model = "claude-haiku-4-5"

    def __init__(self):
        if anthropic is None:  # pragma: no cover - optional dependency guard
            raise ModuleNotFoundError(
                "The 'anthropic' package is required to use Claude detectors. "
                "Install it via 'pip install anthropic'."
            ) from _ANTHROPIC_IMPORT_ERROR
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = os.getenv(self.model_env, self.default_model)

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def predict(self, image_path: str, classes: List[str], strategy: str = 'single_query') -> Dict[str, Any]:
        """Predicts objects in an image using Claude Haiku 4.5 with a given strategy."""
        if strategy == 'single_query':
            return self._predict_single_query(image_path, classes)
        elif strategy == 'iterative':
            return self._predict_iterative(image_path, classes)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _predict_single_query(self, image_path: str, classes: List[str]) -> Dict[str, Any]:
        print(f"Predicting objects in {image_path} with classes: {classes} using Claude Haiku 4.5 (single query).")
        base64_image = self._encode_image(image_path)
        prompt = (
            f"Detect the following objects in the image: {', '.join(classes)}. "
            "For each detected object, provide its name and bounding box in a JSON format like this: "
            "[{\"box_2d\": [x1, y1, x2, y2], \"label\": \"class_name\"}, ...]."
        )
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        )
        print(message.content)
        detections: List[Dict[str, Any]] = []
        try:
            content = message.content[0].text
            detections = parse_detections_from_text(content)
        except (KeyError, IndexError) as e:
            print(f"Error parsing response: {e}")
        return {
            "detections": detections,
            "raw_response": self._serialize_response(message),
        }

    def _predict_iterative(self, image_path: str, classes: List[str]) -> Dict[str, Any]:
        print(f"Predicting objects in {image_path} with classes: {classes} using Claude Haiku 4.5 (iterative).")
        base64_image = self._encode_image(image_path)

        # First query: Identify objects
        id_prompt = f"List all the objects you can detect in this image. Only list the object names, separated by commas."
        id_message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}},
                        {"type": "text", "text": id_prompt}
                    ]
                }
            ]
        )
        print("Identification response:", id_message.content)

        try:
            content = id_message.content[0].text
            object_names = [name.strip() for name in content.split(',')]
        except (KeyError, IndexError) as e:
            print(f"Error parsing identification response: {e}")
            return {
                "detections": [],
                "raw_response": {
                    "identification": self._serialize_response(id_message),
                    "localization": {},
                },
            }

        # Second query: Get bounding box for each object
        detections = []
        raw_localization: Dict[str, Any] = {}
        for name in object_names:
            if name not in classes:
                continue

            loc_prompt = f"Where is the '{name}' in the image? Provide the bounding box coordinates in the format [x1, y1, x2, y2]."
            loc_message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}},
                            {"type": "text", "text": loc_prompt}
                        ]
                    }
                ]
            )
            print(f"Localization response for '{name}':", loc_message.content)
            raw_localization[name] = self._serialize_response(loc_message)

            try:
                content = loc_message.content[0].text
                json_match = re.search(r'\[.*?\]', content, re.DOTALL)
                if json_match:
                    box_str = json_match.group(0)
                    box = json.loads(box_str)
                    detections.append({"box_2d": box, "label": name})
            except (KeyError, IndexError, json.JSONDecodeError, re.error) as e:
                print(f"Error parsing localization response for '{name}': {e}")

        raw_response = {
            "identification": self._serialize_response(id_message),
            "localization": raw_localization,
        }
        return {
            "detections": detections,
            "raw_response": raw_response,
        }

    @staticmethod
    def _serialize_response(message: Any) -> Any:
        for attr in ("model_dump", "to_dict", "dict"):
            if hasattr(message, attr):
                try:
                    return getattr(message, attr)()
                except Exception:  # pragma: no cover - defensive
                    continue
        return str(message)


class ClaudeOpus48Detector(ClaudeHaikuDetector):
    """Claude Opus 4.8 detector."""

    model_env = "ANTHROPIC_CLAUDE_OPUS_48_MODEL"
    default_model = "claude-opus-4-8"


class ClaudeHaiku45Detector(ClaudeHaikuDetector):
    """Claude Haiku 4.5 detector."""

    model_env = "ANTHROPIC_CLAUDE_HAIKU_45_MODEL"
    default_model = "claude-haiku-4-5"
