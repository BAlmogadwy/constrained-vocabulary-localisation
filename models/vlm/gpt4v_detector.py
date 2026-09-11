import os
import base64
import requests
import json
import re
from typing import List, Dict, Any

class GPT4VDetector:
    """GPT-4V detector for zero-shot object detection."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def predict(self, image_path: str, classes: List[str], strategy: str = 'single_query') -> List[Dict[str, Any]]:
        """Predicts objects in an image using GPT-4V with a given strategy."""
        if strategy == 'single_query':
            return self._predict_single_query(image_path, classes)
        elif strategy == 'iterative':
            return self._predict_iterative(image_path, classes)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _predict_single_query(self, image_path: str, classes: List[str]) -> List[Dict[str, Any]]:
        print(f"Predicting objects in {image_path} with classes: {classes} using GPT-4V (single query).")
        base64_image = self._encode_image(image_path)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        prompt = (
            f"Detect the following objects in the image: {', '.join(classes)}. "
            "For each detected object, provide its name and bounding box in a JSON format like this: "
            "[{\"box_2d\": [x1, y1, x2, y2], \"label\": \"class_name\"}, ...]."
        )
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 500
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_json = response.json()
        print(response_json)
        try:
            content = response_json['choices'][0]['message']['content']
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                detections_str = json_match.group(0)
                detections = json.loads(detections_str)
                return detections
        except (KeyError, IndexError, json.JSONDecodeError, re.error) as e:
            print(f"Error parsing response: {e}")
        return []

    def _predict_iterative(self, image_path: str, classes: List[str]) -> List[Dict[str, Any]]:
        print(f"Predicting objects in {image_path} with classes: {classes} using GPT-4V (iterative).")
        base64_image = self._encode_image(image_path)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        # First query: Identify objects
        id_prompt = f"List all the objects you can detect in this image. Only list the object names, separated by commas."
        id_payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": id_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 300
        }
        id_response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=id_payload)
        id_response_json = id_response.json()
        print("Identification response:", id_response_json)

        try:
            content = id_response_json['choices'][0]['message']['content']
            object_names = [name.strip() for name in content.split(',')]
        except (KeyError, IndexError) as e:
            print(f"Error parsing identification response: {e}")
            return []

        # Second query: Get bounding box for each object
        detections = []
        for name in object_names:
            if name not in classes:
                continue # Skip objects we are not interested in

            loc_prompt = f"Where is the '{name}' in the image? Provide the bounding box coordinates in the format [x1, y1, x2, y2]."
            loc_payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": loc_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                "max_tokens": 100
            }
            loc_response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=loc_payload)
            loc_response_json = loc_response.json()
            print(f"Localization response for '{name}':", loc_response_json)

            try:
                content = loc_response_json['choices'][0]['message']['content']
                json_match = re.search(r'\[.*?\]', content, re.DOTALL)
                if json_match:
                    box_str = json_match.group(0)
                    box = json.loads(box_str)
                    detections.append({"box_2d": box, "label": name})
            except (KeyError, IndexError, json.JSONDecodeError, re.error) as e:
                print(f"Error parsing localization response for '{name}': {e}")

        return detections
