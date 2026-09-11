from typing import List, Dict, Any
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import PIL.Image
import json
import re

class Blip2Detector:
    """BLIP-2 detector for zero-shot object detection."""

    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print("BLIP-2 model loaded.")

    def predict(self, image_path: str, classes: List[str], strategy: str = 'single_query') -> List[Dict[str, Any]]:
        """Predicts objects in an image using BLIP-2 with a given strategy."""
        if strategy == 'single_query':
            return self._predict_single_query(image_path, classes)
        elif strategy == 'iterative':
            return self._predict_iterative(image_path, classes)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _predict_single_query(self, image_path: str, classes: List[str]) -> List[Dict[str, Any]]:
        print(f"Predicting objects in {image_path} with classes: {classes} using BLIP-2 (single query).")
        image = PIL.Image.open(image_path)
        prompt = (
            f"Question: Detect the following objects in the image: {', '.join(classes)}. "
            "For each detected object, provide its name and bounding box in a JSON format like this: "
            "[{\"box_2d\": [x1, y1, x2, y2], \"label\": \"class_name\"}, ...]. Answer:"
        )
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, dtype=torch.float16)
        generated_ids = self.model.generate(**inputs, max_length=500)
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(response)
        try:
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                detections_str = json_match.group(0)
                detections = json.loads(detections_str)
                return detections
        except (KeyError, IndexError, json.JSONDecodeError, re.error) as e:
            print(f"Error parsing response: {e}")
        return []

    def _predict_iterative(self, image_path: str, classes: List[str]) -> List[Dict[str, Any]]:
        print(f"Predicting objects in {image_path} with classes: {classes} using BLIP-2 (iterative).")
        image = PIL.Image.open(image_path)

        # First query: Identify objects
        id_prompt = "Question: List all the objects you can detect in this image. Only list the object names, separated by commas. Answer:"
        inputs = self.processor(images=image, text=id_prompt, return_tensors="pt").to(self.device, dtype=torch.float16)
        generated_ids = self.model.generate(**inputs, max_length=300)
        id_response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print("Identification response:", id_response)

        try:
            object_names = [name.strip() for name in id_response.split(',')]
        except (KeyError, IndexError) as e:
            print(f"Error parsing identification response: {e}")
            return []

        # Second query: Get bounding box for each object
        detections = []
        for name in object_names:
            if name not in classes:
                continue

            loc_prompt = f"Question: Where is the '{name}' in the image? Provide the bounding box coordinates in the format [x1, y1, x2, y2]. Answer:"
            inputs = self.processor(images=image, text=loc_prompt, return_tensors="pt").to(self.device, dtype=torch.float16)
            generated_ids = self.model.generate(**inputs, max_length=100)
            loc_response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            print(f"Localization response for '{name}':", loc_response)

            try:
                json_match = re.search(r'\[.*?\]', loc_response, re.DOTALL)
                if json_match:
                    box_str = json_match.group(0)
                    box = json.loads(box_str)
                    detections.append({"box_2d": box, "label": name})
            except (KeyError, IndexError, json.JSONDecodeError, re.error) as e:
                print(f"Error parsing localization response for '{name}': {e}")

        return detections
