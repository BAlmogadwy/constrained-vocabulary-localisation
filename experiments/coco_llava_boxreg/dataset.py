from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from PIL import Image
from pycocotools.coco import COCO  # type: ignore
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CocoCategory:
    """Lightweight representation of a COCO category."""

    id: int
    name: str


def _default_area(ann: Dict[str, Any]) -> float:
    bbox = ann.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    if len(bbox) != 4:
        return 0.0
    width = max(float(bbox[2]), 0.0)
    height = max(float(bbox[3]), 0.0)
    if width <= 0.0 or height <= 0.0:
        return 0.0
    return ann.get("area") or (width * height)


def _normalize_xywh(
    bbox_xywh: Sequence[float],
    width: int,
    height: int,
) -> List[float]:
    """Convert COCO xywh box to normalized xyxy format in [0, 1]."""
    x, y, w, h = bbox_xywh
    x_min = max(min(x, width), 0.0)
    y_min = max(min(y, height), 0.0)
    x_max = max(min(x + w, width), 0.0)
    y_max = max(min(y + h, height), 0.0)
    if width > 0:
        x_min /= width
        x_max /= width
    if height > 0:
        y_min /= height
        y_max /= height
    coords = [x_min, y_min, x_max, y_max]
    return [float(max(0.0, min(1.0, coord))) for coord in coords]


class CocoMainObjectDataset(Dataset):
    """
    Dataset wrapper that selects a single "main" object (largest area) per image.

    Each item returns:
        - pixel_values: tensor (C, H, W) normalized according to the provided processor
        - target_box: normalized xyxy bounding box tensor in [0, 1]
        - target_label: contiguous class index (0-79 for COCO)
        - metadata for logging/inference
    """

    def __init__(
        self,
        images_dir: Path,
        annotation_file: Path,
        image_processor: Any,
        split: str = "train",
        max_samples: Optional[int] = None,
        filter_ids: Optional[Iterable[int]] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        self.annotation_file = Path(annotation_file)
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotation_file}")

        self.split = split
        self.coco = COCO(str(self.annotation_file))
        cat_ids = sorted(self.coco.getCatIds())
        cat_infos = self.coco.loadCats(cat_ids)
        ordered = sorted(cat_infos, key=lambda cat: cat["id"])
        self.categories: List[CocoCategory] = [
            CocoCategory(id=cat["id"], name=cat.get("name", f"cat_{cat['id']}")) for cat in ordered
        ]
        self.cat_id_to_index: Dict[int, int] = {cat.id: idx for idx, cat in enumerate(self.categories)}
        self.index_to_cat_id: Dict[int, int] = {idx: cat.id for idx, cat in enumerate(self.categories)}
        self.category_names: List[str] = [cat.name for cat in self.categories]

        self.image_processor = image_processor
        image_ids = sorted(self.coco.getImgIds())
        if filter_ids is not None:
            allowed = set(filter_ids)
            image_ids = [img_id for img_id in image_ids if img_id in allowed]

        self.samples: List[Dict[str, Any]] = []
        for img_id in image_ids:
            info = self.coco.loadImgs(img_id)[0]
            img_path = self.images_dir / info["file_name"]
            if not img_path.exists():
                continue
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id], iscrowd=None))
            valid = [ann for ann in anns if ann.get("bbox") and ann.get("iscrowd", 0) == 0]
            if not valid:
                continue
            best_ann = max(valid, key=_default_area)
            bbox = best_ann.get("bbox")
            if not bbox or not isinstance(bbox, (list, tuple)):
                continue
            cat_id = int(best_ann["category_id"])
            contiguous_id = self.cat_id_to_index.get(cat_id)
            if contiguous_id is None:
                continue
            sample = {
                "image_id": img_id,
                "file_name": info["file_name"],
                "image_path": img_path,
                "width": int(info["width"]),
                "height": int(info["height"]),
                "bbox": bbox,
                "category_id": cat_id,
                "label_index": contiguous_id,
            }
            self.samples.append(sample)
            if max_samples is not None and len(self.samples) >= max_samples:
                break

        if not self.samples:
            raise RuntimeError(
                f"No usable samples found for split={split} in {self.images_dir} with {self.annotation_file}"
            )

        print(
            f"[dataset] COCO split='{split}' :: {len(self.samples)} samples "
            f"(images_dir={self.images_dir}, annotations={self.annotation_file})."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        if hasattr(self.image_processor, "image_processor"):
            processor = self.image_processor.image_processor
        elif hasattr(self.image_processor, "feature_extractor"):
            processor = self.image_processor.feature_extractor
        else:
            processor = self.image_processor
        processed = processor(images=image, return_tensors="pt")
        if "pixel_values" not in processed:
            processed = self.image_processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"][0]
        norm_box = _normalize_xywh(sample["bbox"], sample["width"], sample["height"])
        return {
            "pixel_values": pixel_values,
            "target_box": torch.tensor(norm_box, dtype=torch.float32),
            "target_label": torch.tensor(sample["label_index"], dtype=torch.long),
            "image_id": sample["image_id"],
            "image_path": str(sample["image_path"]),
            "image_size": torch.tensor(
                [sample["width"], sample["height"]],
                dtype=torch.float32,
            ),
            "category_id": sample["category_id"],
        }


def coco_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
    target_boxes = torch.stack([item["target_box"] for item in batch], dim=0)
    target_labels = torch.stack([item["target_label"] for item in batch], dim=0)
    image_sizes = torch.stack([item["image_size"] for item in batch], dim=0)
    image_ids = [item["image_id"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    category_ids = [item["category_id"] for item in batch]
    return {
        "pixel_values": pixel_values,
        "target_boxes": target_boxes,
        "target_labels": target_labels,
        "image_sizes": image_sizes,
        "image_ids": image_ids,
        "image_paths": image_paths,
        "category_ids": category_ids,
    }
