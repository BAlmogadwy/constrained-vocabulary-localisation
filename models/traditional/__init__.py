"""Traditional detector wrappers."""

from .grounding_dino import GroundingDINODetector, GroundingDINOConfig
from .owl_vit import OWLVitDetector, OWLVitConfig
from .yolo_world import YOLOWorldDetector, YOLOWorldConfig

__all__ = [
    "GroundingDINODetector",
    "GroundingDINOConfig",
    "OWLVitDetector",
    "OWLVitConfig",
    "YOLOWorldDetector",
    "YOLOWorldConfig",
]
