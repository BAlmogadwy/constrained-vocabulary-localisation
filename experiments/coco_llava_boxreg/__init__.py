"""COCO LLaVA bounding-box regression experiment package."""

from .model import CocoLLaVABBoxRegressor  # noqa: F401
from .dataset import CocoMainObjectDataset  # noqa: F401

__all__ = [
    "CocoLLaVABBoxRegressor",
    "CocoMainObjectDataset",
]
