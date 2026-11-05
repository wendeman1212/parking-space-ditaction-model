"""Training entry points for YOLO detectors and CNN classifiers."""

from .classifiers import train_occupancy_classifier, train_scene_classifier
from .yolo import train_parking_space_detector, train_vehicle_detector

__all__ = [
    "train_occupancy_classifier",
    "train_parking_space_detector",
    "train_scene_classifier",
    "train_vehicle_detector",
]
