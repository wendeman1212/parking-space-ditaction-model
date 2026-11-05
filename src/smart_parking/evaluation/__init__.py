"""Evaluation metrics and reports for detection and occupancy."""

from .evaluate import evaluate_occupancy, evaluate_space_detection, evaluate_vehicle_detection
from .metrics import binary_metrics, match_boxes, match_polygons

__all__ = [
    "binary_metrics",
    "evaluate_occupancy",
    "evaluate_space_detection",
    "evaluate_vehicle_detection",
    "match_boxes",
    "match_polygons",
]
