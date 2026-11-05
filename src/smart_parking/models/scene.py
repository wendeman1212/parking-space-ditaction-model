"""Parking-scene validation so buildings, plants and roads are not treated as parking lots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ..config import Config
from ..io_utils import resolve_path
from ..types import ParkingSpace, SceneAssessment, SceneStatus

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SceneClassifier:
    """Binary NON_PARKING/PARKING image classifier."""

    def __init__(self, config: Config, weights: str | Path | None = None, lazy: bool = True):
        self.config = config
        self.weights = str(resolve_path(weights)) if weights else str(resolve_path(config.get("model.path", "")))
        self.input_size = int(config.get("model.input_size", 224))
        self.device_name = str(config.get("model.device", "cpu"))
        self.classes = [str(value).upper() for value in config.get("model.classes", ["NON_PARKING", "PARKING"])]
        self._model: Any | None = None
        self._device: Any | None = None
        self._available = Path(self.weights).exists()
        if not lazy and self._available:
            self._load()

    @property
    def available(self) -> bool:
        return self._available

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from torchvision.models import mobilenet_v3_small

        checkpoint = torch.load(self.weights, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if isinstance(checkpoint, dict) and checkpoint.get("classes"):
            self.classes = [str(value).upper() for value in checkpoint["classes"]]
        if isinstance(checkpoint, dict) and checkpoint.get("input_size"):
            self.input_size = int(checkpoint["input_size"])
        model = mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = torch.nn.Linear(in_features, len(self.classes))
        model.load_state_dict(state_dict)
        self._device = torch.device(self.device_name if self.device_name != "cuda" or torch.cuda.is_available() else "cpu")
        model.to(self._device)
        model.eval()
        self._model = model

    def predict(self, image: np.ndarray) -> np.ndarray | None:
        if not self.available or image is None or image.size == 0:
            return None
        import torch

        self._load()
        resized = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(tensor.to(self._device))
            return torch.softmax(logits, dim=1).cpu().numpy()[0]


class SceneAssessor:
    """Combines layout evidence and optional scene classifier evidence."""

    def __init__(self, config: Config, classifier: SceneClassifier | None = None):
        self.config = config
        self.classifier = classifier or SceneClassifier(config)
        self.confirmed_probability = float(config.get("thresholds.confirmed_probability", 0.75))
        self.possible_probability = float(config.get("thresholds.possible_probability", 0.50))
        self.layout_confirm_spaces = int(config.get("thresholds.layout_confirm_spaces", 3))
        self.layout_possible_spaces = int(config.get("thresholds.layout_possible_spaces", 1))
        self.layout_min_confidence = float(config.get("thresholds.layout_min_confidence", 0.25))

    def assess(self, image: np.ndarray, spaces: Sequence[ParkingSpace]) -> SceneAssessment:
        layout_spaces = len(spaces)
        layout_confidence = (
            float(np.mean([space.confidence for space in spaces])) if spaces else 0.0
        )
        parking_probability = None
        evidence: dict[str, Any] = {
            "layout_spaces": layout_spaces,
            "layout_confidence": layout_confidence,
        }
        if self.classifier.available:
            probabilities = self.classifier.predict(image)
            if probabilities is not None:
                class_lookup = {name: idx for idx, name in enumerate(self.classifier.classes)}
                parking_index = class_lookup.get("PARKING", len(self.classifier.classes) - 1)
                parking_probability = float(probabilities[parking_index])
                evidence["parking_probability"] = parking_probability
        layout_confirmed = layout_spaces >= self.layout_confirm_spaces and layout_confidence >= self.layout_min_confidence
        layout_possible = layout_spaces >= self.layout_possible_spaces and layout_confidence >= self.layout_min_confidence
        if layout_confirmed:
            return SceneAssessment(
                SceneStatus.CONFIRMED,
                confidence=layout_confidence,
                reason=f"layout_with_{layout_spaces}_spaces",
                parking_probability=parking_probability,
                layout_spaces=layout_spaces,
                layout_confidence=layout_confidence,
                evidence=evidence,
            )
        if parking_probability is not None:
            if parking_probability >= self.confirmed_probability:
                return SceneAssessment(
                    SceneStatus.CONFIRMED,
                    confidence=parking_probability,
                    reason="scene_classifier_confirmed",
                    parking_probability=parking_probability,
                    layout_spaces=layout_spaces,
                    layout_confidence=layout_confidence,
                    evidence=evidence,
                )
            if parking_probability >= self.possible_probability or layout_possible:
                return SceneAssessment(
                    SceneStatus.POSSIBLE,
                    confidence=max(parking_probability, layout_confidence),
                    reason="scene_classifier_possible" if parking_probability >= self.possible_probability else "layout_possible",
                    parking_probability=parking_probability,
                    layout_spaces=layout_spaces,
                    layout_confidence=layout_confidence,
                    evidence=evidence,
                )
            return SceneAssessment(
                SceneStatus.NON_PARKING,
                confidence=1.0 - parking_probability,
                reason="scene_classifier_rejected",
                parking_probability=parking_probability,
                layout_spaces=layout_spaces,
                layout_confidence=layout_confidence,
                evidence=evidence,
            )
        if layout_possible:
            return SceneAssessment(
                SceneStatus.POSSIBLE,
                confidence=layout_confidence,
                reason=f"layout_with_{layout_spaces}_spaces",
                layout_spaces=layout_spaces,
                layout_confidence=layout_confidence,
                evidence=evidence,
            )
        return SceneAssessment(
            SceneStatus.INSUFFICIENT,
            confidence=0.0,
            reason="no_parking_layout_identified",
            layout_spaces=layout_spaces,
            layout_confidence=layout_confidence,
            evidence=evidence,
        )
