"""YOLO-based vehicle detection and tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import PROJECT_ROOT, Config
from ..io_utils import resolve_path
from ..types import Detection

_CANONICAL_CLASSES = {
    "car": "car",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "van": "van",
    "pickup truck": "pickup truck",
    "three-wheeler": "three-wheeler",
    "three wheeler": "three-wheeler",
    "auto rickshaw": "three-wheeler",
    "bajaj": "three-wheeler",
    "vehicle": "vehicle",
}


class VehicleDetector:
    """Wraps an Ultralytics YOLO detector with project class normalization."""

    def __init__(
        self,
        config: Config,
        weights: str | Path | None = None,
        device: str | None = None,
        lazy: bool = True,
    ):
        self.config = config
        self.weights = self._resolve_weights(weights)
        self.device = device or str(config.get("model.device", config.get("processing.device", "cpu")))
        self.imgsz = int(config.get("model.imgsz", 640))
        self.confidence = float(config.get("model.confidence", 0.30))
        self.iou = float(config.get("model.iou", 0.50))
        self.max_detections = int(config.get("model.max_detections", 100))
        self.agnostic_nms = bool(config.get("model.class_agnostic_nms", True))
        self.class_mapping = {
            str(key).lower(): str(value) for key, value in (config.get("classes", {}) or {}).items()
        }
        self.supported = {
            str(value) for value in (config.get("supported_output_classes", []) or [])
        }
        self._model: Any | None = None
        if not lazy:
            self._load()

    def _resolve_weights(self, weights: str | Path | None) -> str:
        if weights is not None:
            return str(resolve_path(weights))
        configured = self.config.get("model.path")
        if configured:
            candidate = resolve_path(configured)
            if candidate.exists():
                return str(candidate)
        for fallback in ("models/checkpoints/yolo11n.pt", "yolo11n.pt"):
            candidate = PROJECT_ROOT / fallback
            if candidate.exists():
                return str(candidate)
        return str(self.config.get("model.architecture", "yolo11n.pt"))

    @property
    def available(self) -> bool:
        return True

    def _load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        self._model = YOLO(self.weights)

    def _canonical_name(self, raw_name: str, names: dict[int, str] | None, class_id: int) -> str | None:
        lowered = raw_name.strip().lower()
        canonical = _CANONICAL_CLASSES.get(lowered) or self.class_mapping.get(lowered)
        if canonical is None and names and class_id in names:
            fallback = str(names[class_id]).strip().lower()
            canonical = _CANONICAL_CLASSES.get(fallback) or self.class_mapping.get(fallback)
        if canonical is None:
            return None
        if self.supported and canonical not in self.supported and canonical != "vehicle":
            return None
        return canonical

    @staticmethod
    def _boxes_from_result(result: Any) -> Any:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None
        return boxes

    def detect(self, image: np.ndarray, confidence: float | None = None) -> list[Detection]:
        self._load()
        results = self._model.predict(
            source=image,
            conf=float(confidence if confidence is not None else self.confidence),
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            max_det=self.max_detections,
            agnostic_nms=self.agnostic_nms,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = self._boxes_from_result(result)
            if boxes is None:
                continue
            names = getattr(result, "names", None) or getattr(self._model, "names", {})
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for box, score, class_id in zip(xyxy, confs, classes):
                raw_name = str(names.get(int(class_id), class_id))
                canonical = self._canonical_name(raw_name, names, int(class_id))
                if canonical is None:
                    continue
                detections.append(
                    Detection(
                        class_id=int(class_id),
                        class_name=canonical,
                        confidence=float(score),
                        xyxy=tuple(float(v) for v in box[:4]),
                    )
                )
        return detections

    def track(self, image: np.ndarray, persist: bool = True) -> list[Detection]:
        self._load()
        results = self._model.track(
            source=image,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            max_det=self.max_detections,
            agnostic_nms=self.agnostic_nms,
            persist=persist,
            tracker=str(self.config.get("tracking.tracker", "bytetrack.yaml")),
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = self._boxes_from_result(result)
            if boxes is None:
                continue
            names = getattr(result, "names", None) or getattr(self._model, "names", {})
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            track_ids = (
                boxes.id.cpu().numpy().astype(int)
                if getattr(boxes, "id", None) is not None
                else np.full(len(classes), -1, dtype=int)
            )
            for box, score, class_id, track_id in zip(xyxy, confs, classes, track_ids):
                raw_name = str(names.get(int(class_id), class_id))
                canonical = self._canonical_name(raw_name, names, int(class_id))
                if canonical is None:
                    continue
                detections.append(
                    Detection(
                        class_id=int(class_id),
                        class_name=canonical,
                        confidence=float(score),
                        xyxy=tuple(float(v) for v in box[:4]),
                        track_id=int(track_id) if int(track_id) >= 0 else None,
                    )
                )
        return detections

    def reset_tracking(self) -> None:
        if self._model is not None and getattr(self._model, "predictor", None) is not None:
            self._model.predictor = None
