"""Occupancy classification with geometry rules, a CNN classifier and temporal smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ..config import Config
from ..data.annotations import crop_quality
from ..geometry import bbox_polygon_overlap, expand_polygon, polygon_area, visible_fraction, warp_polygon_crop
from ..io_utils import resolve_path
from ..types import Detection, OccupancyState, ParkingSpace, SpaceStatus, Statistics

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OccupancyClassifier:
    """MobileNetV3-based EMPTY/OCCUPIED classifier for parking-space crops."""

    def __init__(self, config: Config, weights: str | Path | None = None, lazy: bool = True):
        self.config = config
        self.weights = str(resolve_path(weights)) if weights else str(resolve_path(config.get("model.path", "")))
        self.input_size = int(config.get("model.input_size", 160))
        self.device_name = str(config.get("model.device", "cpu"))
        self.classes = [str(value).upper() for value in config.get("model.classes", ["EMPTY", "OCCUPIED"])]
        self.batch_size = int(config.get("model.batch_size", 128))
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

        if not Path(self.weights).exists():
            raise FileNotFoundError(f"Occupancy classifier checkpoint not found: {self.weights}")
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

    def _preprocess(self, crops: Sequence[np.ndarray]) -> Any:
        import torch

        tensors = []
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            resized = cv2.resize(crop, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
            tensors.append(torch.from_numpy(rgb).permute(2, 0, 1))
        return torch.stack(tensors) if tensors else None

    def predict(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Return an ``(N, 2)`` array of EMPTY/OCCUPIED probabilities."""
        if not crops:
            return np.zeros((0, len(self.classes)), dtype=np.float32)
        if not self.available:
            return np.full((len(crops), len(self.classes)), np.nan, dtype=np.float32)
        import torch

        self._load()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(crops), self.batch_size):
                batch = self._preprocess(crops[start : start + self.batch_size])
                if batch is None:
                    continue
                logits = self._model(batch.to(self._device))
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        if not outputs:
            return np.full((len(crops), len(self.classes)), np.nan, dtype=np.float32)
        return np.concatenate(outputs, axis=0)


@dataclass
class _TemporalState:
    stable: OccupancyState
    candidate: OccupancyState | None = None
    count: int = 0
    candidate_start: float | None = None


class TemporalSmoother:
    """Debounces per-space state changes across video frames."""

    def __init__(self, config: Config):
        self.occupied_frames = max(1, int(config.get("temporal.occupied_confirm_frames", 2)))
        self.empty_frames = max(1, int(config.get("temporal.empty_confirm_frames", 2)))
        self.vacated_seconds = float(config.get("temporal.vacated_confirm_seconds", 0.0))
        self._states: dict[str, _TemporalState] = {}

    def reset(self) -> None:
        self._states.clear()

    def update(self, statuses: list[SpaceStatus], timestamp: float | None = None) -> list[SpaceStatus]:
        smoothed: list[SpaceStatus] = []
        for status in statuses:
            state = self._states.get(status.space_id)
            if state is None:
                self._states[status.space_id] = _TemporalState(stable=status.state)
                smoothed.append(status)
                continue
            if status.state == state.stable:
                state.candidate = None
                state.count = 0
                state.candidate_start = None
                smoothed.append(status)
                continue
            if status.state == state.candidate:
                state.count += 1
            else:
                state.candidate = status.state
                state.count = 1
                state.candidate_start = timestamp
            required = self.empty_frames if status.state == OccupancyState.EMPTY else self.occupied_frames
            time_ok = True
            if (
                status.state == OccupancyState.EMPTY
                and self.vacated_seconds > 0
                and timestamp is not None
                and state.candidate_start is not None
            ):
                time_ok = (timestamp - state.candidate_start) >= self.vacated_seconds
            if state.count >= required and time_ok:
                state.stable = status.state
                state.candidate = None
                state.count = 0
                state.candidate_start = None
                smoothed.append(status)
                continue
            delayed = SpaceStatus(
                space_id=status.space_id,
                state=state.stable,
                confidence=max(0.3, status.confidence * 0.75),
                method=f"temporal_{status.method}",
                associated_track_ids=status.associated_track_ids,
                space_coverage=status.space_coverage,
                classifier_empty_probability=status.classifier_empty_probability,
                classifier_occupied_probability=status.classifier_occupied_probability,
                visibility=status.visibility,
                reasons=status.reasons + ["temporal_pending"],
                timestamp=status.timestamp,
            )
            smoothed.append(delayed)
        return smoothed


class OccupancyEngine:
    """Applies the project occupancy rules to spaces, vehicles and classifier scores."""

    def __init__(self, config: Config, classifier: OccupancyClassifier | None = None):
        self.config = config
        self.classifier = classifier or OccupancyClassifier(config)
        thresholds = config.get("thresholds", {}) or {}
        self.occupied_probability = float(thresholds.get("occupied_probability", 0.60))
        self.empty_probability = float(thresholds.get("empty_probability", 0.60))
        self.occupied_with_vehicle_probability = float(
            thresholds.get("occupied_with_vehicle_probability", 0.35)
        )
        self.detector_overlap_occupied = float(thresholds.get("detector_overlap_occupied", 0.30))
        self.detector_overlap_min = float(thresholds.get("detector_overlap_min", 0.08))
        self.ambiguous_overlap_ratio = float(thresholds.get("ambiguous_overlap_ratio", 0.75))
        self.min_detection_confidence = float(thresholds.get("min_detection_confidence", 0.30))
        self.min_visibility = float(thresholds.get("min_visibility_score", 0.45))
        self.min_brightness = float(thresholds.get("min_brightness", 35.0))
        self.min_laplacian = float(thresholds.get("min_laplacian_variance", 25.0))

    def _crop(self, image: np.ndarray, space: ParkingSpace) -> np.ndarray | None:
        polygon = expand_polygon(space.polygon, 0.02, image.shape)
        height, width = image.shape[:2]
        min_x = max(0, int(min(point[0] for point in polygon)) - 3)
        max_x = min(width, int(max(point[0] for point in polygon)) + 4)
        min_y = max(0, int(min(point[1] for point in polygon)) - 3)
        max_y = min(height, int(max(point[1] for point in polygon)) + 4)
        if max_x - min_x < 8 or max_y - min_y < 8:
            return None
        local = image[min_y:max_y, min_x:max_x]
        local_polygon = [(x - min_x, y - min_y) for x, y in polygon]
        size = self.classifier.input_size
        crop = warp_polygon_crop(local, local_polygon, (size, size))
        return crop if crop.size > 0 else None

    def _overlaps(
        self, space: ParkingSpace, detections: Sequence[Detection]
    ) -> tuple[float, float, list[int], float]:
        area = polygon_area(space.polygon)
        if area <= 0:
            return 0.0, 0.0, [], 0.0
        coverage = 0.0
        best_confidence = 0.0
        track_ids: list[int] = []
        for detection in detections:
            if detection.confidence < self.min_detection_confidence:
                continue
            intersection = bbox_polygon_overlap(detection.xyxy, space.polygon)
            if intersection <= 0:
                continue
            ratio = float(intersection / area)
            coverage += ratio
            best_confidence = max(best_confidence, float(detection.confidence))
            if detection.track_id is not None:
                track_ids.append(int(detection.track_id))
        return float(min(1.0, coverage)), best_confidence, track_ids, float(min(1.0, coverage))

    def classify(
        self,
        image: np.ndarray,
        spaces: Sequence[ParkingSpace],
        detections: Sequence[Detection],
        timestamp: float | None = None,
        use_classifier: bool = True,
    ) -> list[SpaceStatus]:
        height, width = image.shape[:2]
        overlap_data = [self._overlaps(space, detections) for space in spaces]
        crops: list[np.ndarray | None] = []
        qualifying: list[int] = []
        for index, space in enumerate(spaces):
            visibility = visible_fraction(space.polygon, width, height)
            if not use_classifier or not self.classifier.available or visibility < self.min_visibility:
                crops.append(None)
                continue
            crop = self._crop(image, space)
            if crop is None:
                crops.append(None)
                continue
            quality = crop_quality(crop)
            if quality["brightness"] < self.min_brightness or quality["laplacian_variance"] < self.min_laplacian:
                crops.append(None)
                continue
            crops.append(crop)
            qualifying.append(index)

        probabilities = np.full((len(spaces), len(self.classifier.classes)), np.nan, dtype=np.float32)
        class_lookup = {name: idx for idx, name in enumerate(self.classifier.classes)}
        empty_idx = class_lookup.get("EMPTY", 0)
        occupied_idx = class_lookup.get("OCCUPIED", 1)
        if qualifying:
            valid_crops = [crop for crop in crops if crop is not None]
            predictions = self.classifier.predict(valid_crops)
            for position, index in enumerate(qualifying):
                if position < len(predictions):
                    probabilities[index, empty_idx] = predictions[position, empty_idx]
                    probabilities[index, occupied_idx] = predictions[position, occupied_idx]

        statuses: list[SpaceStatus] = []
        for index, space in enumerate(spaces):
            coverage, best_confidence, track_ids, _ = overlap_data[index]
            visibility = visible_fraction(space.polygon, width, height)
            empty_p = (
                float(probabilities[index, empty_idx])
                if not np.isnan(probabilities[index, empty_idx])
                else None
            )
            occupied_p = (
                float(probabilities[index, occupied_idx])
                if not np.isnan(probabilities[index, occupied_idx])
                else None
            )
            status = self._decide(
                space=space,
                coverage=coverage,
                best_confidence=best_confidence,
                track_ids=track_ids,
                visibility=visibility,
                empty_p=empty_p,
                occupied_p=occupied_p,
                timestamp=timestamp,
            )
            statuses.append(status)
        return statuses

    def _decide(
        self,
        space: ParkingSpace,
        coverage: float,
        best_confidence: float,
        track_ids: list[int],
        visibility: float,
        empty_p: float | None,
        occupied_p: float | None,
        timestamp: float | None,
    ) -> SpaceStatus:
        if visibility < self.min_visibility:
            return SpaceStatus(
                space_id=space.space_id,
                state=OccupancyState.UNCERTAIN,
                confidence=0.0,
                method="visibility",
                visibility=visibility,
                reasons=[f"space_visible_fraction={visibility:.2f}"],
                timestamp=timestamp,
            )
        reasons: list[str] = []
        if coverage > 0:
            reasons.append(f"vehicle_coverage={coverage:.2f}")
        if best_confidence > 0:
            reasons.append(f"vehicle_confidence={best_confidence:.2f}")
        if empty_p is not None:
            reasons.append(f"classifier_empty={empty_p:.2f}")
        if occupied_p is not None:
            reasons.append(f"classifier_occupied={occupied_p:.2f}")

        if coverage >= self.detector_overlap_occupied:
            return SpaceStatus(
                space_id=space.space_id,
                state=OccupancyState.OCCUPIED,
                confidence=max(0.5, best_confidence, min(1.0, coverage)),
                method="vehicle_overlap",
                associated_track_ids=track_ids,
                space_coverage=coverage,
                classifier_empty_probability=empty_p,
                classifier_occupied_probability=occupied_p,
                visibility=visibility,
                reasons=reasons + ["rule:vehicle_covers_space"],
                timestamp=timestamp,
            )

        if coverage >= self.detector_overlap_min:
            if occupied_p is not None and occupied_p >= self.occupied_with_vehicle_probability:
                return SpaceStatus(
                    space_id=space.space_id,
                    state=OccupancyState.OCCUPIED,
                    confidence=max(occupied_p, best_confidence, 0.5),
                    method="vehicle_classifier",
                    associated_track_ids=track_ids,
                    space_coverage=coverage,
                    classifier_empty_probability=empty_p,
                    classifier_occupied_probability=occupied_p,
                    visibility=visibility,
                    reasons=reasons + ["rule:partial_vehicle_and_classifier"],
                    timestamp=timestamp,
                )
            if occupied_p is not None and empty_p is not None:
                if empty_p >= self.empty_probability and occupied_p < self.occupied_with_vehicle_probability:
                    return SpaceStatus(
                        space_id=space.space_id,
                        state=OccupancyState.EMPTY,
                        confidence=empty_p,
                        method="classifier_empty",
                        space_coverage=coverage,
                        classifier_empty_probability=empty_p,
                        classifier_occupied_probability=occupied_p,
                        visibility=visibility,
                        reasons=reasons + ["rule:classifier_empty_overrides_small_overlap"],
                        timestamp=timestamp,
                    )
            if best_confidence >= max(0.5, self.min_detection_confidence):
                return SpaceStatus(
                    space_id=space.space_id,
                    state=OccupancyState.OCCUPIED,
                    confidence=max(0.5, best_confidence),
                    method="vehicle_any_part",
                    associated_track_ids=track_ids,
                    space_coverage=coverage,
                    classifier_empty_probability=empty_p,
                    classifier_occupied_probability=occupied_p,
                    visibility=visibility,
                    reasons=reasons + ["rule:any_vehicle_part_occupies_space"],
                    timestamp=timestamp,
                )

        if occupied_p is not None and occupied_p >= self.occupied_probability:
            state = OccupancyState.OCCUPIED if occupied_p >= self.occupied_with_vehicle_probability else OccupancyState.UNCERTAIN
            return SpaceStatus(
                space_id=space.space_id,
                state=state,
                confidence=occupied_p,
                method="classifier_occupied",
                associated_track_ids=track_ids,
                space_coverage=coverage,
                classifier_empty_probability=empty_p,
                classifier_occupied_probability=occupied_p,
                visibility=visibility,
                reasons=reasons + ["rule:classifier_occupied"],
                timestamp=timestamp,
            )

        if empty_p is not None and empty_p >= self.empty_probability and coverage < self.detector_overlap_min:
            return SpaceStatus(
                space_id=space.space_id,
                state=OccupancyState.EMPTY,
                confidence=empty_p,
                method="classifier_empty",
                space_coverage=coverage,
                classifier_empty_probability=empty_p,
                classifier_occupied_probability=occupied_p,
                visibility=visibility,
                reasons=reasons + ["rule:classifier_empty_no_vehicle"],
                timestamp=timestamp,
            )

        if coverage <= 0.0 and empty_p is None:
            return SpaceStatus(
                space_id=space.space_id,
                state=OccupancyState.EMPTY,
                confidence=0.60 if best_confidence == 0 else 0.50,
                method="no_vehicle_detected",
                space_coverage=coverage,
                visibility=visibility,
                reasons=reasons + ["rule:no_vehicle_overlap"],
                timestamp=timestamp,
            )

        return SpaceStatus(
            space_id=space.space_id,
            state=OccupancyState.UNCERTAIN,
            confidence=0.5,
            method="ambiguous",
            associated_track_ids=track_ids,
            space_coverage=coverage,
            classifier_empty_probability=empty_p,
            classifier_occupied_probability=occupied_p,
            visibility=visibility,
            reasons=reasons + ["rule:ambiguous_evidence"],
            timestamp=timestamp,
        )


def summarize(states: Sequence[SpaceStatus], detections: Sequence[Detection]) -> Statistics:
    stats = Statistics(
        total_spaces=len(states),
        detected_vehicles=len(detections),
        occupied_spaces=sum(1 for state in states if state.state == OccupancyState.OCCUPIED),
        empty_spaces=sum(1 for state in states if state.state == OccupancyState.EMPTY),
        uncertain_spaces=sum(1 for state in states if state.state == OccupancyState.UNCERTAIN),
        parked_vehicles=sum(1 for state in states if state.state == OccupancyState.OCCUPIED),
    )
    return stats
