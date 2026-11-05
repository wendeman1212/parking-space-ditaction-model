"""Shared data contracts for all pipeline components."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class OccupancyState(str, Enum):
    OCCUPIED = "OCCUPIED"
    EMPTY = "EMPTY"
    UNCERTAIN = "UNCERTAIN"


class SceneStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    POSSIBLE = "POSSIBLE"
    NON_PARKING = "NON_PARKING"
    INVALID = "INVALID"
    INSUFFICIENT = "INSUFFICIENT"


class SpaceSource(str, Enum):
    AUTOMATIC = "automatic"
    CALIBRATED = "calibrated"
    REFERENCE = "reference_annotation"


class InputKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    CAMERA = "camera"


@dataclass
class ValidationResult:
    valid: bool
    kind: InputKind | None
    message: str
    width: int = 0
    height: int = 0
    frame_count: int = 0
    fps: float = 0.0
    duration_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "kind": self.kind.value if self.kind else None,
            "message": self.message,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None
    mask: np.ndarray | None = None
    frame_index: int | None = None
    timestamp: float | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.xyxy[2] - self.xyxy[0])

    @property
    def height(self) -> float:
        return max(0.0, self.xyxy[3] - self.xyxy[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xyxy[0] + self.xyxy[2]) / 2.0, (self.xyxy[1] + self.xyxy[3]) / 2.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 5),
            "bbox_xyxy": [round(float(v), 2) for v in self.xyxy],
            "track_id": int(self.track_id) if self.track_id is not None else None,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "has_mask": self.mask is not None,
        }


@dataclass
class ParkingSpace:
    space_id: str
    polygon: list[tuple[float, float]]
    source: SpaceSource
    confidence: float = 1.0
    method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def centroid(self) -> tuple[float, float]:
        from .geometry import polygon_centroid

        return polygon_centroid(self.polygon)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def area(self) -> float:
        from .geometry import polygon_area

        return polygon_area(self.polygon)

    def to_dict(self) -> dict[str, Any]:
        return {
            "space_id": self.space_id,
            "polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in self.polygon],
            "bbox_xyxy": [round(float(v), 2) for v in self.bbox],
            "source": self.source.value,
            "confidence": round(float(self.confidence), 4),
            "method": self.method,
            "metadata": self.metadata,
        }


@dataclass
class SpaceStatus:
    space_id: str
    state: OccupancyState
    confidence: float
    method: str
    associated_track_ids: list[int] = field(default_factory=list)
    space_coverage: float = 0.0
    classifier_empty_probability: float | None = None
    classifier_occupied_probability: float | None = None
    visibility: float = 1.0
    reasons: list[str] = field(default_factory=list)
    timestamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "space_id": self.space_id,
            "state": self.state.value,
            "confidence": round(float(self.confidence), 4),
            "method": self.method,
            "associated_track_ids": self.associated_track_ids,
            "space_coverage": round(float(self.space_coverage), 4),
            "classifier_empty_probability": (
                round(float(self.classifier_empty_probability), 4)
                if self.classifier_empty_probability is not None
                else None
            ),
            "classifier_occupied_probability": (
                round(float(self.classifier_occupied_probability), 4)
                if self.classifier_occupied_probability is not None
                else None
            ),
            "visibility": round(float(self.visibility), 4),
            "reasons": self.reasons,
            "timestamp": self.timestamp,
        }


@dataclass
class SceneAssessment:
    status: SceneStatus
    confidence: float
    reason: str
    parking_probability: float | None = None
    layout_spaces: int = 0
    layout_confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def occupancy_reportable(self) -> bool:
        return self.status in (SceneStatus.CONFIRMED, SceneStatus.POSSIBLE) and self.layout_spaces > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "confidence": round(float(self.confidence), 4),
            "reason": self.reason,
            "parking_probability": (
                round(float(self.parking_probability), 4) if self.parking_probability is not None else None
            ),
            "layout_spaces": self.layout_spaces,
            "layout_confidence": round(float(self.layout_confidence), 4),
            "occupancy_reportable": self.occupancy_reportable,
            "evidence": self.evidence,
        }


@dataclass
class Statistics:
    total_spaces: int = 0
    occupied_spaces: int = 0
    empty_spaces: int = 0
    uncertain_spaces: int = 0
    detected_vehicles: int = 0
    parked_vehicles: int = 0
    entries: int = 0
    exits: int = 0

    @property
    def occupancy_percentage(self) -> float | None:
        if self.total_spaces <= 0:
            return None
        return round(100.0 * self.occupied_spaces / self.total_spaces, 2)

    @property
    def available_spaces(self) -> int:
        return self.empty_spaces

    def validate(self) -> bool:
        return self.total_spaces == self.occupied_spaces + self.empty_spaces + self.uncertain_spaces

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_spaces": self.total_spaces,
            "occupied_spaces": self.occupied_spaces,
            "empty_spaces": self.empty_spaces,
            "uncertain_spaces": self.uncertain_spaces,
            "available_spaces": self.available_spaces,
            "occupancy_percentage": self.occupancy_percentage,
            "detected_vehicles": self.detected_vehicles,
            "parked_vehicles": self.parked_vehicles,
            "entries": self.entries,
            "exits": self.exits,
            "counts_consistent": self.validate(),
        }


@dataclass
class ParkingEvent:
    event_id: str
    event_type: str
    timestamp: float
    frame_index: int | None = None
    track_id: int | None = None
    space_id: str | None = None
    vehicle_class: str | None = None
    entrance_id: str | None = None
    exit_id: str | None = None
    previous_state: str | None = None
    new_state: str | None = None
    confidence: float = 0.0
    frame_reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": round(float(self.timestamp), 3),
            "frame_index": self.frame_index,
            "track_id": self.track_id,
            "space_id": self.space_id,
            "vehicle_class": self.vehicle_class,
            "entrance_id": self.entrance_id,
            "exit_id": self.exit_id,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "confidence": round(float(self.confidence), 4),
            "frame_reference": self.frame_reference,
            "metadata": self.metadata,
        }


@dataclass
class TrackState:
    track_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    previous_bbox: tuple[float, float, float, float] | None
    first_seen: float
    last_seen: float
    positions: list[tuple[float, float, float]] = field(default_factory=list)
    space_id: str | None = None
    state: str = "active"
    consecutive_frames: int = 1
    missed_frames: int = 0

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "bbox_xyxy": [round(float(v), 2) for v in self.bbox],
            "previous_bbox_xyxy": (
                [round(float(v), 2) for v in self.previous_bbox] if self.previous_bbox else None
            ),
            "first_seen": round(float(self.first_seen), 3),
            "last_seen": round(float(self.last_seen), 3),
            "space_id": self.space_id,
            "state": self.state,
            "consecutive_frames": self.consecutive_frames,
            "missed_frames": self.missed_frames,
            "positions": [[round(float(v), 2) for v in p] for p in self.positions[-30:]],
        }
