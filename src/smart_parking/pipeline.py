"""End-to-end smart-parking pipeline for images and videos."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .config import Config, load_app_config, load_component_config
from .io_utils import imread_unicode, imwrite_unicode, resolve_path, save_json, validate_image, validate_video
from .models.occupancy import OccupancyEngine, TemporalSmoother, summarize
from .models.scene import SceneAssessor
from .models.spaces import ParkingSpaceDetector
from .models.vehicle import VehicleDetector
from .types import (
    Detection,
    OccupancyState,
    ParkingEvent,
    ParkingSpace,
    SceneAssessment,
    SpaceStatus,
)
from .visualization import annotate_frame


class ParkingPipeline:
    """Coordinates vehicle detection, space detection, occupancy rules and reporting."""

    def __init__(
        self,
        config: Config | None = None,
        vehicle_detector: VehicleDetector | None = None,
        space_detector: ParkingSpaceDetector | None = None,
        occupancy_engine: OccupancyEngine | None = None,
        scene_assessor: SceneAssessor | None = None,
    ):
        self.config = config or load_app_config()
        self.vehicle_config = load_component_config("vehicle_detector", "tracker")
        self.space_config = load_component_config("parking_space_detector")
        self.occupancy_config = load_component_config("occupancy_classifier")
        self.scene_config = load_component_config("scene_classifier")
        self.vehicle_detector = vehicle_detector or VehicleDetector(self.vehicle_config)
        self.space_detector = space_detector or ParkingSpaceDetector(self.space_config)
        self.occupancy_engine = occupancy_engine or OccupancyEngine(self.occupancy_config)
        self.scene_assessor = scene_assessor or SceneAssessor(self.scene_config)

    def _detect_spaces(
        self,
        image: np.ndarray,
        layout: dict[str, Any] | str | Path | None,
    ) -> list[ParkingSpace]:
        return self.space_detector.detect(image, layout=layout)

    def _space_payload(self, spaces: Sequence[ParkingSpace], states: Sequence[SpaceStatus]) -> list[dict[str, Any]]:
        status_lookup = {status.space_id: status for status in states}
        payload: list[dict[str, Any]] = []
        for space in spaces:
            item = space.to_dict()
            status = status_lookup.get(space.space_id)
            item["occupancy"] = status.to_dict() if status else None
            payload.append(item)
        return payload

    def _write_json(self, relative: str, data: Any) -> str:
        path = self.config.path("paths.outputs") / relative
        save_json(path, data)
        return str(path)

    def _write_annotation(self, relative: str, image: np.ndarray) -> str:
        path = self.config.path("paths.outputs") / relative
        imwrite_unicode(path, image, quality=int(self.config.get("processing.jpeg_quality", 92)))
        return str(path)

    def process_image(
        self,
        source: str | Path,
        layout: dict[str, Any] | str | Path | None = None,
        save: bool = True,
        output_name: str | None = None,
    ) -> dict[str, Any]:
        path = resolve_path(source)
        validation = validate_image(
            path,
            min_side=int(self.config.get("validation.min_image_side", 64)),
            max_side=int(self.config.get("validation.max_image_side", 8192)),
        )
        if not validation.valid:
            raise ValueError(validation.message)
        image = imread_unicode(path)
        if image is None:
            raise ValueError(f"Could not decode image: {path}")
        timings: dict[str, float] = {}

        started = time.perf_counter()
        spaces = self._detect_spaces(image, layout)
        timings["space_detection_ms"] = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        detections = self.vehicle_detector.detect(image)
        timings["vehicle_detection_ms"] = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        states = self.occupancy_engine.classify(image, spaces, detections, timestamp=0.0)
        timings["occupancy_ms"] = (time.perf_counter() - started) * 1000.0

        scene = self.scene_assessor.assess(image, spaces)
        stats = summarize(states, detections)
        annotated = annotate_frame(image, spaces, states, detections, scene, stats)

        result: dict[str, Any] = {
            "source": str(path),
            "kind": "image",
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "scene": scene.to_dict(),
            "spaces": self._space_payload(spaces, states),
            "detections": [detection.to_dict() for detection in detections],
            "rejected_regions": getattr(self.space_detector, "last_rejected", []),
            "statistics": stats.to_dict(),
            "occupancy_reportable": scene.occupancy_reportable,
            "timings_ms": {key: round(value, 2) for key, value in timings.items()},
        }
        if save:
            stem = output_name or path.stem
            result["annotated_image"] = self._write_annotation(f"annotated/{stem}.jpg", annotated)
            result["result_json"] = self._write_json(f"results/{stem}.json", result)
        return result

    def process_video(
        self,
        source: str | Path,
        layout: dict[str, Any] | str | Path | None = None,
        save: bool = True,
        output_name: str | None = None,
        max_frames: int | None = None,
        sample_fps: float | None = None,
    ) -> dict[str, Any]:
        path = resolve_path(source)
        validation = validate_video(path)
        if not validation.valid:
            raise ValueError(validation.message)
        capture = cv2.VideoCapture(str(path))
        fps = validation.fps if validation.fps > 0 else 25.0
        sample = sample_fps if sample_fps is not None else float(self.config.get("processing.video_sample_fps", 0) or 0)
        step = max(1, int(round(fps / sample))) if sample > 0 else 1
        limit = max_frames if max_frames is not None else int(self.config.get("processing.max_video_frames", 0) or 0)

        stem = output_name or path.stem
        writer = None
        annotated_path: str | None = None
        if save:
            annotated_path = str(self.config.path("paths.outputs") / f"videos/{stem}.mp4")
            Path(annotated_path).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                annotated_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps / step,
                (validation.width, validation.height),
            )
            if not writer.isOpened():
                writer = None
                annotated_path = None

        smoother = TemporalSmoother(self.config)
        spaces: list[ParkingSpace] = []
        scene: SceneAssessment | None = None
        detections: list[Detection] = []
        states: list[SpaceStatus] = []
        previous_states: dict[str, OccupancyState] = {}
        events: list[ParkingEvent] = []
        timeline: list[dict[str, Any]] = []
        frame_index = 0
        processed = 0
        stats = None
        self.vehicle_detector.reset_tracking()

        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if frame_index % step != 0:
                    frame_index += 1
                    continue
                timestamp = frame_index / fps
                if not spaces:
                    spaces = self._detect_spaces(frame, layout)
                    scene = self.scene_assessor.assess(frame, spaces)
                detections = self.vehicle_detector.track(frame, persist=True)
                for detection in detections:
                    detection.frame_index = frame_index
                    detection.timestamp = timestamp
                observed = self.occupancy_engine.classify(frame, spaces, detections, timestamp=timestamp)
                states = smoother.update(observed, timestamp=timestamp)
                stats = summarize(states, detections)
                if processed > 0:
                    for status in states:
                        previous = previous_states.get(status.space_id)
                        if previous is None or previous == status.state:
                            continue
                        if {previous, status.state} <= {OccupancyState.EMPTY, OccupancyState.OCCUPIED}:
                            events.append(
                                ParkingEvent(
                                    event_id=f"{stem}_{status.space_id}_{len(events):04d}",
                                    event_type=(
                                        "space_occupied"
                                        if status.state == OccupancyState.OCCUPIED
                                        else "space_vacated"
                                    ),
                                    timestamp=timestamp,
                                    frame_index=frame_index,
                                    space_id=status.space_id,
                                    previous_state=previous.value,
                                    new_state=status.state.value,
                                    confidence=status.confidence,
                                )
                            )
                previous_states = {status.space_id: status.state for status in states}
                annotated = annotate_frame(frame, spaces, states, detections, scene, stats)
                if writer is not None:
                    writer.write(annotated)
                timeline.append(
                    {
                        "frame_index": frame_index,
                        "timestamp": round(timestamp, 3),
                        "statistics": stats.to_dict(),
                    }
                )
                processed += 1
                frame_index += 1
                if limit and processed >= limit:
                    break
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        stats = stats or summarize([], [])
        stats.entries = sum(1 for event in events if event.event_type == "space_occupied")
        stats.exits = sum(1 for event in events if event.event_type == "space_vacated")
        result: dict[str, Any] = {
            "source": str(path),
            "kind": "video",
            "width": validation.width,
            "height": validation.height,
            "fps": round(fps, 3),
            "frame_count": validation.frame_count,
            "frames_processed": processed,
            "sample_step": step,
            "scene": scene.to_dict() if scene else None,
            "spaces": self._space_payload(spaces, states),
            "statistics": stats.to_dict(),
            "events": [event.to_dict() for event in events],
            "timeline": timeline,
            "occupancy_reportable": bool(scene and scene.occupancy_reportable),
        }
        if save:
            if annotated_path:
                result["annotated_video"] = annotated_path
            result["result_json"] = self._write_json(f"results/{stem}.json", result)
        return result

    def process(
        self,
        source: str | Path,
        layout: dict[str, Any] | str | Path | None = None,
        save: bool = True,
        output_name: str | None = None,
    ) -> dict[str, Any]:
        path = resolve_path(source)
        suffix = path.suffix.lower()
        if suffix in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            return self.process_video(path, layout=layout, save=save, output_name=output_name)
        return self.process_image(path, layout=layout, save=save, output_name=output_name)
