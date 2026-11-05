"""Annotated overlays for parking spaces, vehicles and occupancy summaries."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .types import Detection, OccupancyState, SceneAssessment, SpaceStatus, Statistics

_COLORS = {
    OccupancyState.EMPTY: (80, 200, 80),
    OccupancyState.OCCUPIED: (60, 60, 220),
    OccupancyState.UNCERTAIN: (140, 140, 140),
}
_VEHICLE_COLOR = (220, 170, 40)
_TEXT_COLOR = (255, 255, 255)


def _draw_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    top = max(0, y - height - 6)
    cv2.rectangle(image, (x, top), (x + width + 6, top + height + baseline + 4), color, -1)
    cv2.putText(
        image,
        text,
        (x + 3, top + height + 1),
        font,
        scale,
        _TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def annotate_frame(
    image: np.ndarray,
    spaces: Sequence,
    states: Sequence[SpaceStatus],
    detections: Sequence[Detection] = (),
    scene: SceneAssessment | None = None,
    stats: Statistics | None = None,
) -> np.ndarray:
    output = image.copy()
    status_lookup = {status.space_id: status for status in states}
    for space in spaces:
        status = status_lookup.get(space.space_id)
        color = _COLORS.get(status.state, (200, 200, 200)) if status else (200, 200, 200)
        polygon = np.asarray(space.polygon, dtype=np.int32).reshape(-1, 1, 2)
        # Fill the space polygon with a translucent color for occupancy
        overlay = output.copy()
        cv2.fillPoly(overlay, [polygon], color)
        cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)
        # Draw polygon outline
        cv2.polylines(output, [polygon], True, color, 2, cv2.LINE_AA)
        center = space.centroid
        text = space.space_id if status is None else f"{space.space_id} {status.state.value}"
        _draw_label(output, text, (int(center[0]), int(center[1])), color)

    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.xyxy)
        cv2.rectangle(output, (x1, y1), (x2, y2), _VEHICLE_COLOR, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        if detection.track_id is not None:
            label = f"#{detection.track_id} {label}"
        _draw_label(output, label, (x1, max(12, y1)), _VEHICLE_COLOR)

    lines = []
    if scene is not None and not scene.occupancy_reportable:
        lines.append(f"OCCUPANCY: UNAVAILABLE ({scene.reason})")
        lines.append(f"detected vehicles: {stats.detected_vehicles if stats else len(detections)}")
    else:
        if stats is not None:
            lines.append(
                f"spaces {stats.total_spaces} | occupied {stats.occupied_spaces} | "
                f"empty {stats.empty_spaces} | uncertain {stats.uncertain_spaces} | "
                f"vehicles {stats.detected_vehicles}"
            )
        if scene is not None:
            lines.append(f"scene {scene.status.value} ({scene.reason})")
    if lines:
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 24
        for line in lines:
            (width, height), _baseline = cv2.getTextSize(line, font, 0.55, 1)
            cv2.rectangle(output, (8, y - height - 6), (18 + width, y + 6), (20, 20, 20), -1)
            cv2.putText(output, line, (12, y), font, 0.55, _TEXT_COLOR, 1, cv2.LINE_AA)
            y += height + 14
    return output
