"""Parking-space detection from reference layouts, OBB models and parking-line geometry."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from ..config import PROJECT_ROOT, Config
from ..geometry import (
    bbox_polygon_overlap,
    polygon_area,
    polygon_centroid,
    polygon_iou,
    visible_fraction,
)
from ..io_utils import load_json, resolve_path
from ..types import ParkingSpace, SpaceSource


def _source_from_string(value: str | None) -> SpaceSource:
    if value == "automatic":
        return SpaceSource.AUTOMATIC
    if value == "calibrated":
        return SpaceSource.CALIBRATED
    return SpaceSource.REFERENCE


def spaces_from_layout(
    layout: dict[str, Any] | str | Path,
    source: SpaceSource | None = None,
) -> list[ParkingSpace]:
    """Convert a saved layout document into ``ParkingSpace`` objects."""
    if isinstance(layout, (str, Path)):
        layout = load_json(layout)
    if not isinstance(layout, dict):
        raise ValueError("Layout must be a mapping or a path to a JSON layout")
    spaces: list[ParkingSpace] = []
    for index, item in enumerate(layout.get("spaces", []), start=1):
        polygon = [(float(point[0]), float(point[1])) for point in item.get("polygon", [])]
        if len(polygon) < 3:
            continue
        item_source = source or _source_from_string(item.get("source"))
        spaces.append(
            ParkingSpace(
                space_id=str(item.get("space_id") or f"P{index:03d}"),
                polygon=polygon,
                source=item_source,
                confidence=float(item.get("confidence", 1.0)),
                method=str(item.get("method", "layout")),
                metadata=dict(item.get("metadata", {}) or {}),
            )
        )
    return spaces


def find_layout_for_frame(layout_dir: str | Path, subset: str, image_id: int) -> Path | None:
    directory = resolve_path(layout_dir)
    candidate = directory / f"{subset}_{int(image_id)}.json"
    return candidate if candidate.exists() else None


class ParkingSpaceDetector:
    """Detects parking slots using an OBB model and/or parking-line geometry."""

    def __init__(self, config: Config, device: str | None = None, lazy: bool = True):
        self.config = config
        self.mode = str(config.get("mode", "combined"))
        self.device = device or str(config.get("obb_model.device", "cpu"))
        self.imgsz = int(config.get("obb_model.imgsz", 640))
        self.confidence = float(config.get("obb_model.confidence", 0.25))
        self.iou = float(config.get("obb_model.iou", 0.45))
        self.weights = self._resolve_weights()
        self.min_space_area = float(config.get("automatic.min_space_area_px", 900))
        self.max_space_fraction = float(config.get("automatic.max_space_area_fraction", 0.08))
        self.min_aspect = float(config.get("automatic.min_aspect_ratio", 0.30))
        self.max_aspect = float(config.get("automatic.max_aspect_ratio", 3.50))
        self.min_layout_spaces = int(config.get("automatic.min_layout_spaces", 3))
        self.min_layout_confidence = float(config.get("automatic.min_layout_confidence", 0.25))
        self.max_detections = int(config.get("automatic.max_detections", 300))
        self.duplicate_iou = float(config.get("geometry.duplicate_iou", 0.80))
        self.min_bbox_side = float(config.get("geometry.min_bbox_side_px", 24))
        self.expand_fraction = float(config.get("geometry.expand_fraction", 0.0))
        self.min_ground_y_ratio = float(config.get("automatic.validation.min_ground_y_ratio", 0.14))
        self.max_internal_std = float(config.get("automatic.validation.max_internal_std", 55.0))
        self.max_edge_density = float(config.get("automatic.validation.max_edge_density", 0.11))
        self.max_saturation = float(config.get("automatic.validation.max_saturation", 85.0))
        self.min_brightness = float(config.get("automatic.validation.min_brightness", 30.0))
        self.max_brightness = float(config.get("automatic.validation.max_brightness", 220.0))
        self.last_rejected: list[dict[str, Any]] = []
        self._model: Any | None = None
        if not lazy:
            self._load()

    def _resolve_weights(self) -> str:
        configured = self.config.get("obb_model.path")
        if configured and resolve_path(configured).exists():
            return str(resolve_path(configured))
        architecture = str(self.config.get("obb_model.architecture", "yolo11n-obb.pt"))
        local = PROJECT_ROOT / architecture
        if local.exists():
            return str(local)
        return architecture

    @property
    def model_available(self) -> bool:
        configured = self.config.get("obb_model.path")
        return bool(configured and resolve_path(configured).exists())

    def _load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        self._model = YOLO(self.weights)

    def detect(
        self,
        image: np.ndarray,
        layout: dict[str, Any] | str | Path | None = None,
        mode: str | None = None,
    ) -> list[ParkingSpace]:
        self.last_rejected = []
        mode = (mode or self.mode).lower()
        height, width = image.shape[:2]
        if layout is not None:
            spaces = spaces_from_layout(layout)
            return self._finalize_layout(spaces)
        if mode == "calibrated":
            return []
        if mode == "automatic":
            spaces = self._detect_lines(image)
            if not spaces and self.model_available:
                spaces = self._detect_obb(image)
            return self._finalize(spaces, width, height, image=image)
        spaces = self._detect_obb(image) if self.model_available else []
        if mode == "combined" and len(spaces) < self.min_layout_spaces:
            automatic = self._detect_lines(image)
            spaces = self._merge(spaces, automatic)
        return self._finalize(spaces, width, height, image=image)

    def _detect_obb(self, image: np.ndarray) -> list[ParkingSpace]:
        self._load()
        results = self._model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        spaces: list[ParkingSpace] = []
        for result in results:
            obb = getattr(result, "obb", None)
            if obb is None or len(obb) == 0:
                continue
            polygons = obb.xyxyxyxy.cpu().numpy()
            confidences = obb.conf.cpu().numpy()
            for polygon, score in zip(polygons, confidences):
                points = [(float(point[0]), float(point[1])) for point in polygon]
                if len(points) < 3:
                    continue
                spaces.append(
                    ParkingSpace(
                        space_id="",
                        polygon=points,
                        source=SpaceSource.AUTOMATIC,
                        confidence=float(score),
                        method="obb_model",
                    )
                )
        return spaces

    def _detect_lines(self, image: np.ndarray) -> list[ParkingSpace]:
        lines = self._hough_lines(image)
        if len(lines) < 2:
            return []
        dominant = self._dominant_angle(lines)
        tolerance = math.radians(float(self.config.get("automatic.line_detection.angle_tolerance_deg", 22.0)))
        parallel = [line for line in lines if self._angular_distance(line[5], dominant) <= tolerance]
        if len(parallel) < 2:
            return []
        theta = dominant
        unit = np.array([math.cos(theta), math.sin(theta)])
        normal = np.array([-math.sin(theta), math.cos(theta)])
        projected = []
        for x1, y1, x2, y2, length, _angle in parallel:
            p1 = np.array([x1, y1], dtype=float)
            p2 = np.array([x2, y2], dtype=float)
            s1 = float(np.dot(p1, unit))
            s2 = float(np.dot(p2, unit))
            rho = float(np.dot((p1 + p2) / 2.0, normal))
            projected.append((rho, min(s1, s2), max(s1, s2), length))
        projected.sort(key=lambda item: item[0])
        min_gap = max(12.0, self.min_bbox_side * 0.8)
        max_gap = max(min_gap + 1.0, math.sqrt(self.max_space_fraction * image.shape[0] * image.shape[1]))
        min_overlap = float(self.config.get("automatic.line_detection.min_line_length", 40)) * 0.55
        spaces: list[ParkingSpace] = []
        for first, second in zip(projected, projected[1:]):
            rho1, s1_min, s1_max, len1 = first
            rho2, s2_min, s2_max, len2 = second
            gap = rho2 - rho1
            if gap < min_gap or gap > max_gap:
                continue
            lo = max(s1_min, s2_min)
            hi = min(s1_max, s2_max)
            if hi - lo < min_overlap:
                continue
            points = [
                tuple(unit * lo + normal * rho1),
                tuple(unit * hi + normal * rho1),
                tuple(unit * hi + normal * rho2),
                tuple(unit * lo + normal * rho2),
            ]
            polygon = [(float(x), float(y)) for x, y in points]
            coverage = min(len1, len2) / max(gap, 1.0)
            score = float(min(0.95, 0.45 + 0.25 * min(2.0, coverage) + 0.1 * min(len1, len2) / 100.0))
            spaces.append(
                ParkingSpace(
                    space_id="",
                    polygon=polygon,
                    source=SpaceSource.AUTOMATIC,
                    confidence=score,
                    method="line_pair",
                )
            )
        return spaces

    def _hough_lines(self, image: np.ndarray) -> list[tuple[float, float, float, float, float, float]]:
        cfg = self.config
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(
            blur,
            int(cfg.get("automatic.line_detection.canny_low", 60)),
            int(cfg.get("automatic.line_detection.canny_high", 160)),
        )
        raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=int(cfg.get("automatic.line_detection.hough_threshold", 45)),
            minLineLength=int(cfg.get("automatic.line_detection.min_line_length", 40)),
            maxLineGap=int(cfg.get("automatic.line_detection.max_line_gap", 12)),
        )
        lines: list[tuple[float, float, float, float, float, float]] = []
        if raw is None:
            return lines
        for item in raw.reshape(-1, 4):
            x1, y1, x2, y2 = (float(v) for v in item)
            length = math.hypot(x2 - x1, y2 - y1)
            if length <= 0:
                continue
            angle = math.atan2(y2 - y1, x2 - x1) % math.pi
            lines.append((x1, y1, x2, y2, length, angle))
        return lines

    @staticmethod
    def _angular_distance(a: float, b: float) -> float:
        diff = abs(a - b) % math.pi
        return min(diff, math.pi - diff)

    def _dominant_angle(self, lines: Iterable[tuple[float, float, float, float, float, float]]) -> float:
        bins = np.zeros(180, dtype=float)
        for _x1, _y1, _x2, _y2, length, angle in lines:
            bins[int(math.degrees(angle)) % 180] += length
        peak = int(np.argmax(bins))
        window = []
        for offset in range(-3, 4):
            index = (peak + offset) % 180
            window.append((index, bins[index]))
        total = sum(weight for _index, weight in window)
        if total <= 0:
            return math.radians(peak)
        mean = sum(index * weight for index, weight in window) / total
        return math.radians(mean)

    def _merge(self, primary: list[ParkingSpace], secondary: list[ParkingSpace]) -> list[ParkingSpace]:
        merged = list(primary)
        for candidate in secondary:
            if all(polygon_iou(candidate.polygon, existing.polygon) < 0.5 for existing in merged):
                merged.append(candidate)
        return merged

    def _finalize_layout(self, spaces: list[ParkingSpace]) -> list[ParkingSpace]:
        ordered = sorted(spaces, key=lambda item: (round(item.centroid[1] / 40.0), item.centroid[0]))
        for index, space in enumerate(ordered, start=1):
            if not space.space_id:
                space.space_id = f"P{index:03d}"
        return ordered

    def _finalize(
        self,
        spaces: list[ParkingSpace],
        width: int,
        height: int,
        image: np.ndarray | None = None,
    ) -> list[ParkingSpace]:
        filtered: list[ParkingSpace] = []
        for space in spaces:
            if not self._is_valid(space, width, height, image=image):
                continue
            if any(polygon_iou(space.polygon, kept.polygon) >= self.duplicate_iou for kept in filtered):
                continue
            if self.expand_fraction > 0:
                center_x, center_y = polygon_centroid(space.polygon)
                space.polygon = [
                    (center_x + (x - center_x) * (1.0 + self.expand_fraction),
                     center_y + (y - center_y) * (1.0 + self.expand_fraction))
                    for x, y in space.polygon
                ]
            filtered.append(space)
            if len(filtered) >= self.max_detections:
                break

        # Validate that the detected spaces form an actual coherent parking layout
        # (Requirement 2 & 5: If the image has no identifiable or configured parking layout,
        # detect vehicles normally but report parking occupancy as UNAVAILABLE. Do not invent parking spaces.)
        if len(filtered) < self.min_layout_spaces:
            return []

        mean_conf = sum(s.confidence for s in filtered) / len(filtered)
        if mean_conf < self.min_layout_confidence:
            return []

        filtered.sort(key=lambda item: (round(item.centroid[1] / 40.0), item.centroid[0]))
        for index, space in enumerate(filtered, start=1):
            if not space.space_id:
                space.space_id = f"P{index:03d}"
        return filtered

    def _is_valid(
        self,
        space: ParkingSpace,
        width: int,
        height: int,
        image: np.ndarray | None = None,
    ) -> bool:
        from ..geometry import clip_polygon_to_rect

        # Validate polygon has enough points
        if len(space.polygon) < 3:
            self.last_rejected.append({
                "space_id": space.space_id,
                "method": space.method,
                "polygon": [[round(p[0], 2), round(p[1], 2)] for p in space.polygon],
                "reason": f"Polygon has < 3 points: {len(space.polygon)}",
            })
            return False

        # Clip to image boundary and re-evaluate
        clipped = clip_polygon_to_rect(space.polygon, (0.0, 0.0, float(width), float(height)))
        if len(clipped) < 3:
            self.last_rejected.append({
                "space_id": space.space_id,
                "method": space.method,
                "polygon": [[round(p[0], 2), round(p[1], 2)] for p in space.polygon],
                "reason": "Polygon outside image boundary",
            })
            return False

        orig_area = polygon_area(space.polygon)
        clipped_area = polygon_area(clipped)
        if orig_area <= 0 or clipped_area <= 0:
            return False
        if clipped_area / orig_area < 0.70:
            self.last_rejected.append({
                "space_id": space.space_id,
                "method": space.method,
                "polygon": [[round(p[0], 2), round(p[1], 2)] for p in space.polygon],
                "reason": "Polygon largely outside frame (>30% clipped)",
            })
            return False

        space.polygon = clipped
        area = clipped_area
        if area < self.min_space_area or area > self.max_space_fraction * width * height:
            self.last_rejected.append({
                "space_id": space.space_id,
                "method": space.method,
                "polygon": [[round(p[0], 2), round(p[1], 2)] for p in space.polygon],
                "reason": f"Area {area:.1f} out of bounds [{self.min_space_area}, {self.max_space_fraction * width * height:.1f}]",
            })
            return False

        x1, y1, x2, y2 = space.bbox
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        if min(box_width, box_height) < self.min_bbox_side:
            self.last_rejected.append({
                "space_id": space.space_id,
                "method": space.method,
                "polygon": [[round(p[0], 2), round(p[1], 2)] for p in space.polygon],
                "reason": f"Bbox side {min(box_width, box_height):.1f} < min {self.min_bbox_side}",
            })
            return False

        aspect = max(box_width, box_height) / min(box_width, box_height)
        if aspect < self.min_aspect or aspect > self.max_aspect:
            self.last_rejected.append({
                "space_id": space.space_id,
                "method": space.method,
                "polygon": [[round(p[0], 2), round(p[1], 2)] for p in space.polygon],
                "reason": f"Aspect ratio {aspect:.2f} out of bounds [{self.min_aspect}, {self.max_aspect}]",
            })
            return False

        vis = visible_fraction(space.polygon, width, height)
        if vis < 0.95:
            self.last_rejected.append({
                "space_id": space.space_id,
                "method": space.method,
                "polygon": [[round(p[0], 2), round(p[1], 2)] for p in space.polygon],
                "reason": f"Visibility fraction {vis:.2f} < 0.95",
            })
            return False

        if area < min(self.min_space_area, 400.0):
            return False

        # Ground plane & facade validation
        center_x, center_y = polygon_centroid(space.polygon)
        cy_ratio = center_y / max(height, 1)
        if cy_ratio < self.min_ground_y_ratio:
            self.last_rejected.append({
                "space_id": space.space_id,
                "method": space.method,
                "polygon": [[round(p[0], 2), round(p[1], 2)] for p in space.polygon],
                "reason": f"Upper facade / roofline rejection: centroid y-ratio {cy_ratio:.2f} < min {self.min_ground_y_ratio}",
            })
            return False

        # Visual surface validation against image
        if image is not None and image.size > 0:
            poly_np = np.asarray(space.polygon, dtype=np.int32)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [poly_np], 255)
            mask_pts = (mask == 255)
            if not np.any(mask_pts):
                return False

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            pts_gray = gray[mask_pts]
            pts_sat = hsv[:, :, 1][mask_pts]

            mean_gray = float(np.mean(pts_gray))
            std_gray = float(np.std(pts_gray))
            mean_sat = float(np.mean(pts_sat))

            if std_gray > self.max_internal_std:
                self.last_rejected.append({
                    "space_id": space.space_id,
                    "method": space.method,
                    "polygon": [[round(p[0], 2), round(p[1], 2)] for p in space.polygon],
                    "reason": f"Building facade/window mullions rejection: internal contrast std {std_gray:.1f} > max {self.max_internal_std}",
                })
                return False

            if mean_sat > self.max_saturation:
                self.last_rejected.append({
                    "space_id": space.space_id,
                    "method": space.method,
                    "polygon": [[round(p[0], 2), round(p[1], 2)] for p in space.polygon],
                    "reason": f"Non-pavement saturation rejection: saturation {mean_sat:.1f} > max {self.max_saturation}",
                })
                return False

            if mean_gray < self.min_brightness:
                self.last_rejected.append({
                    "space_id": space.space_id,
                    "method": space.method,
                    "polygon": [[round(p[0], 2), round(p[1], 2)] for p in space.polygon],
                    "reason": f"Dark entrance/shadow rejection: brightness {mean_gray:.1f} < min {self.min_brightness}",
                })
                return False

            if mean_gray > self.max_brightness:
                self.last_rejected.append({
                    "space_id": space.space_id,
                    "method": space.method,
                    "polygon": [[round(p[0], 2), round(p[1], 2)] for p in space.polygon],
                    "reason": f"Sky/reflection rejection: brightness {mean_gray:.1f} > max {self.max_brightness}",
                })
                return False

            edges = cv2.Canny(gray, 50, 150)
            pts_edge = edges[mask_pts]
            edge_dens = float(np.mean(pts_edge > 0))
            if edge_dens > self.max_edge_density:
                self.last_rejected.append({
                    "space_id": space.space_id,
                    "method": space.method,
                    "polygon": [[round(p[0], 2), round(p[1], 2)] for p in space.polygon],
                    "reason": f"High internal edge density rejection: edge density {edge_dens:.3f} > max {self.max_edge_density}",
                })
                return False

        return True

    def coverage(self, space: ParkingSpace, box: tuple[float, float, float, float]) -> float:
        area = polygon_area(space.polygon)
        if area <= 0:
            return 0.0
        return float(bbox_polygon_overlap(box, space.polygon) / area)
