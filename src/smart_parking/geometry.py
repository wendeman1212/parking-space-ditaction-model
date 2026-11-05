"""Polygon and bounding-box geometry used by detection, association and visualization."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import cv2
import numpy as np

Point = tuple[float, float]
Polygon = Sequence[Point]
Box = tuple[float, float, float, float]


def polygon_area(polygon: Polygon) -> float:
    if len(polygon) < 3:
        return 0.0
    pts = np.asarray(polygon, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_centroid(polygon: Polygon) -> Point:
    if not polygon:
        return (0.0, 0.0)
    pts = np.asarray(polygon, dtype=np.float64)
    if len(pts) < 3:
        return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    cross = pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]
    area = cross.sum() / 2.0
    if abs(area) < 1e-9:
        return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    cx = ((pts[:, 0] + np.roll(pts[:, 0], -1)) * cross).sum() / (6.0 * area)
    cy = ((pts[:, 1] + np.roll(pts[:, 1], -1)) * cross).sum() / (6.0 * area)
    return (float(cx), float(cy))


def _clip_edge(poly: list[Point], edge_fn, inside_fn) -> list[Point]:
    output: list[Point] = []
    if not poly:
        return output
    previous = poly[-1]
    for current in poly:
        current_inside = inside_fn(current)
        previous_inside = inside_fn(previous)
        if current_inside:
            if not previous_inside:
                output.append(edge_fn(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(edge_fn(previous, current))
        previous = current
    return output


def clip_polygon_to_rect(polygon: Polygon, rect: Box) -> list[Point]:
    """Sutherland-Hodgman clip of a polygon against an axis-aligned rectangle."""
    x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1:
        return []

    def intersect_vertical(p: Point, q: Point, x: float) -> Point:
        if abs(q[0] - p[0]) < 1e-12:
            return (x, p[1])
        t = (x - p[0]) / (q[0] - p[0])
        return (x, p[1] + t * (q[1] - p[1]))

    def intersect_horizontal(p: Point, q: Point, y: float) -> Point:
        if abs(q[1] - p[1]) < 1e-12:
            return (p[0], y)
        t = (y - p[1]) / (q[1] - p[1])
        return (p[0] + t * (q[0] - p[0]), y)

    poly = [(float(x), float(y)) for x, y in polygon]
    poly = _clip_edge(poly, lambda p, q: intersect_vertical(p, q, x1), lambda p: p[0] >= x1)
    poly = _clip_edge(poly, lambda p, q: intersect_vertical(p, q, x2), lambda p: p[0] <= x2)
    poly = _clip_edge(poly, lambda p, q: intersect_horizontal(p, q, y1), lambda p: p[1] >= y1)
    poly = _clip_edge(poly, lambda p, q: intersect_horizontal(p, q, y2), lambda p: p[1] <= y2)
    return poly


def bbox_polygon_overlap(box: Box, polygon: Polygon) -> float:
    clipped = clip_polygon_to_rect(polygon, box)
    return polygon_area(clipped)


def polygon_iou(poly_a: Polygon, poly_b: Polygon) -> float:
    inter = bbox_polygon_overlap(
        (
            min(p[0] for p in poly_b),
            min(p[1] for p in poly_b),
            max(p[0] for p in poly_b),
            max(p[1] for p in poly_b),
        ),
        poly_a,
    )
    union = polygon_area(poly_a) + polygon_area(poly_b) - inter
    return float(inter / union) if union > 0 else 0.0


def bbox_iou(a: Box, b: Box) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    if len(polygon) < 3:
        return False
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False) >= 0


def rotated_rect_polygon(center: Sequence[float], size: Sequence[float], angle_deg: float) -> list[Point]:
    cx, cy = float(center[0]), float(center[1])
    w, h = float(size[0]), float(size[1])
    rect = ((cx, cy), (w, h), float(angle_deg))
    points = cv2.boxPoints(rect)
    return [(float(x), float(y)) for x, y in points]


def bbox_polygon(box: Box) -> list[Point]:
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def order_points(points: Iterable[Point]) -> np.ndarray:
    pts = np.asarray(list(points), dtype=np.float32)
    if pts.shape[0] != 4:
        rect = cv2.minAreaRect(pts)
        pts = cv2.boxPoints(rect)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_polygon_crop(image: np.ndarray, polygon: Polygon, output_size: tuple[int, int]) -> np.ndarray:
    """Perspective-warp a four-point polygon to a canonical crop."""
    width, height = output_size
    rect = order_points(polygon)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)


def expand_polygon(polygon: Polygon, fraction: float, image_shape: tuple[int, ...] | None = None) -> list[Point]:
    pts = np.asarray(polygon, dtype=np.float32)
    center = pts.mean(axis=0)
    expanded = center + (pts - center) * (1.0 + fraction)
    if image_shape is not None:
        h, w = image_shape[:2]
        expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)
    return [(float(x), float(y)) for x, y in expanded]


def visible_fraction(polygon: Polygon, image_width: int, image_height: int) -> float:
    area = polygon_area(polygon)
    if area <= 0:
        return 0.0
    clipped = clip_polygon_to_rect(polygon, (0.0, 0.0, float(image_width), float(image_height)))
    return float(min(1.0, polygon_area(clipped) / area))


def side_of_line(point: Point, line_start: Point, line_end: Point) -> float:
    return (line_end[0] - line_start[0]) * (point[1] - line_start[1]) - (
        line_end[1] - line_start[1]
    ) * (point[0] - line_start[0])


def line_crossing_sign(a: Point, b: Point) -> float:
    """Alias used by entry/exit logic where a boundary is a directed line."""
    return (b[0] - a[0]) * 0 - 0


def segment_crosses_line(p1: Point, p2: Point, a: Point, b: Point) -> bool:
    s1 = side_of_line(p1, a, b)
    s2 = side_of_line(p2, a, b)
    return s1 * s2 < 0


def polygon_distance_to_point(polygon: Polygon, point: Point) -> float:
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return abs(float(cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), True)))


def angle_between_lines(line_a: Sequence[float], line_b: Sequence[float]) -> float:
    a = math.degrees(math.atan2(line_a[3] - line_a[1], line_a[2] - line_a[0]))
    b = math.degrees(math.atan2(line_b[3] - line_b[1], line_b[2] - line_b[0]))
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)
