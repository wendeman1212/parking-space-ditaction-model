from smart_parking.geometry import (
    bbox_iou,
    bbox_polygon_overlap,
    clip_polygon_to_rect,
    polygon_area,
    polygon_iou,
    visible_fraction,
)


def test_polygon_area_and_iou():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    shifted = [(5, 0), (15, 0), (15, 10), (5, 10)]
    assert polygon_area(square) == 100
    assert abs(polygon_iou(square, shifted) - 50 / 150) < 1e-9
    assert polygon_iou(square, square) == 1.0


def test_bbox_polygon_overlap():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert bbox_polygon_overlap((0, 0, 5, 10), square) == 50
    assert bbox_polygon_overlap((20, 20, 30, 30), square) == 0


def test_clip_and_visibility():
    square = [(-10, 0), (5, 0), (5, 10), (-10, 10)]
    clipped = clip_polygon_to_rect(square, (0, 0, 100, 100))
    assert polygon_area(clipped) == 50
    assert abs(visible_fraction(square, 100, 100) - 1 / 3) < 1e-9


def test_bbox_iou():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
