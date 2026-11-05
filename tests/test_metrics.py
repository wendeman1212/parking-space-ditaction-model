from smart_parking.evaluation.metrics import binary_metrics, match_boxes, match_polygons


def test_binary_metrics_reports_false_empty_rate():
    result = binary_metrics([True, False, True, False], [True, True, False, False])
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["tn"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["false_empty_rate"] == 0.5
    assert result["false_occupied_rate"] == 0.5


def test_match_boxes_is_greedy_and_one_to_one():
    targets = [(0, 0, 10, 10), (30, 30, 40, 40)]
    predictions = [(1, 0, 11, 10), (31, 31, 41, 41), (100, 100, 110, 110)]
    result = match_boxes(predictions, targets, iou_threshold=0.5)
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 0


def test_match_polygons_counts_misses():
    targets = [[(0, 0), (10, 0), (10, 10), (0, 10)]]
    predictions = [[(50, 50), (60, 50), (60, 60), (50, 60)]]
    result = match_polygons(predictions, targets)
    assert result["tp"] == 0
    assert result["fp"] == 1
    assert result["fn"] == 1
