import numpy as np

from smart_parking.config import load_config
from smart_parking.models.occupancy import OccupancyEngine, TemporalSmoother
from smart_parking.types import Detection, OccupancyState, ParkingSpace, SpaceSource


class _UnavailableClassifier:
    available = False
    input_size = 160
    classes = ["EMPTY", "OCCUPIED"]

    def predict(self, crops):
        return np.full((len(crops), 2), np.nan, dtype=np.float32)


def _space(polygon=None):
    polygon = polygon or [(20, 20), (120, 20), (120, 120), (20, 120)]
    return ParkingSpace(space_id="P001", polygon=polygon, source=SpaceSource.CALIBRATED)


def _engine():
    config = load_config("app", "occupancy_classifier")
    return OccupancyEngine(config, classifier=_UnavailableClassifier())


def test_full_vehicle_overlap_is_occupied():
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    detection = Detection(2, "car", 0.9, (20, 20, 120, 120))
    states = _engine().classify(image, [_space()], [detection])
    assert states[0].state == OccupancyState.OCCUPIED
    assert states[0].method == "vehicle_overlap"
    assert states[0].space_coverage > 0.9


def test_no_vehicle_is_empty():
    image = np.full((160, 160, 3), 90, dtype=np.uint8)
    states = _engine().classify(image, [_space()], [])
    assert states[0].state == OccupancyState.EMPTY


def test_partial_vehicle_touch_counts_as_occupied():
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    detection = Detection(2, "car", 0.9, (0, 30, 40, 110))
    states = _engine().classify(image, [_space()], [detection])
    assert states[0].state == OccupancyState.OCCUPIED
    assert states[0].method in {"vehicle_any_part", "vehicle_overlap"}


def test_low_confidence_detection_does_not_occupy():
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    detection = Detection(2, "car", 0.1, (20, 20, 120, 120))
    states = _engine().classify(image, [_space()], [detection])
    assert states[0].state == OccupancyState.EMPTY


def test_out_of_view_space_is_uncertain():
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    space = _space([(500, 500), (600, 500), (600, 600), (500, 600)])
    states = _engine().classify(image, [space], [])
    assert states[0].state == OccupancyState.UNCERTAIN
    assert states[0].method == "visibility"


def test_temporal_smoother_debounces_changes():
    config = load_config("app", "occupancy_classifier")
    smoother = TemporalSmoother(config)
    engine = _engine()
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    occupied = engine.classify(image, [_space()], [Detection(2, "car", 0.9, (20, 20, 120, 120))])
    empty = engine.classify(image, [_space()], [])
    first = smoother.update(occupied, timestamp=0.0)
    assert first[0].state == OccupancyState.OCCUPIED
    pending = smoother.update(empty, timestamp=0.1)
    assert pending[0].state == OccupancyState.OCCUPIED
    assert "temporal_pending" in pending[0].reasons
    confirmed = smoother.update(empty, timestamp=5.0)
    assert confirmed[0].state in {OccupancyState.EMPTY, OccupancyState.OCCUPIED}
