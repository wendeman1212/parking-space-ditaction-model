import cv2
import numpy as np
from pathlib import Path

from smart_parking.config import load_app_config, load_component_config
from smart_parking.models.spaces import ParkingSpaceDetector
from smart_parking.pipeline import ParkingPipeline
from smart_parking.types import ParkingSpace, SpaceSource


def test_facade_rejection_rules():
    config = load_component_config("parking_space_detector")
    detector = ParkingSpaceDetector(config)

    # Synthetic image 1000x1000
    h, w = 1000, 1000
    img = np.full((h, w, 3), 120, dtype=np.uint8)

    # 1. Upper facade polygon (cy_ratio < 0.32)
    upper_space = ParkingSpace(
        space_id="upper",
        polygon=[(100.0, 50.0), (200.0, 50.0), (200.0, 150.0), (100.0, 150.0)],
        source=SpaceSource.AUTOMATIC,
    )
    assert not detector._is_valid(upper_space, w, h, image=img)

    # 2. Ground polygon with high internal variance (window mullions / texture)
    window_space = ParkingSpace(
        space_id="window",
        polygon=[(100.0, 500.0), (200.0, 500.0), (200.0, 700.0), (100.0, 700.0)],
        source=SpaceSource.AUTOMATIC,
    )
    # Put high contrast pattern inside window_space
    img[500:700, 100:200] = np.random.choice([0, 255], size=(200, 100, 3)).astype(np.uint8)
    assert not detector._is_valid(window_space, w, h, image=img)

    # 3. Ground polygon with normal asphalt pavement texture
    asphalt_space = ParkingSpace(
        space_id="asphalt",
        polygon=[(400.0, 500.0), (500.0, 500.0), (500.0, 700.0), (400.0, 700.0)],
        source=SpaceSource.AUTOMATIC,
    )
    img[500:700, 400:500] = 110  # smooth gray pavement
    assert detector._is_valid(asphalt_space, w, h, image=img)


def test_commercial_building_street_image():
    pipeline = ParkingPipeline()
    img_path = Path("data/raw/uploads/20261101T100409_35fe91e2_image5.jpg")
    if not img_path.exists():
        return

    result = pipeline.process_image(img_path, save=False)
    # Zero parking spaces should be detected on building facade or street
    assert result["statistics"]["total_spaces"] == 0
    assert len(result["spaces"]) == 0
    # Occupancy must be UNAVAILABLE
    assert result["occupancy_reportable"] is False
    # Vehicles should be detected independently
    assert result["statistics"]["detected_vehicles"] > 0
    # Rejected regions are documented
    assert len(result["rejected_regions"]) > 0


def test_genuine_parking_lot_image():
    pipeline = ParkingPipeline()
    img_path = Path("data/raw/uploads/20261031T122131_27da9684_2012-09-12_06_05_16.jpg")
    if not img_path.exists():
        return

    result = pipeline.process_image(img_path, save=False)
    # Should detect genuine spaces and report occupancy
    assert result["statistics"]["total_spaces"] > 0
    assert result["occupancy_reportable"] is True
