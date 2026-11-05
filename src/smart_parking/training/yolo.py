"""YOLO training for vehicle detection and parking-space OBB detection."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT, Config
from ..io_utils import save_json

_VEHICLE_DATA = "data/processed/yolo_detect/data.yaml"
_OBB_DATA = "data/processed/yolo_obb/data.yaml"


def _pretrained_path(name: str) -> str:
    candidate = PROJECT_ROOT / f"{name}.pt"
    return str(candidate) if candidate.exists() else f"{name}.pt"


def _train_yolo(
    config: Config,
    weights: str,
    data_yaml: str | Path,
    output_name: str,
    target_path: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    patience: int,
) -> dict[str, Any]:
    from ultralytics import YOLO

    data_path = Path(data_yaml)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}. Run scripts/prepare_dataset.py first.")
    try:
        model = YOLO(weights)
    except Exception:
        fallback = f"{weights[:-3]}.yaml" if weights.endswith(".pt") else None
        if not fallback or Path(weights).exists():
            raise
        model = YOLO(fallback)
        weights = fallback
    results = model.train(
        data=str(data_path),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        device=str(device),
        project=str(config.path("paths.outputs") / "training"),
        name=output_name,
        exist_ok=True,
        patience=int(patience),
        seed=int(config.get("training.seed", 2026)),
        workers=0,
        verbose=True,
    )
    save_dir = Path(getattr(results, "save_dir", config.path("paths.outputs") / "training" / output_name))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training finished without a best checkpoint: {best}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target_path)
    report = {
        "task": "obb" if output_name == "parking_spaces" else "detect",
        "weights": str(target_path),
        "data": str(data_path),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "device": str(device),
        "run_directory": str(save_dir),
        "best_checkpoint": str(best),
    }
    save_json(config.path("paths.outputs") / f"reports/training_{output_name}.json", report)
    return report


def train_vehicle_detector(
    config: Config,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    training = config.get("training", {}) or {}
    weights = _pretrained_path(str(config.get("model.architecture", "yolo11n")))
    return _train_yolo(
        config=config,
        weights=weights,
        data_yaml=_VEHICLE_DATA,
        output_name="vehicle",
        target_path=config.path("model.path"),
        epochs=int(epochs if epochs is not None else training.get("epochs", 20)),
        imgsz=int(imgsz if imgsz is not None else training.get("imgsz", 640)),
        batch=int(batch if batch is not None else training.get("batch_size", 16)),
        device=str(device or config.get("model.device", "cpu")),
        patience=int(training.get("patience", 8)),
    )


def train_parking_space_detector(
    config: Config,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    training = config.get("training", {}) or {}
    architecture = str(config.get("obb_model.architecture", "yolo11n-obb"))
    try:
        weights = _pretrained_path(architecture)
    except Exception:
        weights = f"{architecture}.yaml"
    return _train_yolo(
        config=config,
        weights=weights,
        data_yaml=_OBB_DATA,
        output_name="parking_spaces",
        target_path=config.path("obb_model.path"),
        epochs=int(epochs if epochs is not None else training.get("epochs", 30)),
        imgsz=int(imgsz if imgsz is not None else training.get("imgsz", 640)),
        batch=int(batch if batch is not None else training.get("batch_size", 8)),
        device=str(device or config.get("obb_model.device", "cpu")),
        patience=int(training.get("patience", 10)),
    )
