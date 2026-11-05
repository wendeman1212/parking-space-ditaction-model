"""System evaluation: occupancy accuracy, false-empty rate, space and vehicle detection."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import Config
from ..geometry import bbox_polygon
from ..io_utils import imread_unicode, load_json, resolve_path, save_json
from ..pipeline import ParkingPipeline
from ..types import OccupancyState
from .metrics import binary_metrics, match_boxes, match_polygons


def _frames_for_split(config: Config, split: str) -> list[dict[str, Any]]:
    frames = load_json(config.path("paths.manifests") / "pklot_frames.json")
    return [frame for frame in frames if frame.get("split") == split]


def _layout_path(config: Config, frame: dict[str, Any]) -> Path:
    return config.path("paths.processed") / "layouts" / f"{frame['subset']}_{frame['image_id']}.json"


def evaluate_occupancy(
    pipeline: ParkingPipeline,
    config: Config | None = None,
    split: str = "test",
    limit: int | None = None,
    save: bool = True,
) -> dict[str, Any]:
    config = config or pipeline.config
    frames = _frames_for_split(config, split)
    predictions: list[bool] = []
    targets: list[bool] = []
    per_frame: list[dict[str, Any]] = []
    missing_layouts = 0
    skipped_spaces = 0
    uncertain = 0
    uncertain_on_occupied = 0
    evaluated_frames = 0
    for frame in frames:
        layout_path = _layout_path(config, frame)
        if not layout_path.exists():
            missing_layouts += 1
            continue
        if limit is not None and evaluated_frames >= limit:
            break
        layout = load_json(layout_path)
        result = pipeline.process_image(frame["path"], layout=layout, save=False)
        ground_truth: dict[str, bool] = {}
        for space in layout.get("spaces", []):
            label = str(space.get("metadata", {}).get("occupancy_label", "")).upper()
            if label == "OCCUPIED":
                ground_truth[str(space["space_id"])] = True
            elif label == "EMPTY":
                ground_truth[str(space["space_id"])] = False
        frame_predictions: list[bool] = []
        frame_targets: list[bool] = []
        frame_uncertain = 0
        for item in result.get("spaces", []):
            space_id = str(item["space_id"])
            if space_id not in ground_truth:
                skipped_spaces += 1
                continue
            status = item.get("occupancy") or {}
            state = status.get("state")
            target = ground_truth[space_id]
            if state not in (OccupancyState.OCCUPIED.value, OccupancyState.EMPTY.value):
                uncertain += 1
                frame_uncertain += 1
                if target:
                    uncertain_on_occupied += 1
                continue
            prediction = state == OccupancyState.OCCUPIED.value
            frame_predictions.append(prediction)
            frame_targets.append(target)
            predictions.append(prediction)
            targets.append(target)
        per_frame.append(
            {
                "image_id": int(frame["image_id"]),
                "subset": frame["subset"],
                "spaces": len(frame_predictions) + frame_uncertain,
                "decided": len(frame_predictions),
                "uncertain": frame_uncertain,
                "occupied_ground_truth": sum(1 for value in frame_targets if value),
            }
        )
        evaluated_frames += 1
    metrics = binary_metrics(predictions, targets) if predictions else binary_metrics([], [])
    metrics["uncertain_spaces"] = uncertain
    metrics["uncertain_on_occupied"] = uncertain_on_occupied
    metrics["evaluated_frames"] = evaluated_frames
    metrics["missing_layouts"] = missing_layouts
    metrics["skipped_spaces"] = skipped_spaces
    report = {
        "split": split,
        "metrics": metrics,
        "per_frame": per_frame,
    }
    if save:
        save_json(config.path("paths.outputs") / f"reports/occupancy_eval_{split}.json", report)
    return report


def _car_annotations(config: Config, split: str) -> dict[int, list[tuple[float, float, float, float]]]:
    frames = _frames_for_split(config, split)
    subsets = sorted({frame["subset"] for frame in frames})
    lookup: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for subset in subsets:
        path = config.path("paths.processed") / "pklot" / f"selected_{subset}_cars.json"
        if not path.exists():
            continue
        data = load_json(path)
        for annotation in data.get("annotations", []):
            bbox = annotation.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x, y, width, height = (float(value) for value in bbox)
            if width * height < 64:
                continue
            lookup[int(annotation["image_id"])].append((x, y, x + width, y + height))
    return lookup


def evaluate_vehicle_detection(
    pipeline: ParkingPipeline,
    config: Config | None = None,
    split: str = "test",
    limit: int | None = None,
    iou_threshold: float = 0.5,
    save: bool = True,
) -> dict[str, Any]:
    config = config or pipeline.config
    frames = _frames_for_split(config, split)
    annotations = _car_annotations(config, split)
    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_frame: list[dict[str, Any]] = []
    evaluated = 0
    for frame in frames:
        if limit is not None and evaluated >= limit:
            break
        image = imread_unicode(resolve_path(frame["path"]))
        if image is None:
            continue
        detections = pipeline.vehicle_detector.detect(image)
        predictions = [detection.xyxy for detection in detections]
        targets = annotations.get(int(frame["image_id"]), [])
        matched = match_boxes(predictions, targets, iou_threshold=iou_threshold)
        totals["tp"] += matched["tp"]
        totals["fp"] += matched["fp"]
        totals["fn"] += matched["fn"]
        per_frame.append(
            {
                "image_id": int(frame["image_id"]),
                "predictions": len(predictions),
                "targets": len(targets),
                "tp": matched["tp"],
                "fp": matched["fp"],
                "fn": matched["fn"],
            }
        )
        evaluated += 1
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "split": split,
        "iou_threshold": iou_threshold,
        "evaluated_frames": evaluated,
        "metrics": {
            **totals,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "per_frame": per_frame,
    }
    if save:
        save_json(config.path("paths.outputs") / f"reports/vehicle_eval_{split}.json", report)
    return report


def evaluate_space_detection(
    pipeline: ParkingPipeline,
    config: Config | None = None,
    split: str = "test",
    limit: int | None = None,
    iou_threshold: float = 0.5,
    save: bool = True,
) -> dict[str, Any]:
    config = config or pipeline.config
    frames = _frames_for_split(config, split)
    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_frame: list[dict[str, Any]] = []
    evaluated = 0
    for frame in frames:
        layout_path = _layout_path(config, frame)
        if not layout_path.exists():
            continue
        if limit is not None and evaluated >= limit:
            break
        image = imread_unicode(resolve_path(frame["path"]))
        if image is None:
            continue
        layout = load_json(layout_path)
        targets = [
            [(float(point[0]), float(point[1])) for point in space.get("polygon", [])]
            for space in layout.get("spaces", [])
            if len(space.get("polygon", [])) >= 3
        ]
        spaces = pipeline.space_detector.detect(image)
        predictions = [space.polygon for space in spaces]
        matched = match_polygons(predictions, targets, iou_threshold=iou_threshold)
        totals["tp"] += matched["tp"]
        totals["fp"] += matched["fp"]
        totals["fn"] += matched["fn"]
        per_frame.append(
            {
                "image_id": int(frame["image_id"]),
                "predictions": len(predictions),
                "targets": len(targets),
                "tp": matched["tp"],
                "fp": matched["fp"],
                "fn": matched["fn"],
            }
        )
        evaluated += 1
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "split": split,
        "iou_threshold": iou_threshold,
        "evaluated_frames": evaluated,
        "metrics": {
            **totals,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "per_frame": per_frame,
    }
    if save:
        save_json(config.path("paths.outputs") / f"reports/space_eval_{split}.json", report)
    return report
