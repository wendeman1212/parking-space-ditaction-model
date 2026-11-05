"""Precision/recall helpers for occupancy decisions and box/polygon matching."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from ..geometry import bbox_iou, polygon_iou


def binary_metrics(predictions: Sequence[bool], targets: Sequence[bool]) -> dict[str, Any]:
    """Metrics where ``True`` means OCCUPIED."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    tp = sum(1 for p, t in zip(predictions, targets) if p and t)
    fp = sum(1 for p, t in zip(predictions, targets) if p and not t)
    fn = sum(1 for p, t in zip(predictions, targets) if not p and t)
    tn = sum(1 for p, t in zip(predictions, targets) if not p and not t)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + fn + tn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_empty_rate": round(fn / (tp + fn), 4) if tp + fn else 0.0,
        "false_occupied_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "specificity": round(tn / (tn + fp), 4) if tn + fp else 0.0,
    }


def _greedy_match(
    predictions: Sequence[Any],
    targets: Sequence[Any],
    similarity: Callable[[Any, Any], float],
    threshold: float,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    pairs: list[tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(predictions):
        for target_index, target in enumerate(targets):
            score = float(similarity(prediction, target))
            if score >= threshold:
                pairs.append((score, pred_index, target_index))
    pairs.sort(key=lambda item: item[0], reverse=True)
    used_pred: set[int] = set()
    used_target: set[int] = set()
    matches: list[dict[str, Any]] = []
    for score, pred_index, target_index in pairs:
        if pred_index in used_pred or target_index in used_target:
            continue
        used_pred.add(pred_index)
        used_target.add(target_index)
        matches.append(
            {
                "prediction_index": pred_index,
                "target_index": target_index,
                "iou": round(score, 4),
            }
        )
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(targets) - tp
    return tp, fp, fn, matches


def match_polygons(
    predictions: Sequence[Sequence[tuple[float, float]]],
    targets: Sequence[Sequence[tuple[float, float]]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    tp, fp, fn, matches = _greedy_match(predictions, targets, polygon_iou, iou_threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matches": matches,
    }


def match_boxes(
    predictions: Sequence[tuple[float, float, float, float]],
    targets: Sequence[tuple[float, float, float, float]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    tp, fp, fn, matches = _greedy_match(predictions, targets, bbox_iou, iou_threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matches": matches,
    }
