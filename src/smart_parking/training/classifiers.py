"""MobileNetV3 fine-tuning for occupancy and parking-scene classification."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ..config import Config
from ..io_utils import load_json, resolve_path, save_json

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _read_rgb(path: str | Path, size: int, augment: bool) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    if augment:
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
        alpha = random.uniform(0.85, 1.15)
        beta = random.uniform(-18.0, 18.0)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (rgb - _IMAGENET_MEAN) / _IMAGENET_STD


class _ImageDataset:
    def __init__(self, records: Sequence[tuple[str, int]], size: int, augment: bool):
        self.records = list(records)
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Any:
        import torch

        path, label = self.records[index]
        array = _read_rgb(path, self.size, self.augment)
        if array is None:
            array = np.zeros((self.size, self.size, 3), dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor, int(label)


def _records_from_manifest(
    manifest_path: Path,
    split: str,
    path_key: str,
    class_names: Sequence[str],
    label_key: str | None = None,
) -> list[tuple[str, int]]:
    data = load_json(manifest_path)
    index = {name.upper(): idx for idx, name in enumerate(class_names)}
    records: list[tuple[str, int]] = []
    for item in data:
        if item.get("split") != split:
            continue
        path = resolve_path(item[path_key])
        if not path.exists():
            continue
        label: int | None = None
        if label_key is not None and label_key in item:
            raw_value = item[label_key]
            if isinstance(raw_value, int) or str(raw_value).lstrip("-").isdigit():
                label = int(raw_value)
            else:
                label = index.get(str(raw_value).upper())
        else:
            raw = str(item.get("class_name", item.get("label", ""))).upper()
            label = index.get(raw)
        if label is None or label >= len(class_names):
            continue
        records.append((str(path), label))
    return records


def _make_loader(records: Sequence[tuple[str, int]], size: int, augment: bool, batch_size: int, balance: bool) -> Any:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    dataset = _ImageDataset(records, size, augment)
    sampler = None
    if balance and records:
        counts = Counter(label for _path, label in records)
        weights = [1.0 / counts[label] for _path, label in records]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None and augment,
        sampler=sampler,
        num_workers=0,
    ), dataset


def _run_epoch(model: Any, loader: Any, criterion: Any, optimizer: Any, device: Any, train: bool) -> tuple[float, int, int, list[int], list[int]]:
    import torch

    total_loss = 0.0
    total = 0
    correct = 0
    predictions: list[int] = []
    targets: list[int] = []
    model.train(train)
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total += len(labels)
            predicted = logits.argmax(dim=1)
            correct += int((predicted == labels).sum().item())
            predictions.extend(predicted.detach().cpu().tolist())
            targets.extend(labels.detach().cpu().tolist())
    return (total_loss / max(1, total), correct, total, predictions, targets)


def _classification_report(predictions: Sequence[int], targets: Sequence[int], class_names: Sequence[str]) -> dict[str, Any]:
    per_class = {}
    for index, name in enumerate(class_names):
        tp = sum(1 for p, t in zip(predictions, targets) if p == index and t == index)
        fp = sum(1 for p, t in zip(predictions, targets) if p == index and t != index)
        fn = sum(1 for p, t in zip(predictions, targets) if p != index and t == index)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    accuracy = sum(1 for p, t in zip(predictions, targets) if p == t) / max(1, len(targets))
    return {"accuracy": round(accuracy, 4), "per_class": per_class}


def _train_classifier(
    config: Config,
    manifest_path: Path,
    path_key: str,
    class_names: Sequence[str],
    model_path: Path,
    report_path: Path,
    label_key: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
) -> dict[str, Any]:
    import torch
    from torchvision.models import mobilenet_v3_small

    training = config.get("training", {}) or {}
    seed = int(training.get("seed", 2026))
    _set_seed(seed)
    input_size = int(config.get("model.input_size", 160))
    epochs = int(epochs if epochs is not None else training.get("epochs", 12))
    batch_size = int(batch_size if batch_size is not None else training.get("batch_size", 64))
    learning_rate = float(
        learning_rate if learning_rate is not None else training.get("learning_rate", 3e-4)
    )
    weight_decay = float(training.get("weight_decay", 1e-4))
    patience = int(training.get("patience", 4))
    balance = bool(training.get("class_balance", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_records = _records_from_manifest(manifest_path, "train", path_key, class_names, label_key)
    val_records = _records_from_manifest(manifest_path, "val", path_key, class_names, label_key)
    if not train_records:
        raise ValueError(f"No training records found in {manifest_path}")
    if not val_records:
        val_records = train_records[: max(1, len(train_records) // 5)]

    train_loader, _ = _make_loader(train_records, input_size, True, batch_size, balance)
    val_loader, _ = _make_loader(val_records, input_size, False, batch_size, False)

    model = mobilenet_v3_small(weights="DEFAULT" if config.get("training.pretrained", True) else None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, len(class_names))
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_accuracy = -1.0
    best_state = None
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_loss, _correct, _total, _pred, _target = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, _correct, _total, pred, target = _run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        report = _classification_report(pred, target, class_names)
        accuracy = float(report["accuracy"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_accuracy": accuracy,
            }
        )
        print(f"[{model_path.stem}] epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"[{model_path.stem}] early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_loss, _correct, _total, pred, target = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
    report = _classification_report(pred, target, class_names)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": list(class_names),
            "input_size": input_size,
            "architecture": "mobilenet_v3_small",
            "val_accuracy": report["accuracy"],
        },
        model_path,
    )
    result = {
        "model": str(model_path),
        "architecture": "mobilenet_v3_small",
        "classes": list(class_names),
        "input_size": input_size,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "epochs": len(history),
        "best_val_accuracy": round(best_accuracy, 4),
        "val_report": report,
        "history": history,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(report_path, result)
    return result


def train_occupancy_classifier(config: Config, epochs: int | None = None, batch_size: int | None = None) -> dict[str, Any]:
    manifest = config.path("paths.manifests") / "occupancy_patches.json"
    return _train_classifier(
        config=config,
        manifest_path=manifest,
        path_key="patch",
        class_names=["EMPTY", "OCCUPIED"],
        model_path=config.path("model.path"),
        report_path=config.path("paths.outputs") / "reports/training_occupancy.json",
        label_key="class_name",
        epochs=epochs,
        batch_size=batch_size,
    )


def train_scene_classifier(config: Config, epochs: int | None = None, batch_size: int | None = None) -> dict[str, Any]:
    manifest = config.path("paths.manifests") / "scene_frames.json"
    return _train_classifier(
        config=config,
        manifest_path=manifest,
        path_key="path",
        class_names=["NON_PARKING", "PARKING"],
        model_path=config.path("model.path"),
        report_path=config.path("paths.outputs") / "reports/training_scene.json",
        label_key="label",
        epochs=epochs,
        batch_size=batch_size,
    )
