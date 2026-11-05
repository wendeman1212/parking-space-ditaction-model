"""Unicode-safe image/video I/O, hashing and manifest helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import PROJECT_ROOT
from .types import InputKind, ValidationResult


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: str | Path, image: np.ndarray, quality: int = 95) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params: list[int] = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: str | Path, data: Any, indent: int = 2) -> Path:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=False, default=_json_default)
    return path


def load_json(path: str | Path) -> Any:
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def validate_image(path: str | Path, min_side: int = 64, max_side: int = 8192) -> ValidationResult:
    path = resolve_path(path)
    if not path.exists() or not path.is_file():
        return ValidationResult(False, InputKind.IMAGE, "Image file not found", error="missing_file")
    image = imread_unicode(path)
    if image is None or image.size == 0:
        return ValidationResult(False, InputKind.IMAGE, "File is not a readable image or is corrupted", error="decode_failed")
    height, width = image.shape[:2]
    if min(width, height) < min_side:
        return ValidationResult(
            False, InputKind.IMAGE, f"Image is too small ({width}x{height})", width, height, error="too_small"
        )
    if max(width, height) > max_side:
        return ValidationResult(
            False, InputKind.IMAGE, f"Image is too large ({width}x{height})", width, height, error="too_large"
        )
    return ValidationResult(True, InputKind.IMAGE, "Valid image", width, height)


def validate_video(path: str | Path) -> ValidationResult:
    path = resolve_path(path)
    if not path.exists() or not path.is_file():
        return ValidationResult(False, InputKind.VIDEO, "Video file not found", error="missing_file")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return ValidationResult(False, InputKind.VIDEO, "File is not a readable video or codec is unsupported", error="open_failed")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None or frame.size == 0:
        return ValidationResult(False, InputKind.VIDEO, "Video contains no decodable frames", width, height, frame_count, fps, error="no_frames")
    duration = frame_count / fps if fps > 0 else 0.0
    if width < 32 or height < 32:
        return ValidationResult(False, InputKind.VIDEO, "Video resolution is too small", width, height, frame_count, fps, duration, "too_small")
    return ValidationResult(
        True, InputKind.VIDEO, "Valid video", width, height, frame_count, fps, duration
    )
