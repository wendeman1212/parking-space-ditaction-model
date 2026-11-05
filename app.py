"""Flask web UI for the smart-parking pipeline."""

from __future__ import annotations

import json
import logging
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smart_parking.config import load_app_config  # noqa: E402
from smart_parking.pipeline import ParkingPipeline  # noqa: E402

APP_CONFIG = load_app_config()

OUTPUT_DIR = APP_CONFIG.path("paths.outputs")
UPLOAD_DIR = APP_CONFIG.path("paths.raw") / "uploads"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    str(ext).lower()
    for ext in (APP_CONFIG.get("validation.allowed_image_extensions", []) or [])
}
VIDEO_EXTENSIONS = {
    str(ext).lower()
    for ext in (APP_CONFIG.get("validation.allowed_video_extensions", []) or [])
}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "app" / "templates"),
    static_folder=str(PROJECT_ROOT / "app" / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = int(APP_CONFIG.get("server.max_upload_mb", 250)) * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_pipeline: ParkingPipeline | None = None
_pipeline_lock = threading.Lock()
_process_lock = threading.Lock()


def get_pipeline() -> ParkingPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = ParkingPipeline()
    return _pipeline


def public_output_url(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    try:
        relative = path.resolve().relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        return None
    return url_for("outputs", filename=relative.as_posix())


def parse_layout() -> dict[str, Any] | None:
    upload = request.files.get("layout")
    raw: str | None = None
    if upload is not None and upload.filename:
        raw = upload.read().decode("utf-8-sig", errors="replace")
    else:
        raw = request.form.get("layout")
    if not raw or not raw.strip():
        return None
    try:
        layout = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Layout JSON is invalid: {exc.msg} (line {exc.lineno})") from exc
    if not isinstance(layout, dict) or not isinstance(layout.get("spaces"), list):
        raise ValueError('Layout must be a JSON object with a "spaces" array')
    return layout


def parse_int_field(name: str) -> int | None:
    raw = request.form.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return value if value > 0 else None


def build_response(run_id: str, result: dict[str, Any]) -> dict[str, Any]:
    urls = {
        "annotated": public_output_url(result.get("annotated_image") or result.get("annotated_video")),
        "result_json": public_output_url(result.get("result_json")),
    }
    payload = {
        "run_id": run_id,
        "kind": result.get("kind"),
        "width": result.get("width"),
        "height": result.get("height"),
        "fps": result.get("fps"),
        "frames_processed": result.get("frames_processed"),
        "scene": result.get("scene"),
        "statistics": result.get("statistics"),
        "occupancy_reportable": result.get("occupancy_reportable"),
        "timings_ms": result.get("timings_ms"),
        "spaces": result.get("spaces", []),
        "detections": result.get("detections", []),
        "events": result.get("events", []),
        "urls": urls,
    }
    return payload


@app.route("/")
def index():
    return render_template(
        "index.html",
        max_upload_mb=APP_CONFIG.get("server.max_upload_mb", 250),
        image_extensions=sorted(IMAGE_EXTENSIONS),
        video_extensions=sorted(VIDEO_EXTENSIONS),
    )


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "name": APP_CONFIG.get("project.name", "Smart Parking AI"),
            "version": APP_CONFIG.get("project.version", "0.1.0"),
            "device": APP_CONFIG.get("processing.device", "cpu"),
            "pipeline_loaded": _pipeline is not None,
        }
    )


@app.post("/api/process")
def process_upload():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"ok": False, "error": "No file uploaded"}), 400

    original_name = secure_filename(upload.filename) or "upload"
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({"ok": False, "error": f"Unsupported file type '{extension or 'unknown'}'. Allowed: {allowed}"}), 400

    try:
        layout = parse_layout()
        max_frames = parse_int_field("max_frames")
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    upload_path = UPLOAD_DIR / f"{run_id}_{original_name}"
    upload.save(upload_path)

    try:
        with _process_lock:
            pipeline = get_pipeline()
            if extension in VIDEO_EXTENSIONS:
                result = pipeline.process_video(
                    upload_path,
                    layout=layout,
                    save=True,
                    output_name=run_id,
                    max_frames=max_frames,
                )
            else:
                result = pipeline.process_image(
                    upload_path,
                    layout=layout,
                    save=True,
                    output_name=run_id,
                )
    except ValueError as exc:
        app.logger.warning("Processing rejected for %s: %s", upload_path, exc)
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 - surface unexpected pipeline failures to the UI
        app.logger.exception("Processing failed for %s", upload_path)
        return jsonify({"ok": False, "error": f"Processing failed: {exc}"}), 500

    return jsonify({"ok": True, "result": build_response(run_id, result)})


@app.get("/outputs/<path:filename>")
def outputs(filename: str):
    target = (OUTPUT_DIR / filename).resolve()
    try:
        target.relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        abort(404)
    if not target.is_file():
        abort(404)
    return send_from_directory(OUTPUT_DIR, filename)


@app.errorhandler(413)
def too_large(_error):
    limit = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return jsonify({"ok": False, "error": f"Upload exceeds the {limit} MB limit"}), 413


if __name__ == "__main__":
    host = str(APP_CONFIG.get("server.host", "127.0.0.1"))
    port = int(APP_CONFIG.get("server.port", 5000))
    debug = bool(APP_CONFIG.get("server.debug", False))
    print(f"Smart Parking AI web UI -> http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)
