import json
import os
import platform
import shutil
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smart_parking.io_utils import save_json  # noqa: E402


def package_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return getattr(module, "__version__", "installed")
    except Exception:
        return None


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def main() -> dict:
    import cv2
    import psutil
    import torch
    import ultralytics

    root = PROJECT_ROOT
    disk = shutil.disk_usage(root)
    memory = psutil.virtual_memory()
    report = {
        "os": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "hardware": {
            "cpu": platform.processor(),
            "logical_cores": os.cpu_count(),
            "ram_total_gb": round(memory.total / 1024 ** 3, 2),
            "ram_available_gb": round(memory.available / 1024 ** 3, 2),
            "disk_total_gb": round(disk.total / 1024 ** 3, 2),
            "disk_free_gb": round(disk.free / 1024 ** 3, 2),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_devices": torch.cuda.device_count(),
            "torch_device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "packages": {
            name: package_version(name)
            for name in ["numpy", "opencv-python", "torch", "torchvision", "ultralytics", "flask", "pytest", "yaml", "psutil"]
        },
        "versions": {
            "opencv": cv2.__version__,
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "server_port_5000_in_use": port_in_use(5000),
        "project": {
            "root": str(root),
            "existing_top_level": sorted(p.name for p in root.iterdir() if p.name not in {"__pycache__"}) if root.exists() else [],
            "pklot_frames": len(list((root / "data/raw/pklot").rglob("*.jpg"))) if (root / "data/raw/pklot").exists() else 0,
            "coco_images": len(list((root / "data/raw/coco/val2017").glob("*.jpg"))) if (root / "data/raw/coco/val2017").exists() else 0,
            "model_checkpoints": [p.name for p in (root / "models/checkpoints").glob("*") if p.is_file()],
        },
    }
    save_json(root / "outputs/reports/environment_audit.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
