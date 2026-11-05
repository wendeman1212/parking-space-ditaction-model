import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smart_parking.config import load_app_config  # noqa: E402
from smart_parking.data.preprocess import build_all  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build patches, reference layouts, scene manifest and YOLO datasets")
    args = parser.parse_args()
    config = load_app_config()
    result = build_all(config)
    print(json.dumps(result, indent=2)[:8000])
