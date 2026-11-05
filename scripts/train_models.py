import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smart_parking.config import load_config  # noqa: E402
from smart_parking.training.classifiers import (  # noqa: E402
    train_occupancy_classifier,
    train_scene_classifier,
)
from smart_parking.training.yolo import (  # noqa: E402
    train_parking_space_detector,
    train_vehicle_detector,
)


def main() -> dict:
    parser = argparse.ArgumentParser(description="Train the vehicle, parking-space, occupancy and scene models")
    parser.add_argument(
        "--task",
        choices=["all", "vehicle", "spaces", "occupancy", "scene"],
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override configured epoch count")
    parser.add_argument("--batch", type=int, default=None, help="Override configured batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override YOLO image size")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, mps")
    args = parser.parse_args()

    tasks = ["occupancy", "spaces", "vehicle", "scene"] if args.task == "all" else [args.task]
    results = {}
    for task in tasks:
        print(f"===== training {task} =====")
        if task == "occupancy":
            config = load_config("app", "datasets", "occupancy_classifier")
            results[task] = train_occupancy_classifier(config, epochs=args.epochs, batch_size=args.batch)
        elif task == "scene":
            config = load_config("app", "datasets", "scene_classifier")
            results[task] = train_scene_classifier(config, epochs=args.epochs, batch_size=args.batch)
        elif task == "vehicle":
            config = load_config("app", "datasets", "tracker", "vehicle_detector")
            results[task] = train_vehicle_detector(
                config, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device
            )
        elif task == "spaces":
            config = load_config("app", "datasets", "parking_space_detector")
            results[task] = train_parking_space_detector(
                config, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device
            )
    print(json.dumps(results, indent=2, default=str)[:12000])
    return results


if __name__ == "__main__":
    main()
