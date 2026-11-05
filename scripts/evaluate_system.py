import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smart_parking.evaluation.evaluate import (  # noqa: E402
    evaluate_occupancy,
    evaluate_space_detection,
    evaluate_vehicle_detection,
)
from smart_parking.pipeline import ParkingPipeline  # noqa: E402


def main() -> dict:
    parser = argparse.ArgumentParser(description="Evaluate occupancy, space detection and vehicle detection")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Limit evaluated frames for occupancy")
    parser.add_argument("--space-limit", type=int, default=30, help="Frames used for space-detection evaluation")
    parser.add_argument("--vehicle-limit", type=int, default=40, help="Frames used for vehicle-detection evaluation")
    parser.add_argument("--skip-occupancy", action="store_true")
    parser.add_argument("--skip-space", action="store_true")
    parser.add_argument("--skip-vehicle", action="store_true")
    args = parser.parse_args()

    pipeline = ParkingPipeline()
    results = {}
    if not args.skip_occupancy:
        print("===== occupancy evaluation =====")
        results["occupancy"] = evaluate_occupancy(pipeline, split=args.split, limit=args.limit)
    if not args.skip_space:
        print("===== space detection evaluation =====")
        results["space_detection"] = evaluate_space_detection(
            pipeline, split=args.split, limit=args.space_limit
        )
    if not args.skip_vehicle:
        print("===== vehicle detection evaluation =====")
        results["vehicle_detection"] = evaluate_vehicle_detection(
            pipeline, split=args.split, limit=args.vehicle_limit
        )
    summary = {
        name: {
            "evaluated_frames": report.get("evaluated_frames"),
            "metrics": report.get("metrics"),
        }
        for name, report in results.items()
    }
    print(json.dumps(summary, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
