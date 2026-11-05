import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smart_parking.pipeline import ParkingPipeline  # noqa: E402


def main() -> dict:
    parser = argparse.ArgumentParser(description="Run the smart-parking pipeline on an image or video")
    parser.add_argument("--source", required=True, help="Image or video path")
    parser.add_argument("--layout", default=None, help="Optional JSON layout of verified parking spaces")
    parser.add_argument("--output-name", default=None, help="Stem used for output files")
    parser.add_argument("--no-save", action="store_true", help="Do not write annotated output or JSON")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum processed video frames")
    parser.add_argument("--sample-fps", type=float, default=None, help="Process at most this many frames per second")
    args = parser.parse_args()

    pipeline = ParkingPipeline()
    if args.source.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        result = pipeline.process_video(
            args.source,
            layout=args.layout,
            save=not args.no_save,
            output_name=args.output_name,
            max_frames=args.max_frames,
            sample_fps=args.sample_fps,
        )
    else:
        result = pipeline.process_image(
            args.source,
            layout=args.layout,
            save=not args.no_save,
            output_name=args.output_name,
        )
    summary = {
        key: result.get(key)
        for key in (
            "source",
            "kind",
            "scene",
            "statistics",
            "occupancy_reportable",
            "annotated_image",
            "annotated_video",
            "result_json",
            "frames_processed",
            "timings_ms",
        )
        if key in result
    }
    print(json.dumps(summary, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
