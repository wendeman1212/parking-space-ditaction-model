import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smart_parking.data.download import main  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the documented PKLot/MetaPKLot subset and COCO evaluation images")
    parser.add_argument("--skip-coco", action="store_true", help="Only download the PKLot subset")
    args = parser.parse_args()
    result = main(skip_coco=args.skip_coco)
    print(json.dumps(result, indent=2, default=str)[:6000])
