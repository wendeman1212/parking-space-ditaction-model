import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smart_parking.config import load_app_config  # noqa: E402
from smart_parking.data.annotations import group_annotations_by_image, load_json  # noqa: E402
from smart_parking.data.validate import validate_image_record  # noqa: E402
from smart_parking.io_utils import save_json  # noqa: E402


def main() -> dict:
    config = load_app_config()
    manifests = config.path("paths.manifests")
    frames = load_json(manifests / "pklot_frames.json")
    annotations_by_image = {}
    for subset in config.get("pklot.subsets", []):
        path = config.path("paths.processed") / "pklot" / f"selected_{subset}_spots.json"
        if path.exists():
            data = load_json(path)
            for image_id, anns in group_annotations_by_image(data["annotations"]).items():
                annotations_by_image[int(image_id)] = anns
    results = []
    summary = defaultdict(int)
    for frame in frames:
        result = validate_image_record(frame, annotations_by_image.get(int(frame["image_id"])))
        result["subset"] = frame["subset"]
        result["split"] = frame["split"]
        results.append(result)
        summary["valid" if result["valid"] else "invalid"] += 1
        for problem in result["problems"]:
            summary[problem] += 1
        summary["annotated_spaces"] += int(result.get("annotated_spaces", 0))
    report = {"summary": dict(summary), "records": results}
    save_json(config.path("paths.outputs") / "reports/dataset_validation.json", report)
    print(json.dumps(dict(summary), indent=2))
    return report


if __name__ == "__main__":
    main()
