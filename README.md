# Smart Parking AI

A Flask application for analyzing parking-lot images and videos. It detects
parking spaces and vehicles, estimates occupancy, and provides annotated media
and JSON results.

## Requirements

- Python 3.10 or later
- Git

## Setup

From the project directory, create and activate a virtual environment, then
install the Python dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, run commands using the virtual
environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the web app

```powershell
python app.py
```

Open <http://127.0.0.1:5000>. Upload a parking-lot image or video to analyze it.
The API health endpoint is available at <http://127.0.0.1:5000/api/health>.

## Parking layouts and model weights

Model checkpoints and datasets are intentionally excluded from this repository.
The application uses the configured local checkpoints when available and uses
its built-in detection fallbacks otherwise. For best results, provide a parking
layout with the upload or install the trained checkpoints at the paths specified
in `configs/parking_space_detector.yaml` and
`configs/occupancy_classifier.yaml`.

A layout is a JSON object with a `spaces` array. Each space must have a polygon
of at least three `[x, y]` points in the uploaded image's pixel coordinates:

```json
{
  "spaces": [
    {
      "space_id": "P001",
      "polygon": [[100, 80], [190, 80], [190, 150], [100, 150]]
    }
  ]
}
```

Without a suitable layout or trained space-detector checkpoint, automatic
detection may not identify any parking spaces in a given image. Upload a clear,
adequately sized image of the parking area for best results.

## Tests

```powershell
python -m pytest -q
```
