"""Download the MediaPipe hand landmarker model (.task bundle) into models/.

The Tasks API doesn't bundle model weights in the pip package; you point it
at a .task file on disk. Run directly, or let HandTracker call it on demand.
"""

from __future__ import annotations

import shutil
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Return the model path, downloading the file first if it's missing."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model to {path} ...")
    tmp = path.with_suffix(".part")
    # python.org builds on macOS don't use the system certificate store, so
    # HTTPS fails unless you run "Install Certificates.command". certifi's CA
    # bundle (already installed as a MediaPipe dependency) avoids that step.
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(MODEL_URL, context=ctx) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(path)  # atomic rename so a failed download never looks complete
    print(f"Done ({path.stat().st_size / 1e6:.1f} MB).")
    return path


if __name__ == "__main__":
    try:
        ensure_model()
    except Exception as exc:  # network errors, permission errors, ...
        sys.exit(f"Model download failed: {exc}")
