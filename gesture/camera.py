"""Cross-platform webcam opening."""

from __future__ import annotations

import sys

import cv2


def open_camera(index: int = 0, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    """Open a webcam with the best backend for the current OS.

    macOS uses AVFoundation. On Windows, DirectShow opens much faster than
    the default Media Foundation backend and handles resolution requests
    more reliably.
    """
    if sys.platform == "darwin":
        backend = cv2.CAP_AVFOUNDATION
    elif sys.platform.startswith("win"):
        backend = cv2.CAP_DSHOW
    else:
        backend = cv2.CAP_ANY

    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        hint = ""
        if sys.platform == "darwin":
            hint = ("\nOn macOS, grant Camera access to your terminal/IDE in "
                    "System Settings > Privacy & Security > Camera, then restart it.")
        raise RuntimeError(f"Could not open camera {index}.{hint}")

    # These are requests; the driver may pick the nearest supported mode.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap
