"""Webcam opening (macOS)."""

from __future__ import annotations

import cv2


def open_camera(index: int = 0, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    """Open a webcam using macOS's AVFoundation backend."""
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {index}.\nGrant Camera access to your terminal/IDE in "
            "System Settings > Privacy & Security > Camera, then restart it.")

    # These are requests; the driver may pick the nearest supported mode.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap
