"""Hand tracking: a thin wrapper around MediaPipe's HandLandmarker (Tasks API).

MediaPipe >= 1.0 removed the legacy `mp.solutions.hands` API; only the Tasks
API remains. This module is the only place that touches MediaPipe types.
Everything downstream receives a plain `Hand` dataclass, so swapping or
upgrading the tracker never ripples through the rest of the app.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from scripts.download_model import MODEL_PATH, ensure_model

# Pairs of landmark indices to connect when drawing the hand skeleton.
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS
]

# Landmark indices we refer to by name throughout the app.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)


@dataclass
class Hand:
    """One detected hand.

    landmarks: (21, 3) array of normalized coords. x and y are in [0, 1]
        relative to the frame width/height. z is relative depth (smaller
        means closer to the camera), on roughly the same scale as x.
    handedness: "Left" or "Right" as MediaPipe reports it.
    score: handedness confidence in [0, 1].
    """

    landmarks: np.ndarray
    handedness: str
    score: float


class HandTracker:
    """Detects up to `max_hands` hands per frame.

    Uses RunningMode.VIDEO: calls are synchronous (the result belongs to the
    frame you passed in, no callbacks or threads), and MediaPipe can track
    the hand between frames instead of re-detecting it from scratch, which
    is faster and less jittery than IMAGE mode.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        max_hands: int = 1,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        mirrored: bool = True,
    ) -> None:
        # MediaPipe's Left/Right labels come out swapped when the frame has been
        # mirrored (selfie view), so we swap them back to match the real hand.
        self.mirrored = mirrored
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(self, frame_bgr: np.ndarray) -> list[Hand]:
        """Run detection on a BGR frame (as OpenCV gives it). Returns up to max_hands hands (empty if none)."""
        # MediaPipe expects RGB; OpenCV frames are BGR.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires strictly increasing timestamps. On fast loops two
        # frames can land on the same millisecond, so nudge forward if needed.
        ts_ms = max(int(time.monotonic() * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms

        result = self._landmarker.detect_for_video(image, ts_ms)
        hands = []
        for lms, cats in zip(result.hand_landmarks, result.handedness):
            label = cats[0].category_name
            if self.mirrored:
                label = {"Left": "Right", "Right": "Left"}.get(label, label)
            hands.append(Hand(
                landmarks=np.array([(p.x, p.y, p.z) for p in lms], dtype=np.float32),
                handedness=label,
                score=float(cats[0].score),
            ))
        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
