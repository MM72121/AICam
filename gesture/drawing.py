"""OpenCV drawing helpers for the hand skeleton and HUD text.

We draw with plain cv2 calls instead of MediaPipe's drawing_utils. Those
expect MediaPipe landmark objects, and plain cv2 keeps this module independent of
MediaPipe API changes.
"""

from __future__ import annotations

import cv2
import numpy as np

from gesture.tracking import FINGERTIPS, HAND_CONNECTIONS, Hand

# BGR colors
LINE_COLOR = (255, 255, 255)
JOINT_COLOR = (0, 140, 255)
TIP_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_SHADOW = (0, 0, 0)


def to_pixels(hand: Hand, frame: np.ndarray) -> np.ndarray:
    """Convert normalized landmarks to integer (x, y) pixel coords, shape (21, 2)."""
    h, w = frame.shape[:2]
    return (hand.landmarks[:, :2] * (w, h)).astype(int)


def draw_hand(frame: np.ndarray, hand: Hand) -> None:
    """Draw the 21 landmarks and their connections onto `frame` in place."""
    pts = to_pixels(hand, frame)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), LINE_COLOR, 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        is_tip = i in FINGERTIPS
        cv2.circle(frame, (x, y), 7 if is_tip else 4,
                   TIP_COLOR if is_tip else JOINT_COLOR, -1, cv2.LINE_AA)


def draw_text(frame: np.ndarray, text: str, org: tuple[int, int],
              scale: float = 0.7, color: tuple[int, int, int] = TEXT_COLOR) -> None:
    """Text with a dark outline so it stays readable on any background."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT_SHADOW, 4, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
