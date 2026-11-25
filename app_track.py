"""Phase 1: live hand tracking with landmarks and FPS. Press q (or Esc) to quit.

Tune behavior by editing the settings below.
"""

from __future__ import annotations

import sys

import cv2

from gesture.camera import open_camera
from gesture.drawing import draw_hand, draw_text, to_pixels
from gesture.fps import FPSCounter
from gesture.tracking import HandTracker

WINDOW = "Hand Tracking (q to quit)"

# --- Settings ---
CAMERA_INDEX = 0
MIRROR = True     # flip horizontally like a selfie view
MAX_HANDS = 2


def main() -> int:
    try:
        cap = open_camera(CAMERA_INDEX)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    fps = FPSCounter()
    try:
        with HandTracker(max_hands=MAX_HANDS, mirrored=MIRROR) as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read frame from camera.", file=sys.stderr)
                    return 1

                if MIRROR:
                    frame = cv2.flip(frame, 1)

                for hand in tracker.process(frame):
                    draw_hand(frame, hand)
                    # Labels each hand next to its wrist (landmark 0).
                    wx, wy = to_pixels(hand, frame)[0]
                    draw_text(frame, hand.handedness, (wx - 30, wy + 30))

                draw_text(frame, f"FPS: {fps.tick():.1f}", (10, 30))
                cv2.imshow(WINDOW, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q or Esc
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # Ctrl+C in the terminal: exit quietly
        sys.exit(0)
