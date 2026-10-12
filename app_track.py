"""Phase 1: live hand tracking with landmarks and FPS. Press q (or Esc) to quit.

    python app_track.py [--camera N] [--no-mirror]
"""

from __future__ import annotations

import argparse
import sys

import cv2

from gesture.camera import open_camera
from gesture.drawing import draw_hand, draw_text, to_pixels
from gesture.fps import FPSCounter
from gesture.tracking import HandTracker

WINDOW = "Hand Tracking (q to quit)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--no-mirror", action="store_true", help="don't flip the image horizontally")
    parser.add_argument("--hands", type=int, default=2, help="max hands to track (default 2)")
    args = parser.parse_args()

    try:
        cap = open_camera(args.camera)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    fps = FPSCounter()
    try:
        with HandTracker(max_hands=args.hands, mirrored=not args.no_mirror) as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read frame from camera.", file=sys.stderr)
                    return 1

                # Mirror like a selfie view: moving your hand right moves it right
                # on screen. Flip *before* tracking so landmark coords match what
                # you see (this matters once landmarks drive the cursor).
                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)

                for hand in tracker.process(frame):
                    draw_hand(frame, hand)
                    # Label each hand next to its wrist (landmark 0).
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
    sys.exit(main())
