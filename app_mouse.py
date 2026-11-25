"""Phase 2: hard-coded mouse control.

Index knuckle (or fingertip, see POINTER) moves the cursor; thumb-index pinch = left button
(quick pinch = click, hold and move = drag).

Keys (camera window must be focused; it stays on top of other windows):
    c       toggle control on/off (starts OFF)
    q/Esc   quit

Tune behavior by editing the settings below.
"""

from __future__ import annotations

import sys

import cv2

from gesture.camera import open_camera
from gesture.drawing import draw_hand, draw_text, to_pixels
from gesture.fps import FPSCounter
from gesture.mouse import MouseDriver, PinchDetector, ScreenMapper, Smoother, get_screen_size
from gesture.tracking import INDEX_MCP, INDEX_TIP, THUMB_TIP, HandTracker

WINDOW = "Mouse Control (c = toggle, q = quit)"
ON_COLOR = (0, 220, 0)
OFF_COLOR = (0, 0, 255)

# --- Settings ---
CAMERA_INDEX = 0
CONTROL_HAND = "Right"   # "Right" or "Left": which hand moves the mouse
REGION = 0.6             # fraction of the camera frame mapped to the full screen
SMOOTHING_ALPHA = 0.3    # lower = steadier but laggier, higher = snappier but jittery
POINTER = INDEX_MCP      # INDEX_MCP (knuckle, steady when pinching) or INDEX_TIP
PINCH_CLOSE = 0.3        # pinch registers below this ratio (lower = harder to trigger)
PINCH_OPEN = 0.42        # pinch releases above this ratio (keep ~0.1 above PINCH_CLOSE)
DRAG_SLOP_PX = 40        # screen px to move while pinched before a click becomes a drag


def main() -> int:
    try:
        cap = open_camera(CAMERA_INDEX)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    screen_w, screen_h = get_screen_size()
    mapper = ScreenMapper(screen_w, screen_h, REGION)
    smoother = Smoother(SMOOTHING_ALPHA)
    pinch = PinchDetector(PINCH_CLOSE, PINCH_OPEN)
    mouse = MouseDriver(DRAG_SLOP_PX)
    active = False  # start OFF so the cursor doesn't jump the moment the app opens
    fps = FPSCounter()

    cv2.namedWindow(WINDOW)
    try:
        # Keep the window above other apps so you can click it to press 'c'.
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass

    print(f"Screen {screen_w}x{screen_h}. Controlling with {CONTROL_HAND} hand. Press 'c' to toggle.")
    try:
        # Track 2 hands so we can pick out the control hand even when both are visible.
        with HandTracker(max_hands=2) as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read frame from camera.", file=sys.stderr)
                    return 1
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                hands = tracker.process(frame)
                hand = next((hd for hd in hands if hd.handedness == CONTROL_HAND), None)

                # Draw the active region so you know where the screen edges are.
                x0, y0, x1, y1 = mapper.box()
                cv2.rectangle(frame, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)),
                              ON_COLOR if active else OFF_COLOR, 2)
                for hd in hands:
                    draw_hand(frame, hd)
                    wx, wy = to_pixels(hd, frame)[0]
                    draw_text(frame, hd.handedness, (wx - 30, wy + 30))

                if hand is None:
                    # Hand gone: release any held button and forget the smoothed
                    # position so the cursor doesn't glide from a stale spot later.
                    mouse.release()
                    pinch.reset()
                    smoother.reset()
                else:
                    pinched = pinch.update(hand, w / h)
                    if active:
                        px, py = hand.landmarks[POINTER, :2]
                        mouse.move(*smoother.update(mapper.to_screen(px, py)))
                        mouse.set_pressed(pinched)
                    # Show what's measured: a line between the thumb tip and the
                    # index tip (yellow while pinched), plus a dot on the point
                    # that steers the cursor.
                    pts = to_pixels(hand, frame)
                    color = (0, 255, 255) if pinched else (200, 200, 200)
                    cv2.line(frame, tuple(pts[THUMB_TIP]), tuple(pts[INDEX_TIP]), color, 3, cv2.LINE_AA)
                    cv2.circle(frame, tuple(pts[POINTER]), 10, (255, 0, 255), -1, cv2.LINE_AA)

                status = "CONTROL ON" if active else "CONTROL OFF (press c)"
                draw_text(frame, status, (10, 60), color=ON_COLOR if active else OFF_COLOR)
                draw_text(frame, f"FPS: {fps.tick():.1f}", (10, 30))
                if hand is not None:  # live pinch ratio, for tuning PINCH_CLOSE/PINCH_OPEN
                    draw_text(frame, f"pinch ratio: {pinch.ratio:.2f}", (10, 90), scale=0.6)
                cv2.imshow(WINDOW, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    active = not active
                    if not active:
                        mouse.release()
                    smoother.reset()
    finally:
        mouse.release()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # Ctrl+C in the terminal: exit quietly (cleanup already ran in finally)
        sys.exit(0)
