"""Mouse control building blocks: screen mapping, smoothing, pinch detection.

Each piece is a small independent class so Phase 5 can reuse them with the
trained gesture classifier instead of hard-coded rules.
"""

from __future__ import annotations

import numpy as np
import Quartz  # macOS display APIs; installed with pynput (pyobjc)
from pynput.mouse import Button, Controller

from gesture.tracking import (INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP,
                              THUMB_TIP, WRIST, Hand)


def get_screen_size() -> tuple[int, int]:
    """Size of the main display in points, the units the macOS cursor uses.

    Points, not physical pixels: a Retina display reports half its pixel
    resolution, which is what pynput expects.
    """
    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    return int(bounds.size.width), int(bounds.size.height)


class ScreenMapper:
    """Maps a central box of the camera frame onto the full screen.

    region=0.6 means the middle 60% of the frame (20% margin per side) covers
    the whole screen. Your hand reaches the screen edges without leaving the
    camera's view, where tracking gets unreliable.
    """

    def __init__(self, screen_w: int, screen_h: int, region: float = 0.6) -> None:
        self.screen_w, self.screen_h = screen_w, screen_h
        self.margin = (1.0 - region) / 2

    def box(self) -> tuple[float, float, float, float]:
        """The active region as normalized (x0, y0, x1, y1), for drawing."""
        m = self.margin
        return m, m, 1 - m, 1 - m

    def to_screen(self, nx: float, ny: float) -> tuple[float, float]:
        """Normalized frame coords -> screen coords, clamped to the screen."""
        x0, y0, x1, y1 = self.box()
        u = np.clip((nx - x0) / (x1 - x0), 0.0, 1.0)
        v = np.clip((ny - y0) / (y1 - y0), 0.0, 1.0)
        # -1 so the far edge maps to the last pixel, not one past it.
        return u * (self.screen_w - 1), v * (self.screen_h - 1)


class Smoother:
    """Exponential smoothing with a small dead zone.

    Each update moves `alpha` of the way toward the target:
        pos += alpha * (target - pos)
    Lower alpha = steadier but laggier; higher = snappier but jittery.
    The dead zone ignores sub-pixel-ish wobble while your hand is held still,
    so the cursor doesn't shimmer.
    """

    def __init__(self, alpha: float = 0.3, deadzone_px: float = 2.0) -> None:
        self.alpha = alpha
        self.deadzone = deadzone_px
        self.pos: np.ndarray | None = None

    def update(self, target: tuple[float, float]) -> tuple[float, float]:
        t = np.asarray(target, dtype=float)
        if self.pos is None:
            self.pos = t  # first sample: jump straight there
        elif np.linalg.norm(t - self.pos) > self.deadzone:
            self.pos = self.pos + self.alpha * (t - self.pos)
        return float(self.pos[0]), float(self.pos[1])

    def reset(self) -> None:
        """Forget history (e.g. when the hand reappears), so we don't glide from a stale spot."""
        self.pos = None


def _pinch_ratio(pts: np.ndarray) -> float:
    """Thumb-tip-to-index-tip distance divided by hand size.

    Hand size = the larger of palm length (wrist -> middle knuckle) and palm
    width (index knuckle -> pinky knuckle). Tilting the hand forward/back
    shrinks the length in the image, turning it sideways shrinks the width,
    but rarely both at once, so the max stays fairly stable at any angle.
    """
    size = max(np.linalg.norm(pts[MIDDLE_MCP] - pts[WRIST]),
               np.linalg.norm(pts[PINKY_MCP] - pts[INDEX_MCP]))
    return float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]) / max(size, 1e-6))


def _thumb_nearest_tip(pts: np.ndarray) -> bool:
    """True if the thumb is nearer the index *tip* than the index's middle joint or knuckle.

    Rejects thumb-on-the-side-of-the-index poses: there the thumb tip touches
    the base of the finger, not the tip, so it's not a pinch.
    """
    d = lambda i: np.linalg.norm(pts[THUMB_TIP] - pts[i])
    return d(INDEX_TIP) < d(INDEX_PIP) and d(INDEX_TIP) < d(INDEX_MCP)


class PinchDetector:
    """Thumb-tip-to-index-tip pinch with hysteresis, robust to hand angle.

    The pinch ratio (see _pinch_ratio) is measured two ways and the smaller
    one is used:
      * on the 2D image (x scaled by the aspect ratio so x and y units match)
      * on MediaPipe's estimated 3D world landmarks
    The 2D measure is precise when the palm faces the camera; the 3D one
    copes better with tilted hands but its depth estimate is noisier. Taking
    the min means a pinch counts if either view clearly sees one. The
    "thumb nearest the index tip" check (also either view) stops the min
    from letting thumb-on-knuckle poses through.

    Two thresholds (hysteresis): the pinch closes below `close_at` and only
    opens again above `open_at`. With a single cutoff, a distance hovering
    right at it would flicker on/off and fire a burst of clicks.

    A pinch also has to be seen for `confirm_frames` frames in a row, so a
    single noisy frame (e.g. thumb passing near the index) doesn't click.
    """

    def __init__(self, close_at: float = 0.3, open_at: float = 0.42, confirm_frames: int = 2) -> None:
        self.close_at, self.open_at = close_at, open_at
        self.confirm_frames = confirm_frames
        self.pinched = False
        self.ratio = 0.0
        self._closed_streak = 0

    def update(self, hand: Hand, aspect: float) -> bool:
        img = hand.landmarks[:, :2] * (aspect, 1.0)
        candidates = [_pinch_ratio(p) for p in (img, hand.world) if _thumb_nearest_tip(p)]
        # No view has the thumb at the tip -> treat as wide open.
        self.ratio = min(candidates) if candidates else 1.0
        self._closed_streak = self._closed_streak + 1 if self.ratio < self.close_at else 0
        if self.pinched and self.ratio > self.open_at:
            self.pinched = False
        elif not self.pinched and self._closed_streak >= self.confirm_frames:
            self.pinched = True
        return self.pinched

    def reset(self) -> None:
        self.pinched = False
        self._closed_streak = 0


class MouseDriver:
    """Moves the OS cursor and presses/releases the left button.

    Pinch closing = mouse down and opening = mouse up, so a quick pinch is
    a click and pinch-and-move is a drag.

    "Click slop" (the same trick touchscreens use to tell taps from drags):
    while pinched, the cursor stays locked at the press point until the hand
    has moved more than `drag_slop_px`. A hand is never perfectly still, and
    pinching/unpinching itself shifts the fingertip, so without this nearly
    every click became a tiny drag. Release inside the slop = a clean click;
    move past it = a drag.
    """

    def __init__(self, drag_slop_px: float = 40.0) -> None:
        self._mouse = Controller()
        self.drag_slop = drag_slop_px
        self.pressed = False
        self.dragging = False
        self._last = (0.0, 0.0)    # last position we sent to the OS
        self._anchor = (0.0, 0.0)  # where the button went down

    def move(self, x: float, y: float) -> None:
        if self.pressed and not self.dragging:
            if np.hypot(x - self._anchor[0], y - self._anchor[1]) < self.drag_slop:
                return  # still within the slop: keep the cursor locked for a clean click
            self.dragging = True
        self._last = (x, y)
        self._mouse.position = (int(x), int(y))

    def set_pressed(self, pressed: bool) -> None:
        if pressed and not self.pressed:
            self._anchor = self._last
            self.dragging = False
            self._mouse.press(Button.left)
        elif not pressed and self.pressed:
            self._mouse.release(Button.left)
            self.dragging = False
        self.pressed = pressed

    def release(self) -> None:
        """Safety: never leave the button stuck down (hand lost, control off, quit)."""
        self.set_pressed(False)
