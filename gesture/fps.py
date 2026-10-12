"""Smoothed frames-per-second counter."""

from __future__ import annotations

import time


class FPSCounter:
    """FPS as an exponential moving average of frame intervals.

    A raw 1/dt value jumps around every frame, so it's unreadable. The EMA
    gives a stable number that still reacts within about a second.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self._last = None
        self._avg_dt = None

    def tick(self) -> float:
        """Call once per frame; returns the current smoothed FPS."""
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            self._avg_dt = dt if self._avg_dt is None else (
                self.alpha * dt + (1 - self.alpha) * self._avg_dt)
        self._last = now
        return 1.0 / self._avg_dt if self._avg_dt else 0.0
