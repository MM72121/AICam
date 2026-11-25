"""Phase 3: record labeled gesture samples to data/gestures.csv.

Keys (camera window focused):
    0-5     start recording that gesture; press it again (or Space) to stop
              0 none   1 point   2 pinch   3 two_fingers   4 fist   5 open_palm
    x       undo the last recording
    q/Esc   quit

While recording, move your hand around: different angles, distances,
positions, slight variations. Varied data makes a model that works in
real use, not just in the exact pose you recorded.

Tune behavior by editing the settings below.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2

from gesture.camera import open_camera
from gesture.drawing import draw_hand, draw_text, to_pixels
from gesture.features import LABELS, NUM_FEATURES, hand_features
from gesture.tracking import HandTracker

WINDOW = "Collect Gestures (0-5 = record, x = undo, q = quit)"
RECORD_COLOR = (0, 0, 255)
IDLE_COLOR = (200, 200, 200)

# --- Settings ---
CAMERA_INDEX = 0
CONTROL_HAND = "Right"    # record only this hand (features mirror Left anyway)
DATA_PATH = Path(__file__).resolve().parent / "data" / "gestures.csv"
SAMPLE_EVERY_N_FRAMES = 3  # ~10 samples/s at 30 FPS; consecutive frames are near-duplicates
TARGET_PER_LABEL = 300     # just a guide shown on screen


def load_counts(path: Path) -> dict[str, int]:
    """Count existing samples per label so the display picks up where you left off."""
    counts = {label: 0 for label in LABELS}
    if path.exists():
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                counts[row["label"]] = counts.get(row["label"], 0) + 1
    return counts


class SampleWriter:
    """Appends samples to the CSV, one recording ("burst") at a time.

    Each burst is written when you stop recording, so a crash loses at most
    the current burst. We remember the file size before the last burst so
    undo can just cut the file back to that point.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(["label"] + [f"f{i}" for i in range(NUM_FEATURES)])
        self._undo_pos: int | None = None
        self._undo_label: str | None = None
        self._undo_count = 0

    def write_burst(self, label: str, rows: list) -> None:
        if not rows:
            return
        with open(self.path, "a", newline="") as f:
            self._undo_pos = f.tell()
            w = csv.writer(f)
            for feats in rows:
                w.writerow([label] + [f"{v:.5f}" for v in feats])
        self._undo_label, self._undo_count = label, len(rows)

    def undo(self) -> tuple[str, int] | None:
        """Remove the last burst. Returns (label, count) removed, or None."""
        if self._undo_pos is None:
            return None
        with open(self.path, "r+") as f:
            f.truncate(self._undo_pos)
        removed = (self._undo_label, self._undo_count)
        self._undo_pos = None  # one level of undo
        return removed


def draw_counts(frame, counts: dict[str, int], recording: str | None) -> None:
    """Panel listing each label's key, name and sample count."""
    y = 60
    for key, label in enumerate(LABELS):
        n = counts.get(label, 0)
        mark = ">" if label == recording else " "
        done = " ok" if n >= TARGET_PER_LABEL else ""
        color = RECORD_COLOR if label == recording else (IDLE_COLOR if n < TARGET_PER_LABEL else (0, 220, 0))
        draw_text(frame, f"{mark}{key} {label:<12}{n:>5}{done}", (10, y), scale=0.6, color=color)
        y += 28


def main() -> int:
    try:
        cap = open_camera(CAMERA_INDEX)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    counts = load_counts(DATA_PATH)
    writer = SampleWriter(DATA_PATH)
    recording: str | None = None  # label currently being recorded
    burst: list = []              # samples in the current recording
    frame_i = 0
    message = ""                  # short status line (e.g. "saved 42 pinch")

    def stop_recording() -> None:
        nonlocal recording, burst, message
        if recording is not None:
            writer.write_burst(recording, burst)
            message = f"saved {len(burst)} {recording}"
            recording, burst = None, []

    print(f"Saving to {DATA_PATH}")
    try:
        with HandTracker(max_hands=2) as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read frame from camera.", file=sys.stderr)
                    return 1
                frame = cv2.flip(frame, 1)
                frame_i += 1

                hands = tracker.process(frame)
                hand = next((hd for hd in hands if hd.handedness == CONTROL_HAND), None)
                for hd in hands:
                    draw_hand(frame, hd)
                    wx, wy = to_pixels(hd, frame)[0]
                    draw_text(frame, hd.handedness, (wx - 30, wy + 30))

                # Record only when the hand is visible, and only every Nth frame.
                if recording and hand is not None and frame_i % SAMPLE_EVERY_N_FRAMES == 0:
                    burst.append(hand_features(hand))
                    counts[recording] += 1

                if recording:
                    status = f"REC {recording}: {len(burst)}" + ("" if hand else f"  (show {CONTROL_HAND.lower()} hand)")
                    cv2.circle(frame, (frame.shape[1] - 30, 30), 12, RECORD_COLOR, -1)
                    draw_text(frame, status, (10, 30), color=RECORD_COLOR)
                else:
                    draw_text(frame, message or "press 0-5 to record", (10, 30))
                draw_counts(frame, counts, recording)
                cv2.imshow(WINDOW, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if ord("0") <= key < ord("0") + len(LABELS):
                    label = LABELS[key - ord("0")]
                    was = recording
                    stop_recording()
                    if label != was:  # same key again = just stop
                        recording, message = label, ""
                elif key == ord(" "):
                    stop_recording()
                elif key == ord("x"):
                    stop_recording()
                    removed = writer.undo()
                    if removed:
                        counts[removed[0]] -= removed[1]
                        message = f"undid {removed[1]} {removed[0]}"
                    else:
                        message = "nothing to undo"
    finally:
        stop_recording()  # don't lose a recording that's in progress when you quit
        cap.release()
        cv2.destroyAllWindows()
    print("Samples per label:", counts)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # Ctrl+C in the terminal: exit quietly (cleanup already ran in finally)
        sys.exit(0)
