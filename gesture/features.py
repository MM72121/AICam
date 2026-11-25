"""Turn a tracked hand into a fixed-length feature vector for the classifier.

Shared by data collection (Phase 3), training (Phase 4) and the live app
(Phase 5). They must all compute features *identically*, or the model sees
different numbers live than it was trained on.
"""

from __future__ import annotations

import numpy as np

from gesture.tracking import MIDDLE_MCP, WRIST, Hand

# Gesture labels, in number-key order (key 0 = "none", key 1 = "point", ...).
LABELS = ["none", "point", "pinch", "two_fingers", "fist", "open_palm"]

NUM_FEATURES = 21 * 3


def hand_features(hand: Hand) -> np.ndarray:
    """63 numbers describing the hand's *shape*, independent of where it is.

    Uses MediaPipe's 3D world landmarks (meters) rather than image coords:
    in the flat image a tilted hand looks squashed (foreshortening), so the
    same gesture at different angles would give very different numbers.

    1. Subtract the wrist: every point is measured from the wrist, so the
       hand's position in the frame doesn't matter.
    2. Divide by hand size (wrist -> middle knuckle): big/small hands and
       MediaPipe's depth-estimate scale don't matter.
    3. Mirror left hands (flip x) so they look like right hands. A model
       trained on one hand then works on either.
    """
    pts = hand.world - hand.world[WRIST]
    size = np.linalg.norm(pts[MIDDLE_MCP])
    pts = pts / max(size, 1e-6)
    if hand.handedness == "Left":
        pts[:, 0] = -pts[:, 0]
    return pts.flatten().astype(np.float32)
