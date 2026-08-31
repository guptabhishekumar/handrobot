"""Data carried between the hand tracker and the retargeter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

N_LANDMARKS = 21

#: The rigid part of the hand: the wrist and the four knuckles. These barely
#: move relative to one another whatever the fingers do, which is what makes
#: them the right reference for both position and scale. Anything involving a
#: fingertip moves when you pinch -- and a reference that moves when you grasp
#: means the arm lurches at the exact moment precision matters most.
PALM_LANDMARKS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

#: Direction along the knuckles, used for the jaw angle. Rolling your wrist
#: rotates it; pinching does not.
KNUCKLE_AXIS = (INDEX_MCP, PINKY_MCP)

#: Direction the hand points, from the wrist to the middle knuckle.
POINTING_AXIS = (WRIST, MIDDLE_MCP)


@dataclass(frozen=True)
class Landmarks:
    """One detected hand, exactly as MediaPipe reports it."""

    image: np.ndarray
    """(21, 3) landmarks normalised to [0, 1] in image space; z is relative depth."""

    world: np.ndarray
    """(21, 3) metric landmarks in metres, origin at the hand's geometric centre."""

    handedness: str
    """``"Left"`` or ``"Right"`` as seen in the (already mirrored) preview image."""

    score: float
    """Detector confidence in [0, 1]."""

    def __post_init__(self) -> None:
        if self.image.shape != (N_LANDMARKS, 3):
            raise ValueError(f"image landmarks must be (21, 3), got {self.image.shape}")
        if self.world.shape != (N_LANDMARKS, 3):
            raise ValueError(f"world landmarks must be (21, 3), got {self.world.shape}")


@dataclass(frozen=True)
class HandPose:
    """A hand reduced to what the robot actually needs."""

    palm_position: np.ndarray
    """Centre of the rigid palm, in metres, in the camera frame.

    Deliberately not the pinch point: the palm does not move when the fingers
    close, so grasping and moving stay independent.
    """

    rotation: np.ndarray
    """Palm frame in camera coordinates. Columns are (pointing, up, knuckles).

    Built entirely from the rigid landmarks, for the same reason.
    """

    pinch_distance: float
    """Thumb-tip to index-tip distance in metres."""

    depth: float
    """Estimated distance from the camera to the hand centre, in metres."""

    landmarks: Landmarks
    """The raw detection, kept for on-screen overlays."""

    timestamp: float
    """Seconds since the tracker started."""
