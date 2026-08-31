"""Recover a metric 3D hand pose from a single uncalibrated webcam.

MediaPipe reports two landmark sets per hand: normalised image coordinates, and
"world" landmarks in metres whose origin sits at the hand's geometric centre.
Neither alone gives an absolute position -- the image set has no scale, and the
world set has no translation. Combining them does.

Two choices here matter more than anything else for whether teleoperation is
usable:

**Everything is built from the rigid palm.** The wrist and the four knuckles
hardly move relative to one another no matter what the fingers do. Deriving
position and orientation from them means closing your hand to grasp something
does not also drag the robot sideways. Only the gripper opening reads the
fingertips.

**Scale is fitted, not measured.** Estimating depth from a single bone works
until the hand rotates, at which point that bone foreshortens and the depth
estimate jumps. Fitting one scale factor across all five palm landmarks in a
least-squares sense averages the foreshortening away and is far steadier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from handrobot.geometry import frame_from_axes
from handrobot.hands.types import (
    HandPose,
    INDEX_TIP,
    KNUCKLE_AXIS,
    Landmarks,
    PALM_LANDMARKS,
    POINTING_AXIS,
    THUMB_TIP,
)


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixels."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> "CameraIntrinsics":
        """Assume a symmetric pinhole camera with the given horizontal field of view."""
        f = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        return cls(width=width, height=height, fx=f, fy=f,
                   cx=width / 2.0, cy=height / 2.0)

    def unproject(self, u: float, v: float, depth: float) -> np.ndarray:
        """Pixel plus depth to a 3D point in the camera frame (x right, y down, z forward)."""
        return np.array(
            [(u - self.cx) * depth / self.fx, (v - self.cy) * depth / self.fy, depth]
        )


def pinch_distance(landmarks: Landmarks) -> float:
    """Metric distance between thumb tip and index tip, in metres."""
    return float(np.linalg.norm(landmarks.world[THUMB_TIP] - landmarks.world[INDEX_TIP]))


def estimate_hand_depth(
    landmarks: Landmarks,
    intrinsics: CameraIntrinsics,
    min_pixels: float = 12.0,
) -> float | None:
    """Distance from the camera to the palm, in metres.

    Fits the single weak-perspective scale ``s`` that best explains the observed
    pixel offsets ``p`` given the metric offsets ``w``, over the rigid palm
    landmarks only::

        p ~ s * w        s = sum(p . w) / sum(w . w)        depth = f / s

    Using five points rather than one segment means a rotation that foreshortens
    one of them is averaged against the others instead of moving the whole
    estimate. Restricting it to the palm means the fingers cannot affect it.

    Returns ``None`` when the hand is too small on screen for the fit to mean
    anything, which is also what a hallucinated detection looks like.
    """
    indices = list(PALM_LANDMARKS)
    scale = np.array([intrinsics.width, intrinsics.height])

    pixels = landmarks.image[indices, :2] * scale
    pixels = pixels - pixels.mean(axis=0)
    metres = landmarks.world[indices, :2]
    metres = metres - metres.mean(axis=0)

    spread = float(np.linalg.norm(pixels, axis=1).max())
    if spread < min_pixels:
        return None

    denominator = float(np.sum(metres * metres))
    if denominator < 1e-9:
        return None
    s = float(np.sum(pixels * metres) / denominator)
    if s <= 1e-6:
        return None
    return float(intrinsics.fx / s)


def palm_frame(world: np.ndarray) -> np.ndarray:
    """Right-handed frame from the rigid palm landmarks.

    * x (pointing): wrist to the middle knuckle -- where the hand is aimed.
    * z (knuckles): index knuckle to pinky knuckle -- rolls with your wrist,
      and drives the direction the jaws open.
    * y: completes the triad.

    Raises:
        ValueError: when the two directions are parallel, which happens when the
            hand is edge-on and the detection has collapsed.
    """
    pointing = world[POINTING_AXIS[1]] - world[POINTING_AXIS[0]]
    knuckles = world[KNUCKLE_AXIS[1]] - world[KNUCKLE_AXIS[0]]
    return frame_from_axes(pointing, knuckles)


def palm_centre(world: np.ndarray) -> np.ndarray:
    """Centre of the rigid palm, relative to the world-landmark origin."""
    return world[list(PALM_LANDMARKS)].mean(axis=0)


def resolve_hand_pose(
    landmarks: Landmarks,
    intrinsics: CameraIntrinsics,
    timestamp: float,
    world_z_sign: float = 1.0,
    depth_range: tuple[float, float] = (0.15, 1.60),
) -> tuple[HandPose | None, str | None]:
    """Same as :func:`hand_pose_from_landmarks`, but says why it failed.

    A detected hand that yields no usable pose is a gap in the control loop, and
    gaps are what make teleoperation feel unstable. Knowing which check rejected
    it is the difference between "move closer to the camera" and "turn a light
    on", so the reason is surfaced on screen rather than discarded.
    """
    depth = estimate_hand_depth(landmarks, intrinsics)
    if depth is None:
        return None, "hand too small in frame"
    if depth < depth_range[0]:
        return None, "hand too close to the camera"
    if depth > depth_range[1]:
        return None, "hand too far from the camera"

    world = landmarks.world.copy()
    world[:, 2] *= world_z_sign
    try:
        rotation = palm_frame(world)
    except ValueError:
        return None, "palm edge-on to the camera"

    centre_uv = landmarks.image[:, :2].mean(axis=0) * np.array(
        [intrinsics.width, intrinsics.height]
    )
    centre = intrinsics.unproject(centre_uv[0], centre_uv[1], depth)
    return (
        HandPose(
            palm_position=centre + palm_centre(world),
            rotation=rotation,
            pinch_distance=pinch_distance(landmarks),
            depth=depth,
            landmarks=landmarks,
            timestamp=timestamp,
        ),
        None,
    )


def hand_pose_from_landmarks(
    landmarks: Landmarks,
    intrinsics: CameraIntrinsics,
    timestamp: float,
    world_z_sign: float = 1.0,
    depth_range: tuple[float, float] = (0.15, 1.60),
) -> HandPose | None:
    """Assemble a :class:`HandPose`, or ``None`` if the detection is unusable.

    Args:
        landmarks: one detected hand.
        intrinsics: camera model used to convert pixels to metres.
        timestamp: seconds since tracking started.
        world_z_sign: flips the depth axis of the world landmarks. MediaPipe's
            sign convention has varied between releases; the ``handcheck``
            diagnostic reports which value matches the live camera.
        depth_range: plausible distances from camera to hand, in metres.

    Returns:
        The pose, or ``None`` when depth could not be fitted, the depth is
        implausible, or the hand is too degenerate to define a frame. Use
        :func:`resolve_hand_pose` when the reason matters.
    """
    return resolve_hand_pose(
        landmarks, intrinsics, timestamp, world_z_sign, depth_range
    )[0]

