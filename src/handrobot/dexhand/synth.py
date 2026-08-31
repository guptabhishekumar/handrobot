"""Synthetic human hands for training the retargeting network.

The network needs thousands of (human landmarks, matching robot pose) examples.
Recording them would take days; generating them takes seconds, because the
supervision is not "the right joint angles" -- nobody knows those -- but "the
fingertips should land where the human's did", which the differentiable
kinematics scores directly. All the generator must supply is a wide, plausible
distribution of hand shapes.

A hand here is a simple articulated skeleton in the palm frame (x towards the
fingers, z out of the palm's back, matching :func:`handrobot.hands.geometry
.palm_frame`): four digits, each a chain of bones with sampled curl at every
knuckle, sampled sideways spread, and per-person bone lengths. Curls are drawn
jointly -- a fist, a pinch, a point and everything between -- because the
network will meet all of these live.
"""

from __future__ import annotations

import numpy as np
import torch

# MediaPipe landmark indices for the digits the LEAP hand has.
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
THUMB = (1, 2, 3, 4)

#: The sixteen human keypoints matched against the robot's, in the same order
#: as :data:`handrobot.dexhand.fk.KEYPOINT_NAMES`.
HUMAN_KEYPOINTS = INDEX + MIDDLE + RING + THUMB

#: Mean bone lengths in metres (proximal, middle, distal), scaled per sample.
FINGER_BONES = {
    "index": (0.040, 0.024, 0.019),
    "middle": (0.044, 0.027, 0.020),
    "ring": (0.041, 0.026, 0.019),
}
THUMB_BONES = (0.046, 0.032, 0.025)

#: Knuckle positions in the palm frame, roughly matching MediaPipe's layout.
KNUCKLES = {
    "index": (0.088, 0.026, 0.0),
    "middle": (0.092, 0.000, 0.0),
    "ring": (0.086, -0.024, 0.0),
}
THUMB_ROOT = (0.028, 0.038, -0.006)


def _finger_points(root, bones, curls, spread, scale):
    """Chain a digit's keypoints given curl angles at each knuckle."""
    points = []
    position = np.asarray(root) * scale
    direction_angle = spread
    curl_total = 0.0
    for bone, curl in zip(bones, curls):
        curl_total += curl
        step = np.array([
            np.cos(curl_total) * np.cos(direction_angle),
            np.cos(curl_total) * np.sin(direction_angle),
            -np.sin(curl_total),
        ]) * bone * scale
        position = position + step
        points.append(position.copy())
    return [np.asarray(root) * scale] + points


def sample_hands(n: int, rng: np.random.Generator) -> torch.Tensor:
    """(n, 16, 3) human keypoints in the palm frame, in metres."""
    hands = np.zeros((n, 16, 3))
    for i in range(n):
        scale = rng.uniform(0.85, 1.15)
        # A shared "grip" factor makes fists and open hands common, with
        # independent per-finger variation on top so single-finger poses occur.
        grip = rng.uniform(-0.25, 1.0)
        out = []
        for name in ("index", "middle", "ring"):
            base_curl = np.clip(grip + rng.normal(0, 0.25), -0.35, 1.15)
            curls = (
                base_curl * rng.uniform(0.5, 0.9),
                base_curl * rng.uniform(0.7, 1.1),
                base_curl * rng.uniform(0.4, 0.8),
            )
            spread = rng.normal(0.0, 0.12) + {"index": 0.10, "middle": 0.0, "ring": -0.10}[name]
            points = _finger_points(KNUCKLES[name], FINGER_BONES[name], curls, spread, scale)
            out.extend(points)
        # Thumb: opposition swings it across the palm, flexion curls it.
        opposition = rng.uniform(0.1, 1.2)
        flex = np.clip(grip * rng.uniform(0.4, 1.0) + rng.normal(0, 0.2), -0.2, 1.1)
        position = np.asarray(THUMB_ROOT) * scale
        thumb_points = [position.copy()]
        direction = np.array([np.cos(opposition) * 0.4 + 0.4, 0.9 - opposition * 0.55, -0.25 * opposition])
        direction /= np.linalg.norm(direction)
        curl_axis = np.cross(direction, [0.0, 0.0, 1.0])
        curl_axis /= np.linalg.norm(curl_axis)
        heading = direction.copy()
        for bone, curl in zip(THUMB_BONES, (flex * 0.5, flex * 0.8, flex * 0.7)):
            c, s = np.cos(curl), np.sin(curl)
            heading = (c * heading + s * np.cross(curl_axis, heading)
                       + (1 - c) * curl_axis * np.dot(curl_axis, heading))
            position = position + heading * bone * scale
            thumb_points.append(position.copy())
        out.extend(thumb_points)
        hands[i] = np.stack(out)
    return torch.tensor(hands, dtype=torch.float32)


def landmarks_to_keypoints(world: np.ndarray, handedness: str = "Right") -> np.ndarray:
    """Live MediaPipe world landmarks to the (16, 3) palm-frame keypoints.

    ``handedness`` is MediaPipe's own label for the detected geometry. A
    mirrored camera turns the operator's right hand into left-handed geometry
    -- and in any fixed frame convention that arrives with the curl axis
    negated, which is the difference between the robot copying you and doing
    the opposite. Rather than guessing parity from the pose (every geometric
    heuristic has a flat pose that defeats it), the detector's dedicated
    classifier is trusted: geometry labelled "Left" is reflected back to
    right-handed before the frame is built.

    The frame itself comes from the landmarks' anatomy, in the training
    convention -- x towards the fingers, y towards the thumb, z out of the back
    of the hand, flexion towards -z:

        x = wrist -> middle knuckle
        y = the knuckle line, index minus ring
        z = x cross y
    """
    world = np.asarray(world, dtype=float).copy()
    if handedness == "Left":
        world[:, 0] *= -1.0

    wrist = world[0]
    x = world[9] - wrist
    x_norm = np.linalg.norm(x)
    y_raw = world[5] - world[13]
    z = np.cross(x, y_raw)
    z_norm = np.linalg.norm(z)
    # Coincident or collinear landmarks -- a tracker glitch -- leave no palm
    # plane to build a frame from. Returning silently-zero keypoints would put
    # a garbage pose into a recording or command one live; refusing the frame
    # lets both callers just skip it.
    if x_norm < 1e-6 or z_norm < 1e-6:
        raise ValueError("degenerate hand landmarks: no palm frame")
    x = x / x_norm
    z = z / z_norm
    y = np.cross(z, x)
    frame = np.stack([x, y, z], axis=1)
    local = (world - wrist) @ frame
    return local[list(HUMAN_KEYPOINTS)]
