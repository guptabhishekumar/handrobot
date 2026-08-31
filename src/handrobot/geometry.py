"""Small rotation helpers shared by the hand tracker, retargeter and simulator.

Every rotation in this project is a 3x3 matrix whose *columns* are the frame's
x, y and z axes expressed in the parent frame.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def unit(v: np.ndarray) -> np.ndarray:
    """Return ``v`` scaled to unit length, or a zero vector if it is degenerate."""
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < EPS:
        return np.zeros(3)
    return v / n


def frame_from_axes(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Build a right-handed rotation matrix from two non-parallel directions.

    ``primary`` becomes the x axis exactly. ``secondary`` is projected onto the
    plane perpendicular to it and becomes the z axis; y is chosen to complete a
    right-handed triad.

    Raises:
        ValueError: if the two directions are parallel or either is degenerate.
    """
    x = unit(primary)
    s = np.asarray(secondary, dtype=float)
    if np.linalg.norm(x) < EPS:
        raise ValueError("primary axis is degenerate")
    z = s - np.dot(s, x) * x
    if np.linalg.norm(z) < 1e-6:
        raise ValueError("secondary axis is parallel to primary")
    z = unit(z)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3) via SVD."""
    u, _, vt = np.linalg.svd(np.asarray(R, dtype=float))
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    return R


def rotation_geodesic(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in radians of the relative rotation between two rotation matrices."""
    rel = np.asarray(a, dtype=float).T @ np.asarray(b, dtype=float)
    cos = (np.trace(rel) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def slerp_matrix(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Interpolate between two rotation matrices along the geodesic."""
    rel = np.asarray(a, dtype=float).T @ np.asarray(b, dtype=float)
    angle = rotation_geodesic(a, b)
    if angle < 1e-8:
        return np.array(b, dtype=float)
    axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
    axis = unit(axis)
    if np.linalg.norm(axis) < EPS:
        return np.array(b, dtype=float)
    return orthonormalize(a @ axis_angle_to_matrix(axis, angle * float(t)))


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula."""
    a = unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def top_down_frame(jaw_azimuth: float) -> np.ndarray:
    """Gripper frame that approaches straight down with a chosen jaw direction.

    Args:
        jaw_azimuth: angle in radians of the jaw-opening axis within the
            horizontal plane, measured from world +x.
    """
    approach = np.array([0.0, 0.0, -1.0])
    jaw = np.array([np.cos(jaw_azimuth), np.sin(jaw_azimuth), 0.0])
    return frame_from_axes(approach, jaw)
