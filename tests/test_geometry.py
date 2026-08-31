import numpy as np
import pytest

from handrobot.geometry import (
    axis_angle_to_matrix,
    frame_from_axes,
    orthonormalize,
    rotation_geodesic,
    slerp_matrix,
    top_down_frame,
    unit,
)


def test_unit_normalises_and_survives_zero():
    assert np.isclose(np.linalg.norm(unit([3.0, 4.0, 0.0])), 1.0)
    assert np.allclose(unit([0.0, 0.0, 0.0]), np.zeros(3))


def test_frame_from_axes_is_right_handed_and_orthonormal():
    R = frame_from_axes(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)
    # The primary direction is preserved exactly as the x axis.
    assert np.allclose(R[:, 0], [1.0, 0.0, 0.0])


def test_frame_from_axes_rejects_parallel_inputs():
    with pytest.raises(ValueError):
        frame_from_axes(np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]))


def test_top_down_frame_points_the_approach_axis_at_the_table():
    for azimuth in np.linspace(-np.pi, np.pi, 9):
        R = top_down_frame(azimuth)
        assert np.allclose(R[:, 0], [0.0, 0.0, -1.0], atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0)
        # The jaw axis stays horizontal, at the requested azimuth.
        assert np.isclose(R[2, 2], 0.0, atol=1e-9)
        assert np.isclose(np.arctan2(R[1, 2], R[0, 2]), np.arctan2(np.sin(azimuth), np.cos(azimuth)))


def test_orthonormalize_repairs_a_perturbed_rotation():
    R = top_down_frame(0.3) + 0.01 * np.random.default_rng(0).standard_normal((3, 3))
    fixed = orthonormalize(R)
    assert np.allclose(fixed.T @ fixed, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(fixed), 1.0)


def test_rotation_geodesic_matches_a_known_angle():
    R = axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), 0.7)
    assert np.isclose(rotation_geodesic(np.eye(3), R), 0.7, atol=1e-9)


def test_slerp_endpoints_and_midpoint():
    a = np.eye(3)
    b = axis_angle_to_matrix(np.array([0.0, 1.0, 0.0]), 1.2)
    assert np.allclose(slerp_matrix(a, b, 0.0), a, atol=1e-9)
    assert np.allclose(slerp_matrix(a, b, 1.0), b, atol=1e-6)
    assert np.isclose(rotation_geodesic(a, slerp_matrix(a, b, 0.5)), 0.6, atol=1e-6)
