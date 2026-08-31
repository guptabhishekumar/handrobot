"""Tests for the measured reachability model.

These are the tests that keep the declared workspace honest. If someone widens
:class:`~handrobot.config.WorkspaceConfig`, or the arm model changes, these fail
rather than the robot quietly missing its targets during teleoperation.
"""

import numpy as np
import pytest

from handrobot.config import Config, WorkspaceConfig
from handrobot.retarget.reach import (
    REACHABLE_TOLERANCE,
    ReachTable,
    approach_frame,
    warm_start,
)


#: These tests are about the SO-101 specifically. The Panda has enough joints
#: that a straight-down grasp is reachable everywhere, so it needs no such table
#: and none of this applies to it.
ROBOT = "so101"


@pytest.fixture(scope="module")
def table():
    return ReachTable.cached()


@pytest.fixture
def so101_ik():
    from handrobot.retarget.ik import ArmIK

    config = Config(robot=ROBOT)
    return ArmIK(config.ik, config.spec)


def test_approach_frame_is_a_rotation():
    for pitch in np.radians([0, 15, 45, 70]):
        R = approach_frame(0.2, 0.05, pitch)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_zero_pitch_points_straight_down():
    R = approach_frame(0.2, 0.05, 0.0)
    assert np.allclose(R[:, 0], [0.0, 0.0, -1.0], atol=1e-9)


def test_pitch_leans_the_approach_outward():
    x, y = 0.2, 0.0
    for pitch in np.radians([10, 30, 60]):
        approach = approach_frame(x, y, pitch)[:, 0]
        assert approach[0] > 0, "the gripper should lean away from the base"
        assert approach[2] < 0, "the gripper should still point downward"
        assert np.isclose(np.arccos(-approach[2]), pitch, atol=1e-9)


def test_jaw_azimuth_is_respected():
    for azimuth in np.radians([0, 45, 130, -80]):
        R = approach_frame(0.2, 0.05, 0.0, azimuth)
        assert np.isclose(np.arctan2(R[1, 2], R[0, 2]),
                          np.arctan2(np.sin(azimuth), np.cos(azimuth)), atol=1e-9)


def test_warm_start_turns_the_shoulder_towards_the_target(so101_ik):
    for azimuth in np.radians([-40, 0, 40]):
        x, y = 0.22 * np.cos(azimuth), 0.22 * np.sin(azimuth)
        position, _ = so101_ik.forward(warm_start(x, y))
        assert np.isclose(np.arctan2(position[1], position[0]), azimuth, atol=0.25)


def test_pitch_grows_with_height(table):
    pitches = [table.approach_pitch([0.21, 0.0, z]) for z in np.arange(0.02, 0.20, 0.02)]
    assert pitches[0] == pytest.approx(0.0, abs=1e-9)
    assert all(b >= a - 1e-9 for a, b in zip(pitches, pitches[1:])), pitches
    assert pitches[-1] > np.radians(20)


def test_low_targets_are_reachable_straight_down(table):
    for radius in (0.18, 0.22, 0.26):
        assert table.approach_pitch([radius, 0.0, 0.02]) == pytest.approx(0.0, abs=1e-9)


def test_high_targets_are_reported_unreachable(table):
    assert not table.reachable([0.16, 0.0, 0.20])


def test_interpolation_clamps_outside_the_grid(table):
    assert np.isfinite(table.approach_pitch([0.01, 0.0, 0.0]))
    assert np.isfinite(table.approach_pitch([5.0, 0.0, 5.0]))


def test_table_round_trips_through_disk(tmp_path, table):
    path = table.save(tmp_path / "reach.npz")
    restored = ReachTable.load(path)
    assert np.allclose(restored.pitch, table.pitch)
    assert np.allclose(restored.error, table.error)


def test_declared_workspace_is_reachable(table, so101_ik):
    """Every corner of the declared workspace must be reachable in practice.

    Sampled densely rather than at the corners alone, because the reachable
    region is not convex in the way a box is.
    """
    workspace = WorkspaceConfig()
    worst = (0.0, None)
    for radius in np.linspace(workspace.radius_min, workspace.radius_max, 7):
        for azimuth in np.linspace(-workspace.azimuth_max, workspace.azimuth_max, 7):
            for z in np.linspace(workspace.z_min, workspace.z_max, 7):
                target = workspace.from_polar(radius, azimuth, z)
                q = warm_start(target[0], target[1])
                for _ in range(40):
                    result = so101_ik.solve(target, table.frame_for(target), q)
                    q = result.q
                if result.position_error > worst[0]:
                    worst = (result.position_error, (radius, azimuth, z))
    assert worst[0] <= REACHABLE_TOLERANCE, (
        f"worst position error {worst[0] * 1000:.2f} mm at "
        f"radius/azimuth/z {worst[1]}; the declared workspace is too large"
    )


def test_grasp_height_supports_a_true_top_down_approach(table, so101_ik):
    """At grasping height the gripper must genuinely point down, not just nearly."""
    workspace = WorkspaceConfig()
    straight_down = approach_frame(1.0, 0.0, 0.0)
    for radius in np.linspace(workspace.radius_min, workspace.radius_max, 5):
        for azimuth in np.linspace(-workspace.azimuth_max, workspace.azimuth_max, 5):
            target = workspace.from_polar(radius, azimuth, 0.022)
            q = warm_start(target[0], target[1])
            for _ in range(40):
                q = so101_ik.solve(target, table.frame_for(target), q).q
            _, achieved = so101_ik.forward(q)
            angle_from_vertical = np.arccos(np.clip(-achieved[2, 0], -1, 1))
            assert angle_from_vertical < np.radians(5), (
                f"gripper is {np.degrees(angle_from_vertical):.1f} deg off vertical "
                f"at radius {radius:.2f}, azimuth {azimuth:.2f}"
            )


def test_the_table_builds_against_the_so101_no_matter_the_default(monkeypatch):
    """Regression for a cache-masked bug: ReachTable.build used the default
    arm's IK, which silently became the Panda when the default changed. Only
    machines without the cached table (CI, fresh clones) ever executed the
    build, so every local run hid it."""
    from handrobot.retarget import reach as reach_module

    captured = {}

    class SpyIK:
        def __init__(self, config=None, spec=None, **kw):
            captured["spec"] = spec

        def solve(self, *a, **kw):
            class R:
                position_error = 0.0
            return R()

    monkeypatch.setattr("handrobot.retarget.ik.ArmIK", SpyIK)
    reach_module.ReachTable.build()
    assert captured["spec"] is not None and captured["spec"].name == "so101"
