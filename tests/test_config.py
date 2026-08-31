import numpy as np
import pytest

from handrobot.config import SimConfig, WorkspaceConfig


def test_control_rate_divides_the_physics_timestep_exactly():
    sim = SimConfig()
    assert sim.frame_skip == 20
    assert np.isclose(1.0 / sim.control_dt, sim.control_hz)


def test_frame_skip_rejects_an_impossible_control_rate():
    with pytest.raises(ValueError):
        SimConfig(control_hz=10_000.0).frame_skip


def test_workspace_clip_and_contains_agree():
    workspace = WorkspaceConfig()
    for outside in ([1.0, -5.0, 3.0], [0.0, 0.0, 0.0], [0.05, 0.0, 0.05], [0.2, 0.9, 0.05]):
        point = np.array(outside)
        assert not workspace.contains(point)
        assert workspace.contains(workspace.clip(point))
    assert workspace.contains(workspace.center)


def test_workspace_clip_is_idempotent():
    workspace = WorkspaceConfig()
    once = workspace.clip(np.array([0.9, 0.4, 0.5]))
    assert np.allclose(workspace.clip(once), once)


def test_polar_conversion_round_trips():
    workspace = WorkspaceConfig()
    point = np.array([0.21, -0.08, 0.04])
    assert np.allclose(workspace.from_polar(*workspace.to_polar(point)), point)
