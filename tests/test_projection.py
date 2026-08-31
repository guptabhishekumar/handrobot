"""The pinhole projection, checked against the renderer itself.

Guidance arrows drawn from a wrong projection would point somewhere the goal is
not, which is worse than no arrows. So the projection is verified the only way
that cannot lie: render a probe at a known world position, find its pixels, and
compare with where the maths says it should be.
"""

import numpy as np
import pytest

from handrobot.config import Config
from handrobot.sim.env import PickPlaceEnv
from handrobot.viz.project import project_point

TOLERANCE_PX = 10.0


@pytest.fixture
def probe_env():
    """A scene whose goal marker is turned into an ideal projection probe.

    Ideal means: opaque (the shipped marker is translucent, and blending with
    the arm's shadows drags a measured centroid around), flat (a tall solid's
    silhouette centroid is not its centre off-axis), and small. The model is
    edited in place, which is fine for a throwaway test instance.
    """
    import mujoco

    config = Config(robot="panda")
    env = PickPlaceEnv(config=config, render_cameras=(), seed=0)
    env.reset(seed=0)
    env.data.qpos[env._bin_qpos_adr : env._bin_qpos_adr + 3] = [-3.0, -3.0, 3.0]
    env.data.qpos[env._cube_qpos_adr : env._cube_qpos_adr + 3] = [3.0, 3.0, 3.0]
    env.data.qpos[:7] = [0, -1.2, 0, -2.6, 0, 1.5, -0.785]

    ring = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_ring")
    pip = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_pip")
    env.model.geom_rgba[ring] = (0, 0, 0, 0)          # hide the wide ring
    env.model.geom_rgba[pip] = (0, 1, 0, 1)           # opaque, saturated
    env.model.geom_size[pip][:2] = (0.010, 0.0015)    # small and flat
    yield env
    env.close()


def rendered_centroid(env, camera, position, size=280):
    """Where the goal ring lands on screen.

    The ring, not the puck: a 6 mm-thick disc has next to no silhouette bias,
    whereas a 60 mm puck viewed obliquely shows its side wall and its centroid
    drifts off its true centre by tens of pixels -- a property of the probe,
    not of the projection being tested.
    """
    import mujoco

    env.set_goal_marker(position)
    mujoco.mj_forward(env.model, env.data)
    image = env.render(camera, size, size).astype(float)
    green = (
        (image[:, :, 1] > image[:, :, 0] * 1.12)
        & (image[:, :, 1] > image[:, :, 2] * 1.12)
    )
    if green.sum() < 10:
        return None
    ys, xs = np.nonzero(green)
    return float(xs.mean()), float(ys.mean())


@pytest.mark.parametrize("camera", ["front_cam", "top_cam", "hero_cam"])
def test_projection_matches_the_renderer_at_the_centre(probe_env, camera):
    """Absolute check where the probe's silhouette is unbiased."""
    size = 280
    position = np.array([0.50, 0.00, 0.01])
    measured = rendered_centroid(probe_env, camera, position, size)
    if measured is None:
        pytest.skip(f"probe not visible in {camera}")
    predicted = project_point(probe_env, camera, position, size, size)
    assert predicted is not None
    error = np.hypot(predicted[0] - measured[0], predicted[1] - measured[1])
    assert error < TOLERANCE_PX, f"{camera}: projection off by {error:.1f}px"


@pytest.mark.parametrize("camera", ["front_cam", "top_cam", "hero_cam"])
def test_projected_motion_matches_rendered_motion(probe_env, camera):
    """Differential check across the workspace.

    The probe is a solid, so its silhouette centroid sits off its true centre
    when viewed obliquely -- a property of the probe, not of the projection.
    Comparing *displacements* between two positions cancels that bias, so this
    can be strict everywhere the arrows will actually be drawn.
    """
    size = 280
    a, b = np.array([0.44, 0.15, 0.01]), np.array([0.56, -0.18, 0.01])
    measured_a = rendered_centroid(probe_env, camera, a, size)
    measured_b = rendered_centroid(probe_env, camera, b, size)
    if measured_a is None or measured_b is None:
        pytest.skip(f"probe not visible in {camera}")
    predicted_a = project_point(probe_env, camera, a, size, size)
    predicted_b = project_point(probe_env, camera, b, size, size)
    du = (predicted_b[0] - predicted_a[0]) - (measured_b[0] - measured_a[0])
    dv = (predicted_b[1] - predicted_a[1]) - (measured_b[1] - measured_a[1])
    assert np.hypot(du, dv) < TOLERANCE_PX, (
        f"{camera}: projected motion off by {np.hypot(du, dv):.1f}px"
    )


def test_points_behind_the_camera_are_rejected(probe_env):
    import mujoco

    mujoco.mj_forward(probe_env.model, probe_env.data)
    behind = np.array([5.0, 0.0, 0.5])  # far past the front camera
    assert project_point(probe_env, "front_cam", behind, 280, 280) is None


def test_the_chase_camera_keeps_the_gripper_centred(probe_env):
    """The whole point of the chase view: wherever the arm goes, the gripper
    stays in the middle of the frame and can never be lost off-screen."""
    import mujoco

    for pose in ([0, -0.6, 0, -2.2, 0, 1.7, -0.785], [0.5, -0.9, 0.2, -1.9, 0, 1.6, -0.785]):
        probe_env.data.qpos[:7] = pose
        mujoco.mj_forward(probe_env.model, probe_env.data)
        tcp, _ = probe_env.gripper_pose
        predicted = project_point(probe_env, "chase_cam", tcp, 300, 620)
        assert predicted is not None
        assert abs(predicted[0] - 310) < 90, "gripper drifted from the chase view centre"
        assert abs(predicted[1] - 150) < 90
