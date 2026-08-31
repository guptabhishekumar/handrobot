"""The mapping from hand movement to arm movement, checked against pixels.

This is the one part of the system that cannot be got right by reasoning about
coordinate frames, because there are four of them -- the camera's, the mirrored
preview's, the robot's base, and the viewpoint the operator is actually looking
at -- and an error in any one inverts an axis. It happened: left and right were
swapped, and so were push and pull, and no amount of reading the matrix revealed
it.

So the direction each robot axis appears to move on screen is *measured*, by
rendering a probe at two positions and comparing where it lands, and the mapping
is asserted against that. A future change that flips an axis fails here rather
than in someone's hands.
"""

import numpy as np
import pytest

from handrobot.config import Config
from handrobot.retarget.mapper import HandToGripper
from handrobot.sim.env import PickPlaceEnv

#: Pixels of movement below which a direction is called "no change".
THRESHOLD = 10.0


def probe_centroid(env, camera: str, position, size: int = 260):
    """Where the red object lands on screen, in pixels, at a given place."""
    import mujoco

    address = env._cube_qpos_adr
    env.data.qpos[address : address + 3] = position
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    image = env.render(camera, size, size).astype(float)
    red = (
        (image[:, :, 0] > image[:, :, 1] * 1.6)
        & (image[:, :, 0] > image[:, :, 2] * 1.6)
        & (image[:, :, 0] > 90)
    )
    if red.sum() < 10:
        return None
    ys, xs = np.nonzero(red)
    return float(xs.mean()), float(ys.mean())


def screen_shift(env, camera: str, axis: int, base, delta: float = 0.06):
    """How the probe moves on screen when it moves along one robot axis."""
    step = np.zeros(3)
    step[axis] = delta
    before = probe_centroid(env, camera, base - step)
    after = probe_centroid(env, camera, base + step)
    if before is None or after is None:
        pytest.skip(f"probe not visible in {camera}")
    return after[0] - before[0], after[1] - before[1]


@pytest.fixture
def clear_env():
    """A Panda scene with the bin removed and the arm tucked out of the way."""
    config = Config(robot="panda")
    env = PickPlaceEnv(config=config, render_cameras=(), seed=0)
    env.reset(seed=0)
    env.data.qpos[env._bin_qpos_adr : env._bin_qpos_adr + 3] = [-3.0, -3.0, 3.0]
    env.data.qpos[:7] = [0, -1.2, 0, -2.6, 0, 1.5, -0.785]
    yield env
    env.close()


def test_the_robots_y_axis_appears_to_the_right(clear_env):
    for camera in ("front_cam", "top_cam"):
        du, _ = screen_shift(clear_env, camera, 1, np.array([0.50, 0.0, 0.035]))
        assert du > THRESHOLD, f"robot +y does not appear rightwards in {camera}"


def test_the_robots_z_axis_appears_upwards(clear_env):
    _, dv = screen_shift(clear_env, "front_cam", 2, np.array([0.50, 0.0, 0.12]), delta=0.05)
    assert dv < -THRESHOLD, "robot +z does not appear upwards in the front view"


def test_the_robots_x_axis_appears_away_from_the_operator(clear_env):
    """Both views agree that +x is towards the viewer, so away is -x."""
    _, front = screen_shift(clear_env, "front_cam", 0, np.array([0.50, 0.0, 0.035]))
    _, top = screen_shift(clear_env, "top_cam", 0, np.array([0.50, 0.0, 0.035]))
    assert front > THRESHOLD, "robot +x should come towards the viewer in the front view"
    assert top > THRESHOLD, "robot +x should be downwards in the top view"


def test_moving_the_hand_right_moves_the_arm_right_on_screen(clear_env):
    """The bug a person notices within five seconds of trying it."""
    du, _ = screen_shift(clear_env, "front_cam", 1, np.array([0.50, 0.0, 0.035]))
    rightwards_axis = 1 if du > 0 else -1

    hand_right = np.array([1.0, 0.0, 0.0])          # camera x, on a mirrored preview
    robot = HandToGripper.CAMERA_TO_ROBOT @ hand_right
    assert robot[1] * rightwards_axis > 0.5, (
        "moving the hand right must move the gripper right on screen"
    )


def test_moving_the_hand_down_lowers_the_arm(clear_env):
    _, dv = screen_shift(clear_env, "front_cam", 2, np.array([0.50, 0.0, 0.12]), delta=0.05)
    upwards_axis = 1 if dv < 0 else -1

    hand_down = np.array([0.0, 1.0, 0.0])
    robot = HandToGripper.CAMERA_TO_ROBOT @ hand_down
    assert robot[2] * upwards_axis < -0.5, "moving the hand down must lower the gripper"


def test_pushing_the_hand_away_moves_the_arm_away(clear_env):
    """Pushing away from yourself is moving TOWARDS the camera on the desk, so
    it is camera-backward: negative z. The confusion of these two signs was a
    live inversion an operator found within seconds."""
    _, top = screen_shift(clear_env, "top_cam", 0, np.array([0.50, 0.0, 0.035]))
    away_axis = -1 if top > 0 else 1   # +x is downwards on the top view, so away is -x

    hand_push = np.array([0.0, 0.0, -1.0])   # towards the camera = away from self
    robot = HandToGripper.CAMERA_TO_ROBOT @ hand_push
    assert robot[0] * away_axis > 0.5, "pushing the hand away must move the gripper away"


def test_the_mapping_is_the_mirror_the_selfie_view_demands():
    """The map must be orthonormal with determinant -1 -- a mirror.

    The camera faces the operator (one reversal of rotation sense) and the
    preview is mirrored (a second). Two reversals cancel, so only a mirror map
    keeps the operator's clockwise as the robot's clockwise while also keeping
    all three translations natural. A proper rotation here can fix left/right
    or push/pull, never both, and inverts perceived rotation -- which is
    precisely what an operator reported before this was understood.
    """
    matrix = HandToGripper.CAMERA_TO_ROBOT
    assert np.allclose(matrix.T @ matrix, np.eye(3))
    assert np.isclose(np.linalg.det(matrix), -1.0)


def test_the_operators_clockwise_is_the_robots_clockwise():
    """User-local axes: right R, up U, away-from-self A. In mirrored-preview
    camera coordinates those are R=(1,0,0), U=(0,-1,0), A=(0,0,-1). A clockwise
    turn seen from above carries a direction from A towards R; the robot must
    turn the same way about its vertical axis, which means the rotation axis
    from mapped-A to mapped-R points along -z."""
    M = HandToGripper.CAMERA_TO_ROBOT
    away = M @ np.array([0.0, 0.0, -1.0])
    right = M @ np.array([1.0, 0.0, 0.0])
    axis = np.cross(away, right)
    assert axis[2] < -0.5, "a clockwise hand turn would appear anticlockwise"


def test_the_guidance_names_the_direction_it_means(clear_env):
    """'RIGHT 40' has to mean the operator's right, or it is worse than nothing."""
    mapper = HandToGripper()
    du, _ = screen_shift(clear_env, "front_cam", 1, np.array([0.50, 0.0, 0.035]))
    assert du > THRESHOLD

    # To move the gripper rightwards on screen (+y), the strip must say RIGHT,
    # which it does when the first component of the hand movement is positive.
    hand = mapper.hand_move_for(np.array([0.0, 0.05, 0.0]))
    assert hand[0] > 0, "the strip would say LEFT when it means right"

    # And to move it away, it must say PUSH: with camera-forward pointing at
    # the operator, that is a negative third component.
    hand = mapper.hand_move_for(np.array([-0.05, 0.0, 0.0]))
    assert hand[2] < 0, "the strip would say PULL when it means push"
