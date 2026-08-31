"""Teleoperation session logic, exercised without a camera.

The webcam and MediaPipe are the two parts that cannot run in a test, so the
session takes hand poses as an argument and everything below that line is
testable. These cover the behaviours that are easy to get subtly wrong and hard
to notice while operating: the clutch, homing, and what lands in the dataset.
"""

import numpy as np
import pytest

from handrobot.config import Config
from handrobot.data.dataset import EpisodeWriter, list_episodes, load_episode
from handrobot.hands.types import HandPose, Landmarks
from handrobot.retarget.ik import ArmIK
from handrobot.retarget.mapper import HandToGripper
from handrobot.sim.env import PickPlaceEnv
from handrobot.teleop import OPERATOR_CAMERA, OPERATOR_HEIGHT, OPERATOR_WIDTH, TeleopSession


def make_pose(offset=(0.0, 0.0, 0.0), pinch=0.05) -> HandPose:
    landmarks = Landmarks(
        image=np.zeros((21, 3)), world=np.zeros((21, 3)), handedness="Right", score=1.0
    )
    return HandPose(
        palm_position=np.array([0.0, 0.0, 0.5]) + np.asarray(offset, dtype=float),
        rotation=np.eye(3),
        pinch_distance=pinch,
        depth=0.5,
        landmarks=landmarks,
        timestamp=0.0,
    )


@pytest.fixture(params=["panda", "so101"])
def session(request, tmp_path):
    """A teleoperation session on each supported arm.

    Both are exercised: the code is written against a robot description, and a
    test that only ever ran on one of them would not prove that.
    """
    config = Config(robot=request.param)
    env = PickPlaceEnv(config=config, seed=0)
    policy_cameras = [c.name for c in config.sim.policy_cameras]
    writer = EpisodeWriter(
        tmp_path, policy_cameras + [OPERATOR_CAMERA], source="human",
        policy_cameras=policy_cameras,
    )
    s = TeleopSession(env, HandToGripper(config.hand, config.workspace, config.gripper),
                      ArmIK(config.ik, config.spec), config, writer)
    s.new_episode(seed=3)
    yield s
    env.close()


def run(session, pose, steps):
    for _ in range(steps):
        session.step(pose)
    return session.env.gripper_pose[0]


def test_arm_does_not_move_while_the_clutch_is_released(session):
    before = session.env.gripper_pose[0].copy()
    after = run(session, make_pose((0.10, 0.0, 0.0)), 40)
    # Not exactly zero: the arm settles a little under gravity while its
    # position servos hold the home pose. It must not follow the hand, though.
    assert np.linalg.norm(after - before) < 0.012


def test_arm_follows_the_hand_once_engaged(session):
    session.engage(make_pose())
    before = session.env.gripper_pose[0].copy()
    # Hand to the right is the robot's +y, which is rightwards on screen.
    after = run(session, make_pose((0.05, 0.0, 0.0)), 60)
    assert after[1] > before[1] + 0.01


def test_losing_tracking_freezes_the_arm(session):
    session.engage(make_pose())
    run(session, make_pose((0.04, 0.0, 0.0)), 60)
    held = session.env.gripper_pose[0].copy()
    after = run(session, None, 40)
    assert np.linalg.norm(after - held) < 5e-3


def test_pinch_drives_the_gripper(session):
    """Measured as the jaw gap, because the command is an angle on one arm and
    a tendon length on another."""
    left, right = session.spec.jaw_bodies
    from handrobot.gripper import _point_position

    def gap():
        return float(np.linalg.norm(
            _point_position(session.env.model, session.env.data, left)
            - _point_position(session.env.model, session.env.data, right)
        ))

    session.engage(make_pose(pinch=0.09))
    run(session, make_pose(pinch=0.09), 60)
    wide = gap()
    run(session, make_pose(pinch=0.01), 60)
    assert gap() < wide - 0.015, f"jaws went from {wide * 1000:.1f} to {gap() * 1000:.1f} mm"


def test_homing_actually_reaches_home(session):
    """Regression: homing used to be undone by the very next IK step."""
    session.engage(make_pose())
    run(session, make_pose((0.06, 0.0, -0.03)), 60)
    assert np.linalg.norm(
        session.env.joint_positions[: session.spec.n_arm_joints]
        - session.env._home_ctrl[: session.spec.n_arm_joints]
    ) > 0.2

    session.go_home()
    for _ in range(120):
        session.step(make_pose((0.06, 0.0, -0.03)))
        if not session._homing:
            break
    assert not session._homing, "homing never completed"
    assert np.max(np.abs(
        session.env.joint_positions[: session.spec.n_arm_joints]
        - session.env._home_ctrl[: session.spec.n_arm_joints]
    )) < 0.06


def test_engaging_cancels_homing(session):
    session.go_home()
    session.engage(make_pose())
    assert not session._homing
    assert session.mapper.engaged


def test_homing_is_not_recorded(session):
    session.engage(make_pose())
    for _ in range(6):
        session.step(make_pose((0.03, 0.0, 0.0)))
    before = session.episode_steps
    assert before > 0

    session.go_home()
    for _ in range(30):
        session.step(make_pose())
    assert session.episode_steps == before


def test_released_clutch_frames_are_not_recorded(session):
    """A pause while the operator repositions is not part of the demonstration."""
    session.engage(make_pose())
    for _ in range(6):
        session.step(make_pose((0.02, 0.0, 0.0)))
    engaged_steps = session.episode_steps

    session.disengage()
    for _ in range(20):
        session.step(make_pose((0.10, 0.0, 0.0)))
    assert session.episode_steps == engaged_steps


def test_clutch_survives_a_new_episode(session):
    session.engage(make_pose())
    session.new_episode(seed=4)
    assert session.mapper.engaged
    assert not session.mapper.anchored
    # The next tracked frame re-anchors rather than lurching.
    before = session.env.gripper_pose[0].copy()
    session.step(make_pose((0.5, 0.0, 0.0)))
    assert np.linalg.norm(session.env.gripper_pose[0] - before) < 0.02


def test_recorded_episode_has_the_operator_camera_but_excludes_it_from_policy_inputs(session, tmp_path):
    session.engage(make_pose())
    session.set_operator_frame(
        np.full((OPERATOR_HEIGHT, OPERATOR_WIDTH, 3), 120, dtype=np.uint8)
    )
    for _ in range(12):
        session.step(make_pose((0.02, 0.0, 0.0)))
    session.save_episode(success=False)

    episode = load_episode(list_episodes(tmp_path)[0])
    assert len(episode) == 12
    assert OPERATOR_CAMERA in episode.images
    assert OPERATOR_CAMERA not in episode.policy_cameras
    assert episode.images[OPERATOR_CAMERA][0].shape == (OPERATOR_HEIGHT, OPERATOR_WIDTH, 3)
    assert episode.metadata["seed"] == 3
    assert episode.source == "human"


def test_recording_stops_at_the_episode_step_limit(session):
    limit = session.config.sim.teleop_max_steps
    session.episode_steps = limit
    session.engage(make_pose())
    session.step(make_pose())
    assert not session.recording
    assert "limit" in session.message


def test_discard_leaves_nothing_on_disk(session, tmp_path):
    session.engage(make_pose())
    for _ in range(5):
        session.step(make_pose())
    session.discard_episode()
    assert list_episodes(tmp_path) == []
    assert session.stats.episodes_discarded == 1


def test_saving_an_empty_episode_is_refused(session):
    session.discard_episode()
    session.save_episode(success=False)
    assert session.stats.episodes_saved == 0
    assert "nothing" in session.message


def test_commanded_action_is_exactly_what_the_simulator_executes(session):
    """A recorded action that the simulator silently clips would poison training."""
    session.engage(make_pose())
    for offset in np.linspace(0.0, 0.25, 30):
        info = session.step(make_pose((offset, offset / 2, -offset / 2)))
        assert np.allclose(session.env.commanded_positions, session._q, atol=0.0)


def test_a_recorded_step_renders_each_camera_exactly_once(session, monkeypatch):
    """Regression: env.step used to render an observation teleop then discarded,
    which doubled the render cost and dropped the loop from 30 fps to 19."""
    calls = []
    original = session.env.render
    monkeypatch.setattr(session.env, "render", lambda camera, *a, **k: (
        calls.append(camera), original(camera, *a, **k))[1])

    session.engage(make_pose())
    session.step(make_pose((0.01, 0.0, 0.0)))
    assert sorted(calls) == sorted(c.name for c in session.config.sim.policy_cameras)


def test_an_unrecorded_step_renders_nothing(session, monkeypatch):
    calls = []
    original = session.env.render
    monkeypatch.setattr(session.env, "render", lambda camera, *a, **k: (
        calls.append(camera), original(camera, *a, **k))[1])

    session.disengage()
    session.step(make_pose())
    session.go_home()
    session.step(make_pose())
    assert calls == []


def test_alignment_readout_tracks_the_objects(session):
    """An operator aid, and only that: it must never reach a policy."""
    cube, bin_position, _ = session.spec.layout.sample(
        np.random.default_rng(0), session.spec.cube_half_extent
    )
    session.env.reset(seed=0, cube_position=cube, bin_position=bin_position)
    alignment = session.alignment()
    gripper, _ = session.env.gripper_pose
    assert alignment["cube_planar"] == pytest.approx(
        float(np.linalg.norm(gripper[:2] - session.env.cube_position[:2])), abs=1e-9
    )
    assert alignment["cube_height"] > 0, "the gripper starts above the table"
    assert alignment["bin_planar"] > 0


def test_alignment_is_not_part_of_any_recorded_observation(session, tmp_path):
    from handrobot.data.dataset import list_episodes, load_episode

    session.engage(make_pose())
    session.set_operator_frame(np.zeros((360, 480, 3), np.uint8))
    for _ in range(5):
        session.step(make_pose((0.01, 0.0, 0.0)))
    session.save_episode(success=False)

    episode = load_episode(list_episodes(tmp_path)[0])
    assert set(episode.images) <= {"front_cam", "wrist_cam", "operator_cam"}
    assert episode.states.shape[1] == len(session.spec.actuators), (
        "state is joint positions only"
    )


def test_the_teleop_episode_limit_is_far_longer_than_the_scoring_one(session):
    """A person lining a gripper up by hand is not on the same clock as a
    controller that already knows where everything is."""
    sim = session.config.sim
    assert sim.teleop_max_steps >= 4 * sim.max_episode_steps
    assert sim.teleop_max_steps / sim.control_hz > 60, "under a minute is too short"


def test_the_commanded_joints_never_exceed_the_speed_limit(session):
    """A hard ceiling on how violently the arm can react to anything upstream."""
    limit = session.rate_limiter.max_speed * session.config.sim.control_dt
    session.engage(make_pose())
    previous = session.env.commanded_positions.copy()
    for offset in np.linspace(0.0, 0.6, 60):
        # A wildly jumping hand, far faster than anyone could actually move.
        session.step(make_pose((offset, -offset, offset / 2)))
        commanded = session.env.commanded_positions
        assert np.all(np.abs(commanded - previous) <= limit + 1e-9), (
            f"joint jumped {np.max(np.abs(commanded - previous)):.4f} rad in one step"
        )
        previous = commanded.copy()


def test_the_rate_limiter_smooths_the_engage_transient(session):
    """Engaging swings the arm from its folded home pose into a top-down grip.
    That is a big joint move, and letting it happen in one control step is a
    visible snap. The limiter spreads it over a few tenths of a second."""
    session.engage(make_pose())
    for _ in range(30):
        session.step(make_pose())
    assert session.rate_limiter.clipped > 0, "the transient was not smoothed at all"


def test_the_rate_limiter_does_not_slow_ordinary_teleoperation(session):
    """Once settled, it must never bind: it only ever removes the extremes."""
    session.engage(make_pose())
    for _ in range(60):          # let the engage transient finish
        session.step(make_pose())

    session.rate_limiter.clipped = 0
    before = session.env.gripper_pose[0].copy()
    for i in range(60):
        session.step(make_pose((0.0006 * i, 0.0, 0.0)))
    assert session.rate_limiter.clipped == 0, "steady teleoperation is being clipped"
    assert abs(session.env.gripper_pose[0][1] - before[1]) > 0.005


def test_clipping_detection_flags_a_hand_at_the_frame_edge():
    from handrobot.hands.types import Landmarks
    from handrobot.viz.overlay import hand_is_clipped

    centred = np.tile(np.array([0.5, 0.5, 0.0]), (21, 1))
    assert not hand_is_clipped(Landmarks(centred, np.zeros((21, 3)), "Right", 1.0))

    at_edge = centred.copy()
    at_edge[4, :2] = [0.5, 0.005]
    assert hand_is_clipped(Landmarks(at_edge, np.zeros((21, 3)), "Right", 1.0))


def test_alignment_tells_the_operator_which_way_to_move(session):
    """A distance is not actionable; a direction is."""
    cube, bin_position, _ = session.spec.layout.sample(
        np.random.default_rng(0), session.spec.cube_half_extent
    )
    session.env.reset(seed=0, cube_position=cube, bin_position=bin_position)
    session.engage(make_pose())
    alignment = session.alignment()
    move = alignment["hand_move"]

    # The object spawns on the robot's +y side, which is the operator's right.
    assert move[0] > 0.005
    assert alignment["goal_name"] == "cube"

    # Following the instruction must actually close the gap.
    gripper, _ = session.env.gripper_pose
    moved = gripper + session.mapper.CAMERA_TO_ROBOT @ (move * session.mapper.position_gain)
    assert np.linalg.norm(moved - alignment["goal"]) < 1e-6


def test_the_goal_switches_to_the_bin_once_the_cube_is_lifted(session):
    cube, bin_position, _ = session.spec.layout.sample(
        np.random.default_rng(0), session.spec.cube_half_extent
    )
    session.env.reset(seed=0, cube_position=cube, bin_position=bin_position)
    assert session.alignment()["goal_name"] == "cube"

    address = session.env._cube_qpos_adr
    session.env.data.qpos[address : address + 3] = [
        cube[0], cube[1], 4 * session.spec.cube_half_extent
    ]
    session.env.data.qvel[:] = 0.0
    import mujoco

    mujoco.mj_forward(session.env.model, session.env.data)
    assert session.alignment()["goal_name"] == "bin"
