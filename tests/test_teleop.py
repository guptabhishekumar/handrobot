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


# -- the interface around the session ----------------------------------------


def test_interface_sizes_resolve_to_sixteen_by_nine():
    from handrobot.teleop import UI_PRESETS, window_size

    assert window_size(None) == UI_PRESETS["720p"]
    assert window_size("8k") == (7680, 4320)
    assert window_size("4K") == (3840, 2160)
    width, height = window_size(1440)
    assert (width, height) == (2560, 1440)
    for name in UI_PRESETS:
        w, h = window_size(name)
        assert abs(w / h - 16 / 9) < 1e-6
        assert w % 2 == 0 and h % 2 == 0


def test_an_unreadable_or_unknown_interface_size_is_refused():
    from handrobot.teleop import window_size

    with pytest.raises(ValueError):
        window_size("enormous")
    with pytest.raises(ValueError):
        window_size(200)


def test_renders_never_exceed_the_offscreen_buffer(session):
    """MuJoCo refuses to build a renderer larger than the model's framebuffer.

    An interface asking for 4K panels would therefore not be slow, it would
    raise on its first frame.
    """
    from handrobot.teleop import render_size

    limit_h, limit_w = session.env.max_render_size
    width, height = render_size(3776, 2100, (limit_h, limit_w))
    assert width <= limit_w and height <= limit_h
    assert abs((width / height) - (3776 / 2100)) < 0.02, "the panel was cropped, not scaled"
    # And the environment itself clamps, so asking too much is never fatal.
    image = session.env.render("front_cam", 4000, 4000)
    assert image.shape[0] <= limit_h and image.shape[1] <= limit_w


def test_the_render_cadence_follows_the_measured_loop():
    from handrobot.teleop import RenderPacer

    pacer = RenderPacer(budget_ms=33.0, settle=5)
    for _ in range(200):
        pacer.update(60.0)
    assert pacer.cadence > 1, "an overrunning loop kept redrawing every frame"

    for _ in range(400):
        pacer.update(8.0)
    assert pacer.cadence == 1, "a loop with headroom never returned to every frame"


def test_the_render_cadence_cannot_oscillate():
    from handrobot.teleop import RenderPacer

    pacer = RenderPacer(budget_ms=33.0, settle=15)
    changes = 0
    previous = pacer.cadence
    for i in range(300):
        # Right on the budget, alternating either side of it.
        pacer.update(33.0 + (6.0 if i % 2 else -6.0))
        changes += pacer.cadence != previous
        previous = pacer.cadence
    assert changes <= 2, "the cadence chattered around the budget"


def test_the_tracking_readout_reflects_now_not_the_whole_session():
    """A session average cannot move after ten good minutes.

    Which is exactly when an operator most needs to be told that tracking has
    just collapsed.
    """
    from handrobot.teleop import TeleopStats

    stats = TeleopStats()
    for _ in range(1000):
        stats.frames_seen += 1
        stats.note_frame(True, None)
    for _ in range(90):
        stats.frames_seen += 1
        stats.note_frame(False, "hand too close to the camera")

    assert stats.tracking_rate > 0.9, "the session total should still look healthy"
    assert stats.live_tracking_rate == 0.0
    assert stats.live_rejection == "hand too close to the camera"


def test_a_healthy_recent_window_says_nothing():
    from handrobot.teleop import TeleopStats

    stats = TeleopStats()
    for i in range(90):
        stats.frames_seen += 1
        stats.note_frame(i % 20 != 0, "no hand in frame")
    assert stats.live_tracking_rate > 0.9
    assert stats.live_rejection is None, "normal frame loss must not nag"


def test_a_saved_success_raises_the_banner(session):
    session.mapper._engaged = True
    session.step(make_pose())
    assert session.flash == 0.0
    session.save_episode(success=True)
    assert session.flash == 1.0


def test_a_saved_failure_raises_nothing(session):
    session.mapper._engaged = True
    session.step(make_pose())
    session.save_episode(success=False)
    assert session.flash == 0.0


def test_a_message_stops_being_shown_once_it_is_stale(session, monkeypatch):
    """Messages report what just happened. One still on screen a minute later
    is describing a different moment, and nothing says which."""
    import handrobot.teleop as teleop

    clock = {"now": 1000.0}
    monkeypatch.setattr(teleop.time, "perf_counter", lambda: clock["now"])

    session.message = "clutch engaged"
    assert session.visible_message == "clutch engaged"
    clock["now"] += session.MESSAGE_SECONDS + 1
    assert session.visible_message == ""
    assert session.message == "clutch engaged", "the message itself is still readable"


def test_the_timeline_records_why_each_frame_was_poor():
    from handrobot.teleop import TeleopStats

    stats = TeleopStats()
    stats.note_frame(True, None)
    stats.note_frame(True, None, clipped=True)
    stats.note_frame(False, "no hand in frame")
    assert list(stats.recent_quality) == [1, 2, 0]
