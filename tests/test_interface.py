"""The whole operator interface, driven without a webcam or a window.

The loop owns the two things that need hardware -- a camera and a window --
and :class:`~handrobot.teleop.Interface` owns the pixels, so everything the
operator actually looks at can be composed here from synthetic hand poses. A
drawing bug that only appears with a camera attached is a drawing bug that
ships.
"""

import time

import numpy as np
import pytest

from handrobot.config import Config
from handrobot.hands.geometry import CameraIntrinsics
from handrobot.hands.types import HandPose, Landmarks
from handrobot.retarget.ik import ArmIK
from handrobot.retarget.mapper import HandToGripper
from handrobot.sim.env import PickPlaceEnv
from handrobot.teleop import Interface, TeleopSession

CAMERA_SIZE = (640, 480)


class StubTracker:
    """Only the three things the interface reads off the tracker."""

    def __init__(self) -> None:
        self.intrinsics = CameraIntrinsics.from_hfov(*CAMERA_SIZE, 62.0)
        self.hands_seen = 1
        self.followed_hand = "right"


def make_landmarks() -> Landmarks:
    """A hand roughly in the middle of the frame, at a plausible scale."""
    rng = np.random.default_rng(0)
    image = np.column_stack([
        rng.uniform(0.35, 0.65, 21),
        rng.uniform(0.35, 0.65, 21),
        np.zeros(21),
    ])
    world = rng.uniform(-0.05, 0.05, (21, 3))
    return Landmarks(image=image, world=world, handedness="Left", score=0.95)


def make_pose(offset=(0.0, 0.0, 0.0), pinch=0.05) -> HandPose:
    return HandPose(
        palm_position=np.array([0.0, 0.0, 0.55]) + np.asarray(offset, dtype=float),
        rotation=np.eye(3),
        pinch_distance=pinch,
        depth=0.55,
        landmarks=make_landmarks(),
        timestamp=0.0,
    )


@pytest.fixture
def rig():
    config = Config(robot="panda")
    env = PickPlaceEnv(config=config, seed=0)
    mapper = HandToGripper(config.hand, config.workspace, config.gripper)
    session = TeleopSession(env, mapper, ArmIK(config.ik, config.spec), config)
    session.new_episode(seed=1)
    yield config, env, mapper, session
    env.close()


def frame_of(rig, ui="720p", engaged=False, pose=None, landmarks=None, interface=None,
             frames=1):
    config, env, mapper, session = rig
    interface = interface or Interface(config, env, mapper, CAMERA_SIZE, ui=ui)
    tracker = StubTracker()
    webcam = np.full((CAMERA_SIZE[1], CAMERA_SIZE[0], 3), 90, np.uint8)
    if engaged and pose is not None:
        mapper.engage(pose, env.gripper_pose[0])
    out = None
    for _ in range(frames):
        info = session.step(pose)
        out = interface.render(session, webcam, pose, landmarks, info, tracker, 0.033)
    return interface, out


@pytest.mark.parametrize("ui,expected", [
    ("720p", (720, 1280)),
    ("1080p", (1080, 1920)),
    ("4k", (2160, 3840)),
])
def test_the_interface_composes_at_every_size(rig, ui, expected):
    _, frame = frame_of(rig, ui=ui, pose=make_pose(), landmarks=make_landmarks())
    assert frame.shape == (*expected, 3)


def test_panels_never_ask_for_more_than_the_offscreen_buffer(rig):
    config, env, mapper, _ = rig
    limit_h, limit_w = env.max_render_size
    interface = Interface(config, env, mapper, CAMERA_SIZE, ui="8k")
    for width, height in (interface.stage_render, interface.tile_render,
                          interface.full_render):
        assert width <= limit_w and height <= limit_h


def test_a_lost_hand_still_composes(rig):
    _, frame = frame_of(rig, pose=None, landmarks=None)
    assert frame.shape == (720, 1280, 3)


def test_a_seen_but_unresolved_hand_still_composes(rig):
    _, frame = frame_of(rig, pose=None, landmarks=make_landmarks())
    assert frame.shape == (720, 1280, 3)


def test_the_reach_outline_appears_once_the_clutch_is_anchored(rig):
    pose, landmarks = make_pose(), make_landmarks()
    _, released = frame_of(rig, pose=pose, landmarks=landmarks)
    interface, engaged = frame_of(rig, pose=pose, landmarks=landmarks, engaged=True)
    assert not np.array_equal(released, engaged)

    config, env, mapper, _ = rig
    polygons = interface.envelope.polygons(
        mapper.hand_anchor, mapper.robot_anchor, mapper.position_gain,
        float(mapper.command_position[0]),
    )
    assert polygons, "no reachable region was traced for the live mapping"
    # And it lands inside the camera panel rather than off the side of it.
    points = np.concatenate(polygons)
    assert points[:, 0].min() < CAMERA_SIZE[0] and points[:, 0].max() > 0


def test_switching_the_stage_changes_the_picture(rig):
    pose, landmarks = make_pose(), make_landmarks()
    interface, first = frame_of(rig, pose=pose, landmarks=landmarks)
    _, _, _, session = rig
    assert interface.stage == "camera"
    assert interface.handle_key(ord("v"), session)
    assert interface.stage in ("chase_cam", "front_cam", "hero_cam", "wrist_cam")
    assert "camera" not in interface.tile_names or interface.stage != "camera"
    _, second = frame_of(rig, pose=pose, landmarks=landmarks, interface=interface)
    assert not np.array_equal(first, second)
    assert "stage" in session.message


def test_the_column_can_be_dropped_for_the_whole_window(rig):
    pose, landmarks = make_pose(), make_landmarks()
    interface, with_tiles = frame_of(rig, pose=pose, landmarks=landmarks)
    _, _, _, session = rig
    assert interface.handle_key(ord("t"), session)
    assert interface.tile_names == []
    _, full = frame_of(rig, pose=pose, landmarks=landmarks, interface=interface)
    assert not np.array_equal(with_tiles, full)
    assert interface.preview_size()[0] > 0.9 * interface.width


def test_the_wrist_inset_can_be_turned_off(rig):
    pose, landmarks = make_pose(), make_landmarks()
    interface, with_inset = frame_of(rig, pose=pose, landmarks=landmarks)
    _, _, _, session = rig
    assert interface.handle_key(ord("w"), session)
    _, without = frame_of(rig, pose=pose, landmarks=landmarks, interface=interface)
    assert not np.array_equal(with_inset, without)


def test_a_view_on_the_stage_is_not_also_a_tile(rig):
    config, env, mapper, session = rig
    interface = Interface(config, env, mapper, CAMERA_SIZE)
    interface.handle_key(ord("w"), session)
    assert interface.stage == "wrist_cam"
    assert "wrist_cam" not in interface.tile_names, "the wrist was shown twice"
    frame_of(rig, pose=make_pose(), landmarks=make_landmarks(), interface=interface)


def test_help_covers_the_camera_panel_and_goes_away(rig):
    pose, landmarks = make_pose(), make_landmarks()
    interface, plain = frame_of(rig, pose=pose, landmarks=landmarks)
    _, _, _, session = rig
    assert interface.handle_key(ord("?"), session)
    _, helped = frame_of(rig, pose=pose, landmarks=landmarks, interface=interface)
    assert not np.array_equal(plain, helped)
    assert interface.handle_key(ord("/"), session)
    _, back = frame_of(rig, pose=pose, landmarks=landmarks, interface=interface)
    assert np.array_equal(plain[:592, :600], back[:592, :600])


def test_keys_the_interface_does_not_own_are_left_for_the_session(rig):
    _, _, _, session = rig
    config, env, mapper, _ = rig
    interface = Interface(config, env, mapper, CAMERA_SIZE)
    for key in (ord(" "), ord("n"), ord("s"), ord("d"), ord("h"), ord("["), ord("]")):
        assert not interface.handle_key(key, session)


def test_panels_are_held_between_refreshes_rather_than_redrawn(rig):
    """Rendering is the only cost in the loop that can be spent selectively, and
    the operator cannot see a panel held for one extra frame -- but they can
    feel a control period that overran."""
    config, env, mapper, session = rig
    interface = Interface(config, env, mapper, CAMERA_SIZE)
    interface.pacer.cadence = 3

    renders = {"n": 0}
    original = env.render

    def counted(*args, **kwargs):
        renders["n"] += 1
        return original(*args, **kwargs)

    env.render = counted
    pose, landmarks = make_pose(), make_landmarks()
    frame_of(rig, pose=pose, landmarks=landmarks, interface=interface, frames=6)
    env.render = original

    # Six frames, a refresh every third: two refreshes of three cameras.
    assert renders["n"] == 6, f"panels were redrawn {renders['n']} times in six frames"


def test_one_interface_frame_fits_inside_a_control_period(rig):
    """Not a benchmark -- a regression guard, and a deliberately loose one.

    The bound has to hold on a CI runner rendering MuJoCo in software, which is
    an order of magnitude slower than the laptop this is meant for: it is here
    to catch a change that makes the interface ten times more expensive, not to
    measure it.
    """
    pose, landmarks = make_pose(), make_landmarks()
    interface, _ = frame_of(rig, pose=pose, landmarks=landmarks)
    _, _, _, session = rig
    tracker = StubTracker()
    webcam = np.full((CAMERA_SIZE[1], CAMERA_SIZE[0], 3), 90, np.uint8)

    start = time.perf_counter()
    for _ in range(10):
        info = session.step(pose)
        interface.render(session, webcam, pose, landmarks, info, tracker, 0.033)
    each = (time.perf_counter() - start) / 10
    assert each < 0.75, f"one interface frame took {each * 1000:.0f} ms"


def test_nothing_is_drawn_under_the_status_ribbon(rig):
    """The ribbon is composited last. Anything the overlays put in those rows
    is drawn where the operator cannot see it, which is how an outline ends up
    looking like it stops halfway."""
    config, env, mapper, session = rig
    interface = Interface(config, env, mapper, CAMERA_SIZE, ui="1080p")
    width, height = interface.preview_size()
    inset = interface.ribbon_inset(height)
    assert inset > 0

    pose, landmarks = make_pose(), make_landmarks()
    mapper.engage(pose, env.gripper_pose[0])
    webcam = np.zeros((CAMERA_SIZE[1], CAMERA_SIZE[0], 3), np.uint8)
    preview = interface._preview(
        webcam, landmarks, pose, False, StubTracker().intrinsics,
        session.step(pose)["command"], True, np.array([0.4, 0.4, 0.0]),
    )
    assert preview[height - inset :].max() == 0, "an overlay was drawn under the ribbon"
