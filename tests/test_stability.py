"""The two ways teleoperation must hold still.

A stationary hand must produce a stationary arm: landmark noise, multiplied by
the position gain, must never leak into the command. And releasing the clutch,
moving the hand anywhere at all, and re-engaging must be seamless -- no jump,
no lunge, no wrist snap. Both are pinned here at the mapper level, where the
commands are made.
"""

import numpy as np
import pytest

from handrobot.config import Config
from handrobot.hands.types import HandPose, Landmarks
from handrobot.retarget.mapper import HandToGripper, wrap_to_half_turn

FPS = 30.0
DT = 1.0 / FPS


def make_pose(position, t, yaw: float = 0.0, pinch: float = 0.08) -> HandPose:
    """A synthetic hand at a camera-frame position, knuckle line at ``yaw``."""
    position = np.asarray(position, dtype=float)
    # Columns are (pointing, up, knuckles); only the knuckle column matters
    # here. Yaw rotates it within the camera's right/forward plane.
    knuckles = np.array([np.sin(yaw), 0.0, np.cos(yaw)])
    pointing = np.array([np.cos(yaw), 0.0, -np.sin(yaw)])
    rotation = np.stack([pointing, np.array([0.0, 1.0, 0.0]), knuckles], axis=1)
    landmarks = Landmarks(
        image=np.zeros((21, 3)), world=np.zeros((21, 3)),
        handedness="Right", score=1.0,
    )
    return HandPose(
        palm_position=position, rotation=rotation, pinch_distance=pinch,
        depth=float(position[2]), landmarks=landmarks, timestamp=t,
    )


@pytest.fixture
def mapper() -> HandToGripper:
    config = Config(robot="panda")
    return HandToGripper(config.hand, config.workspace, config.gripper)


def centre(mapper: HandToGripper) -> np.ndarray:
    return np.asarray(mapper.workspace.center, dtype=float)


def run(mapper, poses, start):
    commands = []
    for pose in poses:
        commands.append(mapper(pose, start).position)
    return np.asarray(commands)


def still_poses(rng, base, start_t, n, noise=0.003):
    return [
        make_pose(base + rng.normal(0, noise, 3), start_t + i * DT)
        for i in range(n)
    ]


def test_a_still_hand_commands_a_still_arm(mapper):
    """Realistic landmark noise on a stationary hand: the deadband must hold
    the command still to well under what an operator can perceive."""
    rng = np.random.default_rng(0)
    base = np.array([0.0, 0.0, 0.45])
    start = centre(mapper)
    mapper.engage(make_pose(base, 0.0), start)

    commands = run(mapper, still_poses(rng, base, DT, 240), start)
    # The first seconds may contain the slow, deliberate drain of any parked
    # error -- a couple of millimetres over a couple of seconds. After that,
    # nothing may move.
    settled = commands[150:]
    wander = np.linalg.norm(settled - settled[-1], axis=1)
    assert wander.max() < 0.001, (
        f"arm wandered {wander.max() * 1000:.2f} mm from a stationary hand"
    )
    drained = np.linalg.norm(commands[45:] - commands[-1], axis=1)
    assert drained.max() < 0.005, (
        f"settling correction was {drained.max() * 1000:.2f} mm -- too visible"
    )


def test_a_deliberate_move_still_gets_through(mapper):
    """Parking must not eat real movement: a 6 cm reach lands as a reach."""
    rng = np.random.default_rng(1)
    base = np.array([0.0, 0.0, 0.45])
    start = centre(mapper)
    mapper.engage(make_pose(base, 0.0), start)
    t = DT
    for pose in still_poses(rng, base, t, 45):     # park first
        mapper(pose, start)
    t += 45 * DT

    moved = base + np.array([0.06, 0.0, 0.0])       # 6 cm to camera-right
    poses = [
        make_pose(base + (moved - base) * min(1.0, i / 20), t + i * DT)
        for i in range(60)
    ]
    commands = run(mapper, poses, start)
    travelled = commands[-1] - commands[0]
    # Camera-right is robot +y; the gain scales the reach.
    expected = 0.06 * mapper.position_gain
    assert travelled[1] > 0.75 * expected
    assert abs(travelled[0]) < 0.01 and abs(travelled[2]) < 0.01


def test_reengaging_somewhere_else_entirely_is_seamless(mapper):
    """Clutch out, hand teleports 30 cm, clutch in: the arm must not move more
    than the glide limit per frame, and must settle exactly where it was held."""
    rng = np.random.default_rng(2)
    base = np.array([0.0, 0.0, 0.45])
    start = centre(mapper)
    mapper.engage(make_pose(base, 0.0), start)
    t = DT
    for pose in still_poses(rng, base, t, 30):
        held = mapper(pose, start).position
    t += 30 * DT

    mapper.disengage()
    elsewhere = base + np.array([0.30, 0.05, -0.10])
    for i, pose in enumerate(still_poses(rng, elsewhere, t, 30)):
        mapper(pose, start)                          # filter keeps tracking
    t += 30 * DT

    mapper.engage(make_pose(elsewhere, t), held)
    commands = run(mapper, still_poses(rng, elsewhere, t + DT, 60), held)

    steps = np.linalg.norm(np.diff(commands, axis=0), axis=1)
    limit = mapper.hand.max_command_speed * DT + 1e-9
    assert steps.max() <= limit, (
        f"re-engage caused a {steps.max() * 1000:.1f} mm frame step"
    )
    # The residual is bounded by estimation noise, not by the 30 cm moved:
    # two noise-averaged anchors a few millimetres apart, times the gain.
    drift = np.linalg.norm(commands[-1] - held)
    assert drift < 0.006, (
        f"arm drifted {drift * 1000:.1f} mm after re-engaging with a still hand"
    )


def test_reengaging_with_a_turned_wrist_does_not_snap_the_jaw(mapper):
    """The angle anchors like the position: a wrist turned while the clutch was
    out must not turn the jaw on re-engage -- only turning it afterwards does."""
    base = np.array([0.0, 0.0, 0.45])
    start = centre(mapper)
    mapper.engage(make_pose(base, 0.0), start)
    for i in range(30):
        held_azimuth = mapper(make_pose(base, (i + 1) * DT), start).jaw_azimuth

    mapper.disengage()
    t = 31 * DT
    turned = 0.9                                     # ~50 degrees while out
    mapper.engage(make_pose(base, t, yaw=turned), start)
    resumed = mapper(make_pose(base, t + DT, yaw=turned), start).jaw_azimuth
    assert abs(wrap_to_half_turn(resumed - held_azimuth)) < 0.05, (
        "jaw snapped to the absolute wrist angle on re-engage"
    )

    # Turning the wrist afterwards must still turn the jaw, same sense.
    azimuth = resumed
    for i in range(60):
        azimuth = mapper(
            make_pose(base, t + (i + 2) * DT, yaw=turned + 0.4), start
        ).jaw_azimuth
    assert wrap_to_half_turn(azimuth - resumed) > 0.2


def test_noise_alone_never_escapes_the_deadband(mapper):
    """Twenty seconds of 3 mm landmark noise: the command must stay within
    perception the whole time -- no occasional escape, no slow random walk."""
    rng = np.random.default_rng(3)
    base = np.array([0.0, 0.0, 0.45])
    start = centre(mapper)
    mapper.engage(make_pose(base, 0.0), start)
    commands = run(mapper, still_poses(rng, base, DT, 600), start)
    settled = commands[150:]
    assert np.linalg.norm(settled - settled[-1], axis=1).max() < 0.001


def test_continuous_wrist_rotation_never_leaves_the_half_turn_branch():
    """Rotating the wrist steadily through several half turns: the filtered
    jaw azimuth must always come back on the near branch, (-pi/2, pi/2]."""
    from handrobot.filters import AngleFilter

    f = AngleFilter(min_cutoff=1.0, beta=0.3, half_turn_symmetric=True)
    for i in range(400):                      # ~4 full turns of input
        angle = wrap_to_half_turn(i * 0.06)
        out = f(angle, DT)
        assert -np.pi / 2 < out <= np.pi / 2 + 1e-12, (
            f"azimuth {out:.3f} escaped the half-turn branch at step {i}"
        )


def test_degenerate_landmarks_are_refused_not_zeroed():
    """A tracker glitch that collapses the landmarks must raise, so record and
    live skip the frame instead of storing or commanding a garbage pose."""
    from handrobot.dexhand.synth import landmarks_to_keypoints

    with pytest.raises(ValueError, match="degenerate"):
        landmarks_to_keypoints(np.zeros((21, 3)), "Right")
    collinear = np.outer(np.arange(21), np.array([1.0, 0.0, 0.0])) * 0.01
    with pytest.raises(ValueError, match="degenerate"):
        landmarks_to_keypoints(collinear, "Left")
