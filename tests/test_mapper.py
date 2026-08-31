import numpy as np
import pytest

from handrobot.config import GripperConfig, HandConfig, WorkspaceConfig
from handrobot.geometry import frame_from_axes
from handrobot.hands.types import HandPose, Landmarks
from handrobot.retarget.mapper import HandToGripper, wrap_to_pi

HOME = np.array([0.215, 0.0, 0.055])


def make_pose(position, rotation=None, pinch=0.05) -> HandPose:
    landmarks = Landmarks(
        image=np.zeros((21, 3)), world=np.zeros((21, 3)), handedness="Right", score=1.0
    )
    return HandPose(
        palm_position=np.asarray(position, dtype=float),
        rotation=np.eye(3) if rotation is None else rotation,
        pinch_distance=pinch,
        depth=0.5,
        landmarks=landmarks,
        timestamp=0.0,
    )


def settle(mapper, pose, steps=80):
    for _ in range(steps):
        command = mapper(pose, HOME)
    return command


@pytest.fixture
def mapper():
    return HandToGripper(HandConfig(), WorkspaceConfig(), GripperConfig())


def test_wrap_to_pi_covers_the_branch_cut():
    assert wrap_to_pi(0.0) == pytest.approx(0.0)
    assert wrap_to_pi(3 * np.pi) == pytest.approx(np.pi)
    assert wrap_to_pi(-3 * np.pi) == pytest.approx(np.pi)


def test_disengaged_mapper_holds_position(mapper):
    command = mapper(make_pose([0.0, 0.0, 0.5]), HOME)
    assert not command.engaged
    assert np.allclose(command.position, HOME)


def test_lost_tracking_holds_the_last_command(mapper):
    mapper.engage(make_pose([0.0, 0.0, 0.5]), HOME)
    held = settle(mapper, make_pose([0.02, 0.0, 0.5]))
    dropped = mapper(None, HOME)
    assert not dropped.engaged
    assert np.allclose(dropped.position, held.position)
    assert dropped.jaw_gap == pytest.approx(held.jaw_gap)


def test_camera_axes_map_onto_robot_axes(mapper):
    """Directions follow what the operator sees on screen; the measured
    convention lives in tests/test_screen_directions.py."""
    # Hand to the right, on a mirrored preview, is the robot's +y -- which is
    # rightwards in both simulator views.
    mapper.engage(make_pose([0.0, 0.0, 0.5]), HOME)
    assert settle(mapper, make_pose([0.04, 0.0, 0.5])).position[1] > HOME[1] + 0.01

    # The hand drifting away from the camera is the hand pulled towards the
    # operator, so the robot comes towards the viewer: +x.
    mapper.reset()
    mapper.engage(make_pose([0.0, 0.0, 0.5]), HOME)
    assert settle(mapper, make_pose([0.0, 0.0, 0.53])).position[0] > HOME[0] + 0.005

    # Hand downward is the robot's -z.
    mapper.reset()
    mapper.engage(make_pose([0.0, 0.0, 0.5]), HOME)
    assert settle(mapper, make_pose([0.0, 0.03, 0.5])).position[2] < HOME[2] - 0.01


def test_commands_stay_inside_the_workspace(mapper):
    workspace = WorkspaceConfig()
    mapper.engage(make_pose([0.0, 0.0, 0.5]), HOME)
    for offset in np.linspace(-2.0, 2.0, 41):
        command = mapper(make_pose([offset, offset, 0.5 + offset]), HOME)
        assert workspace.contains(command.position, tolerance=1e-6)


def test_re_engaging_re_anchors_without_moving_the_arm(mapper):
    mapper.engage(make_pose([0.0, 0.0, 0.5]), HOME)
    moved = settle(mapper, make_pose([0.02, 0.0, 0.5])).position.copy()

    mapper.disengage()
    # The operator moves their hand a long way with the clutch released. The
    # filter keeps tracking throughout, which is what makes re-engaging clean.
    for _ in range(60):
        mapper(make_pose([0.30, 0.0, 0.5]), moved)
    mapper.engage(make_pose([0.30, 0.0, 0.5]), moved)
    resumed = mapper(make_pose([0.30, 0.0, 0.5]), moved)
    assert np.allclose(resumed.position, moved, atol=2e-3)


def test_re_engaging_after_lost_tracking_does_not_drift(mapper):
    """With no poses at all while released, the filter holds a stale position.
    Anchoring to that would make the arm crawl as the filter caught up."""
    mapper.engage(make_pose([0.0, 0.0, 0.5]), HOME)
    moved = settle(mapper, make_pose([0.02, 0.0, 0.5])).position.copy()

    mapper.disengage()
    # No frames at all, then the hand reappears somewhere completely different.
    mapper.engage(make_pose([0.30, 0.0, 0.5]), moved)
    for _ in range(30):
        resumed = mapper(make_pose([0.30, 0.0, 0.5]), moved)
    assert np.allclose(resumed.position, moved, atol=2e-3)


def test_pinch_maps_monotonically_onto_the_jaw_gap(mapper):
    hand, gripper = HandConfig(), GripperConfig()
    gaps = [mapper.jaw_gap_from_pinch(p) for p in np.linspace(0.0, 0.15, 25)]
    assert all(b >= a - 1e-12 for a, b in zip(gaps, gaps[1:]))
    assert gaps[0] == pytest.approx(gripper.min_command_gap)
    assert gaps[-1] == pytest.approx(gripper.max_command_gap)
    assert mapper.jaw_gap_from_pinch(hand.pinch_closed_m) == pytest.approx(gripper.min_command_gap)
    assert mapper.jaw_gap_from_pinch(hand.pinch_open_m) == pytest.approx(gripper.max_command_gap)


def test_jaw_azimuth_follows_the_knuckle_direction(mapper):
    """A parallel jaw is symmetric, so only the axis matters, never its sign."""

    def axis_angle(azimuth):
        """Wrap to a half turn: opposite directions are the same jaw axis."""
        return wrap_to_pi(2 * azimuth) / 2

    # Knuckles along camera x, which is the robot's +y: jaws open across y.
    rotation = frame_from_axes(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    azimuth = mapper.jaw_azimuth_from_hand(make_pose([0, 0, 0.5], rotation))
    assert abs(axis_angle(azimuth - np.pi / 2)) < 1e-6

    # Knuckles along camera z, which is the robot's -x: jaws open across x.
    rotation = frame_from_axes(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    azimuth = mapper.jaw_azimuth_from_hand(make_pose([0, 0, 0.5], rotation))
    assert abs(axis_angle(azimuth - 0.0)) < 1e-6


def test_jaw_azimuth_does_not_flip_across_the_half_turn(mapper):
    """A parallel jaw is symmetric, so the command must not spin 180 degrees."""
    mapper.engage(make_pose([0.0, 0.0, 0.5]), HOME)
    forward = frame_from_axes(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    settle(mapper, make_pose([0.0, 0.0, 0.5], forward))
    before = mapper._filtered_azimuth

    backward = frame_from_axes(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0]))
    after = settle(mapper, make_pose([0.0, 0.0, 0.5], backward)).jaw_azimuth
    assert abs(wrap_to_pi(after - before)) < 1e-3


def test_degenerate_pinch_direction_keeps_the_previous_azimuth(mapper):
    mapper.engage(make_pose([0.0, 0.0, 0.5]), HOME)
    sideways = frame_from_axes(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    settle(mapper, make_pose([0.0, 0.0, 0.5], sideways))
    previous = mapper._filtered_azimuth

    # A pinch axis along camera y maps onto the robot's vertical, which has no azimuth.
    vertical = frame_from_axes(np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))
    assert mapper.jaw_azimuth_from_hand(make_pose([0, 0, 0.5], vertical)) == previous


def test_smoothing_reduces_jitter(mapper):
    mapper.engage(make_pose([0.0, 0.0, 0.5]), HOME)
    rng = np.random.default_rng(0)
    anchor = np.array([0.0, 0.0, 0.5])
    raw, filtered = [], []
    for _ in range(400):
        noise = rng.normal(scale=0.005, size=3)
        # Camera x maps onto the robot's -y, scaled by the position gain.
        raw.append(-noise[0] * mapper.hand.position_gain)
        filtered.append(mapper(make_pose(anchor + noise), HOME).position[1] - HOME[1])
    assert np.std(filtered) < 0.75 * np.std(raw), (
        f"raw {np.std(raw):.5f} vs filtered {np.std(filtered):.5f}"
    )


def test_reset_clears_every_filter(mapper):
    mapper.engage(make_pose([0.0, 0.0, 0.5]), HOME)
    settle(mapper, make_pose([0.02, 0.0, 0.5]))
    mapper.reset()
    assert not mapper.engaged and not mapper.anchored
    assert mapper._filtered_position is None
    assert mapper._filtered_azimuth is None
    assert mapper._filtered_gap is None


def test_gain_is_adjustable_and_clamped(mapper):
    assert mapper.position_gain == pytest.approx(HandConfig().position_gain)
    assert mapper.adjust_gain(0.2) > HandConfig().position_gain
    for _ in range(50):
        mapper.adjust_gain(1.0)
    assert mapper.position_gain == pytest.approx(4.0)
    for _ in range(50):
        mapper.adjust_gain(-1.0)
    assert mapper.position_gain == pytest.approx(0.3)


def test_changing_the_gain_does_not_move_the_arm(mapper):
    """Rescaling the offset from the anchor would otherwise teleport the gripper."""
    mapper.engage(make_pose([0.0, 0.0, 0.5]), HOME)
    settle(mapper, make_pose([0.03, 0.0, 0.5]))
    before = mapper._filtered_position.copy()
    mapper.adjust_gain(1.5)
    after = mapper(make_pose([0.03, 0.0, 0.5]), HOME).position
    assert np.linalg.norm(after - before) < 2e-3


def test_a_higher_gain_moves_the_arm_further(mapper):
    mapper.reset()
    mapper.position_gain = 0.5
    mapper.engage(make_pose([0.0, 0.0, 0.5]), HOME)
    small = settle(mapper, make_pose([0.02, 0.0, 0.5])).position.copy()

    mapper.reset()
    mapper.position_gain = 2.0
    mapper.engage(make_pose([0.0, 0.0, 0.5]), HOME)
    large = settle(mapper, make_pose([0.02, 0.0, 0.5])).position.copy()

    assert abs(large[1] - HOME[1]) > abs(small[1] - HOME[1])


def test_reset_keeps_the_operator_chosen_gain(mapper):
    mapper.adjust_gain(0.6)
    chosen = mapper.position_gain
    mapper.reset()
    assert mapper.position_gain == pytest.approx(chosen)


def test_pinching_does_not_move_the_arm(mapper):
    """The bug that made teleoperation unusable: position came from the
    thumb-index midpoint, so closing your hand to grasp something dragged the
    gripper sideways at the exact moment precision mattered most."""
    from handrobot.hands.types import (
        INDEX_TIP,
        Landmarks,
        THUMB_TIP,
    )

    def hand(pinch_offset):
        """A hand held perfectly still, with only the fingers closing."""
        world = np.zeros((21, 3))
        world[0] = [0.0, 0.04, 0.0]      # wrist
        world[5] = [-0.03, -0.02, 0.0]   # index knuckle
        world[9] = [-0.01, -0.03, 0.0]   # middle knuckle
        world[13] = [0.01, -0.03, 0.0]   # ring knuckle
        world[17] = [0.03, -0.02, 0.0]   # pinky knuckle
        world[THUMB_TIP] = [-0.05, -0.05, 0.0] + np.array([pinch_offset, 0.0, 0.0])
        world[INDEX_TIP] = [-0.05, -0.09, 0.0] - np.array([pinch_offset, 0.0, 0.0])
        landmarks = Landmarks(np.zeros((21, 3)), world, "Right", 1.0)
        return HandPose(
            palm_position=np.array([0.0, 0.0, 0.5]),
            rotation=np.eye(3),
            pinch_distance=float(np.linalg.norm(world[THUMB_TIP] - world[INDEX_TIP])),
            depth=0.5,
            landmarks=landmarks,
            timestamp=0.0,
        )

    open_hand, closed_hand = hand(0.02), hand(-0.005)
    assert closed_hand.pinch_distance < open_hand.pinch_distance

    mapper.engage(open_hand, HOME)
    for i in range(40):
        command = mapper(HandPose(**{**open_hand.__dict__, "timestamp": i / 30}), HOME)
    settled = command.position.copy()

    for i in range(40, 80):
        command = mapper(HandPose(**{**closed_hand.__dict__, "timestamp": i / 30}), HOME)

    assert command.jaw_gap < mapper.gripper.max_command_gap * 0.6, "the jaws should close"
    assert np.linalg.norm(command.position - settled) < 1e-9, (
        "closing the hand moved the arm"
    )


def test_hand_jitter_is_strongly_suppressed(mapper):
    """A still hand must produce a still arm, despite MediaPipe's landmark noise."""
    rng = np.random.default_rng(0)
    anchor = np.array([0.0, 0.0, 0.5])
    mapper.engage(make_pose(anchor), HOME)

    raw, filtered = [], []
    for i in range(400):
        noise = rng.normal(scale=0.004, size=3)
        pose = make_pose(anchor + noise)
        pose = HandPose(**{**pose.__dict__, "timestamp": i / 30})
        raw.append(-noise[0] * mapper.position_gain)
        filtered.append(mapper(pose, HOME).position[1] - HOME[1])
    # A plain exponential filter tuned for this much lag keeps about 0.54.
    assert np.std(filtered[100:]) < 0.35 * np.std(raw[100:]), (
        f"raw {np.std(raw[100:]):.5f} vs filtered {np.std(filtered[100:]):.5f}"
    )


def test_the_depth_axis_is_smoothed_harder_than_the_others(mapper):
    """Monocular depth is the noisiest channel, and depth jitter reads as the
    gripper lunging towards and away from the operator."""
    rng = np.random.default_rng(1)
    anchor = np.array([0.0, 0.0, 0.5])
    # Parking freezes a still hand's command outright, which is the right
    # behaviour but hides the filter this test measures. Disable it here.
    from dataclasses import replace

    mapper = HandToGripper(
        replace(mapper.hand, deadband_radius=0.0, deadband_recenter=0.0),
        mapper.workspace, mapper.gripper
    )
    mapper.engage(make_pose(anchor), HOME)

    lateral, depth = [], []
    # Enough samples for the two standard deviations to be distinguishable; the
    # predicted ratio is about 0.65, which needs more than a few hundred frames.
    for i in range(2500):
        noise = rng.normal(scale=0.004, size=3)
        pose = HandPose(**{**make_pose(anchor + noise).__dict__, "timestamp": i / 30})
        position = mapper(pose, HOME).position
        lateral.append(position[1])
        depth.append(position[0])
    ratio = np.std(depth[500:]) / np.std(lateral[500:])
    assert ratio < 0.85, f"depth is not being smoothed harder (ratio {ratio:.2f})"


def test_a_deliberate_move_is_still_followed(mapper):
    """Smoothing must not turn into lag: adaptivity is the whole point."""
    anchor = np.array([0.0, 0.0, 0.5])
    mapper.engage(make_pose(anchor), HOME)
    for i in range(45):
        offset = np.array([0.0015 * i, 0.0, 0.0])
        pose = HandPose(**{**make_pose(anchor + offset).__dict__, "timestamp": i / 30})
        command = mapper(pose, HOME)
    expected = HOME[1] + 0.0015 * 44 * mapper.position_gain
    lag = abs(command.position[1] - expected)
    # Some lag while the hand is in transit is the deliberate cost of smoothing
    # hard enough to hold still over a 25 mm cube. It vanishes once you stop.
    assert lag < 0.06, f"lagging by {lag * 1000:.1f} mm"
    assert command.position[1] > HOME[1] + 0.02, "the arm barely moved at all"


def test_a_camera_stall_cannot_snap_the_arm(mapper):
    """A frozen camera would otherwise hand the filter a huge timestep, and one
    frame would jump the gripper across the workspace."""
    anchor = np.array([0.0, 0.0, 0.5])
    mapper.engage(make_pose(anchor), HOME)
    for i in range(20):
        mapper(HandPose(**{**make_pose(anchor).__dict__, "timestamp": i / 30}), HOME)
    before = mapper._filtered_position.copy()

    stalled = HandPose(**{**make_pose(anchor + np.array([0.3, 0.0, 0.0])).__dict__,
                          "timestamp": 30.0})
    after = mapper(stalled, HOME).position
    assert np.linalg.norm(after - before) < 0.05


def test_a_long_tracking_gap_re_anchors_instead_of_chasing(mapper):
    """Losing the hand is exactly like lifting a mouse: when it comes back, the
    arm must stay where it is rather than leap across to catch up. At 55%
    tracking this is the difference between usable and unusable."""
    anchor = np.array([0.0, 0.0, 0.5])
    mapper.engage(make_pose(anchor), HOME)
    for i in range(40):
        command = mapper(HandPose(**{**make_pose(anchor).__dict__, "timestamp": i / 30}), HOME)
    before = command.position.copy()

    # The hand vanishes for a second and reappears 20 cm away.
    reappeared = HandPose(
        **{**make_pose(anchor + np.array([0.20, 0.0, 0.0])).__dict__, "timestamp": 40 / 30 + 1.0}
    )
    after = mapper(reappeared, HOME).position
    assert np.linalg.norm(after - before) < 2e-3, "the arm chased the hand across the gap"
    assert mapper.tracking_gaps == 1


def test_a_short_gap_is_ridden_through(mapper):
    """One or two dropped frames are normal and must not reset anything."""
    anchor = np.array([0.0, 0.0, 0.5])
    mapper.engage(make_pose(anchor), HOME)
    for i in range(30):
        mapper(HandPose(**{**make_pose(anchor).__dict__, "timestamp": i / 30}), HOME)
    # Two frames missing, then tracking resumes normally.
    mapper(HandPose(**{**make_pose(anchor).__dict__, "timestamp": 32 / 30}), HOME)
    assert mapper.tracking_gaps == 0


def test_pushing_past_the_edge_does_not_wind_up(mapper):
    """The bug that made the arm feel dead: pushing past the reachable region
    kept growing the offset from the anchor, so every centimetre of overshoot
    had to be retraced before the arm moved again."""
    anchor = np.array([0.0, 0.0, 0.5])
    mapper.engage(make_pose(anchor), HOME)

    far = anchor + np.array([0.40, 0.0, 0.0])
    for i in range(150):
        at_edge = mapper(HandPose(**{**make_pose(far).__dict__, "timestamp": i / 30}), HOME)
    assert mapper.saturated

    # Coming back a couple of centimetres must move the arm immediately.
    back = anchor + np.array([0.38, 0.0, 0.0])
    for i in range(150, 190):
        recovered = mapper(HandPose(**{**make_pose(back).__dict__, "timestamp": i / 30}), HOME)
    assert np.linalg.norm(recovered.position - at_edge.position) > 0.02


def test_the_saturation_flag_clears_once_back_inside(mapper):
    anchor = np.array([0.0, 0.0, 0.5])
    mapper.engage(make_pose(anchor), HOME)
    for i in range(80):
        mapper(HandPose(**{**make_pose(anchor + np.array([0.4, 0, 0])).__dict__,
                           "timestamp": i / 30}), HOME)
    assert mapper.saturated
    for i in range(80, 200):
        mapper(HandPose(**{**make_pose(anchor + np.array([0.38, 0, 0])).__dict__,
                           "timestamp": i / 30}), HOME)
    assert not mapper.saturated


def test_hand_move_inverts_the_position_mapping(mapper):
    delta = np.array([0.02, -0.03, 0.01])
    hand = mapper.hand_move_for(delta)
    assert np.allclose(mapper.CAMERA_TO_ROBOT @ (hand * mapper.position_gain), delta)
