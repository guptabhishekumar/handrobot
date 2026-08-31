"""Tests for the hand pipeline.

These run against a real photograph of a real hand rather than synthetic
landmarks, so a change that breaks the detector integration is caught rather
than papered over by a mock.
"""

import numpy as np
import pytest

from handrobot.diagnostics import depth_axis_correlation
from handrobot.hands.geometry import (
    CameraIntrinsics,
    estimate_hand_depth,
    hand_pose_from_landmarks,
    palm_centre,
    palm_frame,
    pinch_distance,
)
from handrobot.hands.types import (
    INDEX_MCP,
    INDEX_TIP,
    Landmarks,
    MIDDLE_MCP,
    PINKY_MCP,
    THUMB_TIP,
    WRIST,
)


def test_intrinsics_from_field_of_view():
    intrinsics = CameraIntrinsics.from_hfov(640, 480, 90.0)
    assert intrinsics.fx == pytest.approx(320.0)
    assert intrinsics.cx == pytest.approx(320.0)


def test_unproject_inverts_a_pinhole_projection():
    intrinsics = CameraIntrinsics.from_hfov(640, 480, 62.0)
    point = np.array([0.10, -0.05, 0.60])
    u = intrinsics.fx * point[0] / point[2] + intrinsics.cx
    v = intrinsics.fy * point[1] / point[2] + intrinsics.cy
    assert np.allclose(intrinsics.unproject(u, v, point[2]), point, atol=1e-9)


def test_landmarks_reject_the_wrong_shape():
    with pytest.raises(ValueError):
        Landmarks(image=np.zeros((5, 3)), world=np.zeros((21, 3)), handedness="Left", score=1.0)


def test_real_hand_is_detected(hand_landmarks):
    pose, landmarks = hand_landmarks
    assert landmarks.handedness in {"Left", "Right"}
    assert landmarks.score > 0.5
    assert pose is not None


def test_real_hand_has_plausible_metric_dimensions(hand_landmarks):
    _, landmarks = hand_landmarks
    span = np.linalg.norm(landmarks.world.max(0) - landmarks.world.min(0))
    assert 0.08 < span < 0.35, f"hand span {span:.3f} m is not human-sized"


def test_real_hand_depth_is_plausible(hand_landmarks):
    _, landmarks = hand_landmarks
    intrinsics = CameraIntrinsics.from_hfov(640, 960, 62.0)
    depth = estimate_hand_depth(landmarks, intrinsics)
    assert depth is not None
    assert 0.15 < depth < 2.0


def test_palm_frame_is_a_rotation(hand_landmarks):
    _, landmarks = hand_landmarks
    R = palm_frame(landmarks.world)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_palm_frame_axes_follow_the_anatomy(hand_landmarks):
    _, landmarks = hand_landmarks
    world = landmarks.world
    R = palm_frame(world)
    pointing = world[MIDDLE_MCP] - world[WRIST]
    assert np.dot(R[:, 0], pointing / np.linalg.norm(pointing)) > 0.99
    knuckles = world[PINKY_MCP] - world[INDEX_MCP]
    assert np.dot(R[:, 2], knuckles / np.linalg.norm(knuckles)) > 0.5


def test_the_palm_frame_ignores_the_fingers(hand_landmarks):
    """The point of using the palm: closing your hand must not rotate the arm."""
    _, landmarks = hand_landmarks
    before = palm_frame(landmarks.world)
    curled = landmarks.world.copy()
    for tip in (THUMB_TIP, INDEX_TIP):
        curled[tip] = curled[WRIST] + 0.3 * (curled[tip] - curled[WRIST])
    assert np.allclose(palm_frame(curled), before, atol=1e-12)


def test_the_palm_centre_ignores_the_fingers(hand_landmarks):
    _, landmarks = hand_landmarks
    before = palm_centre(landmarks.world)
    curled = landmarks.world.copy()
    curled[[THUMB_TIP, INDEX_TIP]] = curled[WRIST]
    assert np.allclose(palm_centre(curled), before, atol=1e-12)


def test_depth_is_unaffected_by_finger_movement(hand_landmarks):
    """Depth is fitted over the palm alone, so a pinch cannot move the arm in x."""
    _, landmarks = hand_landmarks
    intrinsics = CameraIntrinsics.from_hfov(640, 960, 62.0)
    before = estimate_hand_depth(landmarks, intrinsics)

    moved_image = landmarks.image.copy()
    moved_world = landmarks.world.copy()
    for tip in (THUMB_TIP, INDEX_TIP):
        moved_image[tip, :2] += 0.05
        moved_world[tip] += 0.02
    after = estimate_hand_depth(
        Landmarks(moved_image, moved_world, landmarks.handedness, landmarks.score),
        intrinsics,
    )
    assert after == pytest.approx(before, rel=1e-9)


def _single_bone_depth(landmarks, intrinsics):
    """The old estimator: one bone measured in pixels against its metric length.

    Kept here only so the replacement can be measured against it.
    """
    scale = np.array([intrinsics.width, intrinsics.height])
    metres = np.linalg.norm(landmarks.world[WRIST] - landmarks.world[MIDDLE_MCP])
    pixels = np.linalg.norm(
        landmarks.image[WRIST, :2] * scale - landmarks.image[MIDDLE_MCP, :2] * scale
    )
    return intrinsics.fx * metres / max(pixels, 1e-6)


@pytest.mark.parametrize("shrink", [0.9, 0.7, 0.5, 0.2])
def test_fitted_depth_beats_a_single_bone_when_the_hand_rotates(hand_landmarks, shrink):
    """A rotating hand foreshortens whichever bone you measure. Fitting one scale
    across five palm landmarks averages that away; measuring one bone does not,
    and the arm lurches in depth every time the wrist turns."""
    _, landmarks = hand_landmarks
    intrinsics = CameraIntrinsics.from_hfov(640, 960, 62.0)

    fitted_before = estimate_hand_depth(landmarks, intrinsics)
    bone_before = _single_bone_depth(landmarks, intrinsics)

    # Foreshorten the wrist-to-middle-knuckle bone, as a wrist rotation would.
    perturbed = landmarks.image.copy()
    anchor = perturbed[WRIST, :2].copy()
    perturbed[MIDDLE_MCP, :2] = anchor + shrink * (perturbed[MIDDLE_MCP, :2] - anchor)
    moved = Landmarks(perturbed, landmarks.world, landmarks.handedness, landmarks.score)

    fitted_error = abs(estimate_hand_depth(moved, intrinsics) - fitted_before) / fitted_before
    bone_error = abs(_single_bone_depth(moved, intrinsics) - bone_before) / bone_before
    assert fitted_error < bone_error * 0.5, (
        f"fitted moved {fitted_error:.1%}, single bone moved {bone_error:.1%}"
    )


def test_pinch_distance_is_a_plausible_finger_span(hand_landmarks):
    _, landmarks = hand_landmarks
    assert 0.0 < pinch_distance(landmarks) < 0.20


def test_pose_is_rejected_when_depth_is_out_of_range(hand_landmarks):
    _, landmarks = hand_landmarks
    intrinsics = CameraIntrinsics.from_hfov(640, 960, 62.0)
    pose = hand_pose_from_landmarks(landmarks, intrinsics, 0.0, depth_range=(5.0, 6.0))
    assert pose is None


def test_pose_is_rejected_for_a_collapsed_hand():
    landmarks = Landmarks(
        image=np.tile(np.array([0.5, 0.5, 0.0]), (21, 1)),
        world=np.zeros((21, 3)),
        handedness="Right",
        score=1.0,
    )
    intrinsics = CameraIntrinsics.from_hfov(640, 480, 62.0)
    assert hand_pose_from_landmarks(landmarks, intrinsics, 0.0) is None


def test_depth_axis_report_flags_an_inverted_convention(hand_landmarks):
    _, landmarks = hand_landmarks
    upright = depth_axis_correlation([landmarks])
    flipped_world = landmarks.world.copy()
    flipped_world[:, 2] *= -1
    flipped = depth_axis_correlation(
        [Landmarks(landmarks.image, flipped_world, landmarks.handedness, landmarks.score)]
    )
    assert flipped.correlation == pytest.approx(-upright.correlation, abs=1e-9)
    assert upright.recommended_sign == -flipped.recommended_sign


class _FakeCategory:
    def __init__(self, name, score=0.9):
        self.category_name = name
        self.score = score


def test_two_hands_choose_the_preferred_one():
    """The operator's other hand rests on the keyboard; following it by accident
    throws the arm across the workspace. The operator's own 'right' arrives
    mirrored, so it carries MediaPipe's 'Left' label."""
    from handrobot.hands.tracker import HandTracker

    tracker = HandTracker.__new__(HandTracker)
    tracker.prefer_hand = "right"
    tracker._followed = None

    def hand(label, centre, spread=0.2):
        image = np.tile(np.array([centre[0], centre[1], 0.0]), (21, 1))
        image[:5, :2] += spread / 2
        image[5:, :2] -= spread / 2
        return Landmarks(image, np.zeros((21, 3)), label, 0.9)

    keyboard = hand("Right", (0.2, 0.8))      # operator's left, resting low
    raised = hand("Left", (0.7, 0.4))         # operator's right, held up
    chosen = tracker._choose([keyboard, raised])
    assert chosen is raised

    # And it stays chosen even when the other hand scores better next frame.
    keyboard_better = hand("Right", (0.2, 0.8))
    keyboard_better = Landmarks(keyboard_better.image, keyboard_better.world,
                                "Right", 0.99)
    assert tracker._choose([keyboard_better, raised]) is raised


def test_without_a_preference_the_larger_hand_wins_then_sticks():
    from handrobot.hands.tracker import HandTracker

    tracker = HandTracker.__new__(HandTracker)
    tracker.prefer_hand = None
    tracker._followed = None

    def hand(label, centre, spread):
        image = np.tile(np.array([centre[0], centre[1], 0.0]), (21, 1))
        image[:10, :2] += spread / 2
        image[10:, :2] -= spread / 2
        return Landmarks(image, np.zeros((21, 3)), label, 0.9)

    small = hand("Right", (0.25, 0.75), 0.08)   # far away, on the desk
    large = hand("Left", (0.6, 0.45), 0.30)     # near the camera, raised
    assert tracker._choose([small, large]) is large

    # The followed hand moves; the choice follows it by position, not label.
    moved = hand("Left", (0.55, 0.5), 0.28)
    assert tracker._choose([moved, small]) is moved

    tracker.forget_hand()
    assert tracker._followed is None


def test_operator_labels_translate_to_mirrored_mediapipe_labels():
    from handrobot.hands.tracker import HandTracker

    assert HandTracker._mediapipe_label("right") == "Left"
    assert HandTracker._mediapipe_label("left") == "Right"
