"""The reach envelope drawn on the webcam preview.

This outline is a promise to the operator: inside it, the arm follows; outside
it, the arm stops. A decorative outline that was merely close would be worse
than none at all, because the operator would trust it and be wrong. So the test
is the promise itself -- for a grid of hand positions, being inside the drawn
polygon has to agree with the mapping actually landing inside the workspace.
"""

import numpy as np
import pytest

from handrobot.config import Config
from handrobot.hands.geometry import CameraIntrinsics
from handrobot.retarget.mapper import HandToGripper
from handrobot.viz.roi import (
    ReachEnvelope,
    draw_depth_band,
    draw_envelope,
    draw_frame_margin,
    envelope_polygons,
)

M = HandToGripper.CAMERA_TO_ROBOT


def _plain_setup(robot="panda"):
    config = Config(robot=robot)
    intrinsics = CameraIntrinsics.from_hfov(640, 480, 62.0)
    return (config, config.workspace, intrinsics,
            np.array([0.0, 0.0, 0.55]), config.workspace.center)


#: A fixed setup for tests that are about drawing style rather than geometry.
SETUP_FOR_STYLE = _plain_setup()


@pytest.fixture(params=["panda", "so101"])
def setup(request):
    config = Config(robot=request.param)
    workspace = config.workspace
    intrinsics = CameraIntrinsics.from_hfov(640, 480, 62.0)
    hand_anchor = np.array([0.0, 0.0, 0.55])
    robot_anchor = workspace.center
    return config, workspace, intrinsics, hand_anchor, robot_anchor


def polygons_for(setup, gain=1.6, plane=None):
    _, workspace, intrinsics, hand_anchor, robot_anchor = setup
    plane_x = float(robot_anchor[0]) if plane is None else plane
    return envelope_polygons(workspace, M, hand_anchor, robot_anchor, gain, plane_x, intrinsics)


def test_the_envelope_exists_around_the_anchor(setup):
    assert polygons_for(setup), "no reachable region was traced at the anchor"


def test_inside_the_outline_means_the_arm_will_follow(setup):
    """The whole promise, checked against the mapping it claims to describe."""
    import cv2

    _, workspace, intrinsics, hand_anchor, robot_anchor = setup
    gain = 1.6
    plane_x = float(robot_anchor[0])
    polygons = polygons_for(setup, gain)
    contours = [np.round(p).astype(np.int32) for p in polygons]

    checked = 0
    for dx in np.linspace(-0.25, 0.25, 17):
        for dy in np.linspace(-0.25, 0.25, 17):
            hand = hand_anchor + np.array([dx, dy, 0.0])
            robot = robot_anchor + M @ (hand - hand_anchor) * gain
            # The slice is taken at one plane; samples the mapping sends off it
            # are not what the outline is drawing.
            assert abs(robot[0] - plane_x) < 1e-9

            u = intrinsics.cx + intrinsics.fx * hand[0] / hand[2]
            v = intrinsics.cy + intrinsics.fy * hand[1] / hand[2]
            drawn = max(
                cv2.pointPolygonTest(contour, (float(u), float(v)), True)
                for contour in contours
            )
            # Ignore samples within a couple of pixels of the outline: the
            # envelope is traced on a finite grid and the boundary itself is
            # where the two definitions are allowed to disagree.
            if abs(drawn) < 4.0:
                continue
            reachable = workspace.contains(robot)
            assert reachable == (drawn > 0), (
                f"hand offset {dx:+.3f},{dy:+.3f}: outline says "
                f"{'inside' if drawn > 0 else 'outside'}, mapping says "
                f"{'reachable' if reachable else 'unreachable'}"
            )
            checked += 1
    assert checked > 100, "the sweep never left the boundary band"


def test_more_sensitivity_shrinks_the_region_you_have_to_move_in(setup):
    """Higher gain means less hand travel covers the same arm travel."""
    def extent(gain):
        points = np.concatenate(polygons_for(setup, gain))
        return float(np.ptp(points[:, 0]))

    assert extent(3.2) < 0.6 * extent(1.6)


def test_the_outline_follows_the_anchor(setup):
    """Re-anchoring somewhere else moves the region, it does not rescale it."""
    _, workspace, intrinsics, hand_anchor, robot_anchor = setup
    base = np.concatenate(
        envelope_polygons(workspace, M, hand_anchor, robot_anchor, 1.6,
                          float(robot_anchor[0]), intrinsics)
    )
    shifted_hand = hand_anchor + np.array([0.05, 0.0, 0.0])
    shifted = np.concatenate(
        envelope_polygons(workspace, M, shifted_hand, robot_anchor, 1.6,
                          float(robot_anchor[0]), intrinsics)
    )
    moved = shifted[:, 0].mean() - base[:, 0].mean()
    assert moved > 20, "moving the anchor right did not move the outline right"
    assert abs(np.ptp(shifted[:, 0]) - np.ptp(base[:, 0])) < 2.0


def test_a_plane_beyond_the_workspace_has_no_envelope(setup):
    _, workspace, _, _, robot_anchor = setup
    assert polygons_for(setup, plane=float(robot_anchor[0]) + 5.0) == []


def test_the_cache_rebuilds_only_when_the_mapping_changes(setup, monkeypatch):
    _, workspace, intrinsics, hand_anchor, robot_anchor = setup
    envelope = ReachEnvelope(workspace=workspace, camera_to_robot=M, intrinsics=intrinsics)

    calls = {"n": 0}
    import handrobot.viz.roi as roi

    original = roi.envelope_polygons

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(roi, "envelope_polygons", counted)

    plane = float(robot_anchor[0])
    for _ in range(5):
        envelope.polygons(hand_anchor, robot_anchor, 1.6, plane)
    assert calls["n"] == 1, "the envelope was retraced for an unchanged mapping"

    envelope.polygons(hand_anchor, robot_anchor, 2.0, plane)
    assert calls["n"] == 2, "a sensitivity change must retrace the envelope"


def test_nothing_is_drawn_without_an_anchor(setup):
    _, workspace, intrinsics, _, _ = setup
    envelope = ReachEnvelope(workspace=workspace, camera_to_robot=M, intrinsics=intrinsics)
    assert envelope.polygons(None, None, 1.6, 0.4) == []


def test_the_envelope_is_visible_and_warns_when_saturated(setup):
    polygons = polygons_for(setup)
    normal = np.zeros((480, 640, 3), np.uint8)
    pinned = np.zeros((480, 640, 3), np.uint8)
    draw_envelope(normal, polygons, saturated=False)
    draw_envelope(pinned, polygons, saturated=True)
    assert normal.any(), "the reachable region was never drawn"
    assert not np.array_equal(normal, pinned), "saturation looks identical to normal"


def test_the_depth_gauge_says_which_way_to_move():
    def drawn(distance):
        canvas = np.zeros((480, 640, 3), np.uint8)
        draw_depth_band(canvas, distance, (0.30, 0.80), (0.15, 1.60))
        return canvas

    inside, too_close, too_far = drawn(0.55), drawn(0.18), drawn(1.4)
    assert not np.array_equal(inside, too_close)
    assert not np.array_equal(too_close, too_far)
    # The marker tracks the distance: closer sits lower on the gauge.
    def marker_row(image):
        rows = np.argwhere(image[:, 600:, 1] > 150)
        return rows[:, 0].mean()

    assert marker_row(too_close) > marker_row(too_far)


def test_an_unresolved_hand_still_draws_the_gauge():
    canvas = np.zeros((480, 640, 3), np.uint8)
    draw_depth_band(canvas, None, (0.30, 0.80), (0.15, 1.60))
    assert canvas.any()


def test_the_frame_margin_turns_red_when_the_hand_reaches_it():
    calm = np.zeros((480, 640, 3), np.uint8)
    alarm = np.zeros((480, 640, 3), np.uint8)
    draw_frame_margin(calm, clipped=False)
    draw_frame_margin(alarm, clipped=True)
    assert int((alarm[:, :, 2] > 150).sum()) > int((calm[:, :, 2] > 150).sum())


def test_saturation_is_dashed_as_well_as_recoloured():
    """Green against amber is the commonest form of colour blindness, and this
    warning explains why the arm has stopped moving."""
    polygons = polygons_for(SETUP_FOR_STYLE)
    solid = np.zeros((480, 640, 3), np.uint8)
    dashed = np.zeros((480, 640, 3), np.uint8)
    draw_envelope(solid, polygons, saturated=False)
    draw_envelope(dashed, polygons, saturated=True)
    lit = lambda i: int((i.max(axis=2) > 40).sum())
    assert lit(dashed) < 0.75 * lit(solid), "the saturated outline is not dashed"
    assert lit(dashed) > 0, "nothing was drawn at all"


def test_the_outline_never_leaves_the_picture():
    """An outline running off the edge reads as a broken drawing. Clipped, it
    stays a closed region and its border edges say "this carries on"."""
    from handrobot.viz.roi import clip_to_frame

    _, workspace, intrinsics, hand_anchor, robot_anchor = SETUP_FOR_STYLE
    # A very low gain blows the region up far beyond the frame.
    polygons = envelope_polygons(workspace, M, hand_anchor, robot_anchor, 0.35,
                                 float(robot_anchor[0]), intrinsics)
    assert polygons
    raw = np.concatenate(polygons)
    assert raw[:, 0].min() < 0 or raw[:, 0].max() > 640, "this case is meant to overflow"

    for polygon in polygons:
        clipped = clip_to_frame(polygon, 640, 480)
        if len(clipped) == 0:
            continue
        assert clipped[:, 0].min() >= -0.01 and clipped[:, 0].max() <= 640
        assert clipped[:, 1].min() >= -0.01 and clipped[:, 1].max() <= 480


def test_an_outline_that_fills_the_view_is_not_drawn():
    _, workspace, intrinsics, hand_anchor, robot_anchor = SETUP_FOR_STYLE
    polygons = envelope_polygons(workspace, M, hand_anchor, robot_anchor, 0.2,
                                 float(robot_anchor[0]), intrinsics)
    canvas = np.zeros((480, 640, 3), np.uint8)
    draw_envelope(canvas, polygons)
    assert canvas.max() == 0, "an outline hugging the border only crowds the picture"
