"""Tests for the teleoperation interface.

Drawing code is easy to leave broken because it still produces *an* image. These
check the things that would actually mislead an operator: that the guidance
names the right direction, that the state is distinguishable at a glance, and
that the panels are where they are supposed to be.
"""

import numpy as np
import pytest

from handrobot.viz.hud import (
    STRIP_HEIGHT,
    HudState,
    compose,
    draw_bar,
    draw_direction,
    draw_strip,
)


def state(**overrides) -> HudState:
    fields = dict(
        engaged=True, recording=True, episode_steps=42, saved=3, successes=2,
        tracking=0.85, fps=29.0, sensitivity=1.6, message="", goal_name="cube",
        goal_distance=0.08, hand_move=np.array([-0.04, 0.01, 0.02]),
    )
    fields.update(overrides)
    return HudState(**fields)


def panels():
    return (
        np.full((480, 640, 3), 60, np.uint8),
        np.full((295, 613, 3), 70, np.uint8),
        np.full((295, 613, 3), 80, np.uint8),
    )


def test_compose_produces_the_requested_size():
    frame = compose(*panels(), state(), width=1280, height=720)
    assert frame.shape == (720, 1280, 3)


def test_the_camera_panel_is_on_the_left_and_the_views_on_the_right():
    camera, top, front = panels()
    camera[:] = 10
    top[:] = 120
    front[:] = 200
    frame = compose(camera, top, front, state())
    body = 720 - STRIP_HEIGHT
    assert frame[body // 2, 100].max() < 60, "the camera should be on the left"
    assert 100 < frame[body // 4, 1000].max() < 160, "the top view belongs above"
    assert frame[3 * body // 4, 1000].max() > 180, "the front view belongs below"


def test_a_missing_view_does_not_break_the_layout():
    camera, top, _ = panels()
    assert compose(camera, top, None, state()).shape == (720, 1280, 3)
    assert compose(camera, None, None, state()).shape == (720, 1280, 3)


def test_the_strip_distinguishes_the_three_states():
    recording = draw_strip(1280, state(engaged=True, recording=True))
    following = draw_strip(1280, state(engaged=True, recording=False))
    paused = draw_strip(1280, state(engaged=False, recording=False))
    assert not np.array_equal(recording, following)
    assert not np.array_equal(following, paused)


def test_saturation_is_impossible_to_miss():
    normal = draw_strip(1280, state())
    pinned = draw_strip(1280, state(saturated=True))
    assert not np.array_equal(normal, pinned)
    # Drawn in red, which nothing else in the strip uses at that scale.
    reds = int(((pinned[:, :, 2] > 200) & (pinned[:, :, 1] < 120)).sum())
    assert reds > ((normal[:, :, 2] > 200) & (normal[:, :, 1] < 120)).sum()


def test_the_compass_points_where_the_hand_should_go():
    for direction, expect in (
        (np.array([80.0, 0.0, 0.0]), "right"),
        (np.array([-80.0, 0.0, 0.0]), "left"),
        (np.array([0.0, 80.0, 0.0]), "down"),
        (np.array([0.0, -80.0, 0.0]), "up"),
    ):
        canvas = np.zeros((160, 200, 3), np.uint8)
        draw_direction(canvas, (100, 80), direction)
        lit = np.argwhere(canvas.max(axis=2) > 40)
        assert len(lit) > 0, "nothing was drawn"
        centre = lit.mean(axis=0)
        if expect == "right":
            assert centre[1] > 100
        elif expect == "left":
            assert centre[1] < 100
        elif expect == "down":
            assert centre[0] > 80
        else:
            assert centre[0] < 80


def test_a_tiny_correction_draws_no_arrow():
    canvas = np.zeros((160, 200, 3), np.uint8)
    draw_direction(canvas, (100, 80), np.array([1.0, 1.0, 1.0]))
    # Only the empty frame and crosshair, no arrow reaching towards an edge.
    assert canvas.max(axis=2)[:, 160:].max() == 0


def test_the_bar_fills_as_the_target_gets_closer():
    def filled(distance):
        canvas = np.zeros((40, 300, 3), np.uint8)
        draw_bar(canvas, (10, 10), (280, 14), distance, 350.0, 25.0)
        return int((canvas.max(axis=2) > 40).sum())

    assert filled(0.0) > filled(150.0) > filled(340.0)


def test_the_strip_says_what_to_do_when_the_clutch_is_off():
    off = draw_strip(1280, state(engaged=False))
    on = draw_strip(1280, state(engaged=True))
    assert not np.array_equal(off, on)


@pytest.mark.parametrize("width", [960, 1280, 1600])
def test_the_strip_adapts_to_the_window_width(width):
    assert draw_strip(width, state()).shape == (STRIP_HEIGHT, width, 3)
