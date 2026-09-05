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
    draw_guidance_arrow,
    draw_quality_bar,
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
    """A stage and two tiles, each a flat colour so they can be told apart."""
    return (
        np.full((480, 640, 3), 60, np.uint8),
        np.full((295, 613, 3), 70, np.uint8),
        np.full((295, 613, 3), 80, np.uint8),
    )


def framed(stage, top=None, front=None, state_=None, **kwargs):
    """compose() with the tile column spelled out, the way the loop builds it."""
    tiles = []
    if top is not None:
        tiles.append((top, "TOP VIEW"))
    if front is not None:
        tiles.append((front, "FOLLOW VIEW"))
    return compose(stage, tiles, state_ if state_ is not None else state(), **kwargs)


def test_compose_produces_the_requested_size():
    frame = framed(*panels(), width=1280, height=720)
    assert frame.shape == (720, 1280, 3)


def test_the_camera_panel_is_on_the_left_and_the_views_on_the_right():
    camera, top, front = panels()
    camera[:] = 10
    top[:] = 120
    front[:] = 200
    frame = framed(camera, top, front)
    body = 720 - STRIP_HEIGHT
    assert frame[body // 2, 100].max() < 60, "the stage should be on the left"
    assert 100 < frame[body // 4, 1000].max() < 160, "the first tile belongs above"
    assert frame[3 * body // 4, 1000].max() > 180, "the second tile belongs below"


def test_a_missing_view_does_not_break_the_layout():
    camera, top, _ = panels()
    assert framed(camera, top, None).shape == (720, 1280, 3)
    assert framed(camera, None, None).shape == (720, 1280, 3)


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


def test_the_arrow_points_where_the_hand_should_go():
    """The correction is drawn at the hand, not as a compass in the corner: the
    operator should not have to look away from what they are aiming to read it."""
    for direction, expect in (
        (np.array([80.0, 0.0, 0.0]), "right"),
        (np.array([-80.0, 0.0, 0.0]), "left"),
        (np.array([0.0, 80.0, 0.0]), "down"),
        (np.array([0.0, -80.0, 0.0]), "up"),
    ):
        canvas = np.zeros((320, 400, 3), np.uint8)
        draw_guidance_arrow(canvas, (200, 160), direction)
        lit = np.argwhere(canvas.max(axis=2) > 40)
        assert len(lit) > 0, "nothing was drawn"
        centre = lit.mean(axis=0)
        if expect == "right":
            assert centre[1] > 200
        elif expect == "left":
            assert centre[1] < 200
        elif expect == "down":
            assert centre[0] > 160
        else:
            assert centre[0] < 160


def test_a_tiny_correction_draws_no_arrow():
    canvas = np.zeros((320, 400, 3), np.uint8)
    draw_guidance_arrow(canvas, (200, 160), np.array([1.0, 1.0, 1.0]))
    assert canvas.max() == 0, "an arrow was drawn for a hand that is already there"


def test_the_arrow_never_grows_across_the_whole_stage():
    """It used to scale with the error, which meant the worst case drew a line
    clear across the hand it was guiding."""
    canvas = np.zeros((600, 800, 3), np.uint8)
    draw_guidance_arrow(canvas, (400, 300), np.array([400.0, 0.0, 0.0]))
    lit = np.argwhere(canvas.max(axis=2) > 40)
    assert lit[:, 1].max() - 400 < 200


def test_the_distance_bar_fills_as_the_target_gets_closer():
    def filled(distance):
        strip = draw_strip(1280, state(goal_distance=distance))
        return int((strip.max(axis=2) > 40).sum())

    assert filled(0.0) > filled(0.15) > filled(0.34)


def test_the_strip_says_what_to_do_when_the_clutch_is_off():
    off = draw_strip(1280, state(engaged=False))
    on = draw_strip(1280, state(engaged=True))
    assert not np.array_equal(off, on)


@pytest.mark.parametrize("width", [960, 1280, 1600])
def test_the_strip_adapts_to_the_window_width(width):
    assert draw_strip(width, state()).shape == (STRIP_HEIGHT, width, 3)


# -- resolution independence -------------------------------------------------
#
# The interface is drawn once and shown at whatever size the operator's screen
# is. Every constant in the module is logical, so the only way to know the
# scaling is honest is to compose the same state at several sizes and check the
# geometry rather than the pixels.


@pytest.mark.parametrize("width,height", [(1280, 720), (1920, 1080), (3840, 2160), (7680, 4320)])
def test_the_interface_composes_at_every_size(width, height):
    camera, top, front = panels()
    frame = framed(camera, top, front, width=width, height=height)
    assert frame.shape == (height, width, 3)


@pytest.mark.parametrize("scale", [1.0, 1.5, 3.0, 6.0])
def test_the_strip_scales_without_losing_a_pixel(scale):
    width = round(1280 * scale)
    strip = draw_strip(width, state(), scale=scale)
    assert strip.shape == (round(STRIP_HEIGHT * scale), width, 3)


def test_the_layout_covers_the_frame_exactly():
    from handrobot.viz.hud import panel_geometry

    for width, height in ((1280, 720), (1920, 1080), (7680, 4320)):
        layout = panel_geometry(width, height, height / 720)
        assert layout["stage_height"] == height
        assert layout["stage_width"] + layout["column_width"] == width
        assert layout["ribbon_height"] < height, "the ribbon lies over the stage"


def test_bigger_frames_carry_more_ink_not_bigger_pixels():
    """Text drawn at 4x is drawn, not upscaled: it gains detail, not blur."""
    small = draw_strip(1280, state(), scale=1.0)
    large = draw_strip(5120, state(), scale=4.0)
    import cv2

    shrunk = cv2.resize(large, (1280, STRIP_HEIGHT), interpolation=cv2.INTER_AREA)
    assert not np.array_equal(shrunk, small)
    # Edges survive the downscale: an upscaled 720p strip would be softer than
    # the natively drawn one, never sharper.
    def sharpness(image):
        grey = image.mean(axis=2)
        return float(np.abs(np.diff(grey, axis=1)).mean())

    assert sharpness(shrunk) > 0.8 * sharpness(small)


# -- what the panels say -----------------------------------------------------


def test_panels_are_captioned_with_the_camera_they_show():
    from handrobot.viz.hud import view_label

    assert view_label("wrist_cam") == "WRIST VIEW"
    assert view_label("chase_cam") == "FOLLOW VIEW"
    camera, top, front = panels()
    named = compose(camera, [(top, "TOP VIEW"), (front, "WRIST VIEW")], state())
    other = compose(camera, [(top, "TOP VIEW"), (front, "FOLLOW VIEW")], state())
    assert not np.array_equal(named, other), "the tile caption never reached the frame"


def test_the_stage_takes_the_whole_window_when_there_are_no_tiles():
    """Hiding the column is the operator asking for every pixel on one view."""
    camera = np.full((480, 640, 3), 200, np.uint8)
    tile = np.full((240, 320, 3), 90, np.uint8)

    def stage_pixels(frame):
        return int((frame.max(axis=2) > 150).sum())

    with_column = compose(camera, [(tile, "TOP VIEW")], state())
    full = compose(camera, [], state())
    assert stage_pixels(full) > 1.15 * stage_pixels(with_column)


def test_the_help_overlay_covers_the_camera_panel_only():
    camera, top, front = panels()
    plain = framed(camera, top, front)
    helped = framed(camera, top, front, help_open=True)
    assert not np.array_equal(plain[:600, :600], helped[:600, :600])
    assert np.array_equal(plain[:, 1000:], helped[:, 1000:]), "help spilled onto the tiles"


# -- states that must never be confused with one another ---------------------


def test_a_lost_hand_does_not_look_like_a_released_clutch():
    lost = draw_strip(1280, state(engaged=True, tracked_now=False))
    following = draw_strip(1280, state(engaged=True, tracked_now=True))
    paused = draw_strip(1280, state(engaged=False))
    assert not np.array_equal(lost, following)
    assert not np.array_equal(lost, paused)


def test_a_saved_success_is_announced_and_then_fades():
    camera, top, front = panels()
    quiet = framed(camera, top, front, state_=state(flash=0.0))
    loud = framed(camera, top, front, state_=state(flash=1.0))
    faded = framed(camera, top, front, state_=state(flash=0.005))
    assert not np.array_equal(quiet, loud)
    assert np.array_equal(quiet, faded), "the banner never goes away"


def test_the_episode_limit_is_announced_before_it_bites():
    early = draw_strip(1280, state(episode_steps=100, step_limit=2000))
    late = draw_strip(1280, state(episode_steps=1900, step_limit=2000))
    assert not np.array_equal(early, late)


def test_every_key_the_loop_handles_is_on_screen():
    from handrobot.viz.hud import HELP_LINES

    listed = {key for key, _ in HELP_LINES}
    assert {"SPACE", "N", "S", "D", "H", "[ ]", "V", "W", "?", "Q"} <= listed


# -- text that has to fit ----------------------------------------------------


def test_long_messages_are_trimmed_rather_than_running_off_the_strip():
    from handrobot.viz.hud import elide, text_width

    long = "saved episode_0123456789.npz (success) - " * 4
    trimmed = elide(long, 300, 0.44)
    assert trimmed.endswith("...")
    assert text_width(trimmed, 0.44) <= 300


def test_a_short_message_is_left_alone():
    from handrobot.viz.hud import elide

    assert elide("clutch engaged", 400, 0.44) == "clutch engaged"


def test_the_strip_survives_a_narrow_window():
    assert draw_strip(640, state(message="x" * 200)).shape == (STRIP_HEIGHT, 640, 3)


def test_composing_twice_from_the_same_panels_gives_the_same_frame():
    """Captions are drawn on the frame, never on the caller's image.

    Drawing them on the panel that was passed in leaves them on it, so the
    second frame is captioned twice, the third three times, and the operator
    watches the caption slowly turn into a smudge.
    """
    camera, top, front = panels()
    first = framed(camera, top, front).copy()
    second = framed(camera, top, front)
    assert np.array_equal(first, second)
    assert top.max() == top.min(), "the caption was left on the caller's panel"


def test_the_emphasis_font_is_actually_heavier_than_the_body_font():
    """OpenCV's font enum is a trap: 1 is FONT_HERSHEY_PLAIN, the smallest and
    thinnest face it has, and 2 is DUPLEX. Naming 1 "bold" drew every headline
    lighter and smaller than the explanation underneath it."""
    import cv2

    from handrobot.viz.hud import FONT, FONT_BOLD, text_width

    assert FONT_BOLD == cv2.FONT_HERSHEY_DUPLEX
    headline = "LINED UP WITH THE CUBE"
    assert text_width(headline, 0.6, 1, FONT_BOLD) >= text_width(headline, 0.6, 1, FONT)


# -- the tracking hairline ---------------------------------------------------


def test_the_hairline_marks_lost_and_clipped_frames_only():
    def bar(history):
        canvas = np.zeros((80, 400, 3), np.uint8)
        draw_quality_bar(canvas, history, 1.0)
        return canvas

    healthy = bar(tuple([1] * 120))
    lost = bar(tuple([1] * 60 + [0] * 60))
    clipped = bar(tuple([1] * 60 + [2] * 60))
    assert healthy.max() == 0, "a bar that is always on is a bar nobody reads"
    assert lost.any() and clipped.any()
    assert not np.array_equal(lost, clipped), "a lost frame looks like a clipped one"


def test_the_hairline_stays_at_the_top_of_the_stage():
    camera, top, front = panels()
    plain = framed(camera, top, front)
    marked = framed(camera, top, front, state_=state(tracking_history=tuple([0] * 200)))
    assert not np.array_equal(plain[:8, :600], marked[:8, :600])
    assert np.array_equal(plain[40:600, :600], marked[40:600, :600]), "it covered the picture"


def test_a_tile_column_of_three_fills_the_height():
    camera = np.full((480, 640, 3), 20, np.uint8)
    tiles = [(np.full((240, 320, 3), value, np.uint8), name)
             for value, name in ((80, "TOP VIEW"), (140, "FOLLOW VIEW"), (220, "WRIST VIEW"))]
    frame = compose(camera, tiles, state())
    column = frame[:, 1100]
    assert column[120].max() < 110, "the first tile is missing"
    assert 110 < column[360].max() < 190, "the second tile is missing"
    assert column[600].max() > 190, "the third tile is missing"


# -- the jaw gauge -----------------------------------------------------------


def test_the_jaw_gauge_warns_while_the_jaws_are_narrower_than_the_object():
    wide = draw_strip(1280, state(jaw_gap=0.070, jaw_range=(0.018, 0.078), object_width=0.060))
    narrow = draw_strip(1280, state(jaw_gap=0.030, jaw_range=(0.018, 0.078), object_width=0.060))
    assert not np.array_equal(wide, narrow)
    greens = lambda i: int(((i[:, :, 1] > 200) & (i[:, :, 0] < 160)).sum())
    assert greens(wide) > greens(narrow), "an opening wide enough to fit should read as good"


def test_no_jaw_gauge_before_anything_has_been_commanded():
    assert np.array_equal(draw_strip(1280, state(jaw_gap=None)), draw_strip(1280, state()))


# -- goals that leave the panel ----------------------------------------------


def test_a_goal_outside_the_panel_is_pinned_to_its_edge(monkeypatch):
    """An arrow towards something off the panel points at nothing.

    Which is the case the operator most needs help with: the puck spawned
    outside this view and there is no other clue where it went.
    """
    import handrobot.viz.project as project

    from handrobot.viz.hud import draw_scene_overlays

    placements = iter([(150.0, 100.0), (900.0, 100.0)])  # gripper inside, goal outside
    monkeypatch.setattr(project, "project_point",
                        lambda env, camera, point, h, w: next(placements))

    panel = np.zeros((200, 300, 3), np.uint8)
    drawn = draw_scene_overlays(panel, object(), "top_cam", np.zeros(3), np.ones(3))
    lit = np.argwhere(drawn.max(axis=2) > 40)
    assert lit[:, 1].max() < 300, "the marker was drawn off the panel"
    assert lit[:, 1].max() > 250, "an off-panel goal must show at the edge it left by"


def test_the_arrow_stays_inside_the_picture():
    canvas = np.zeros((300, 400, 3), np.uint8)
    draw_guidance_arrow(canvas, (380, 280), np.array([300.0, 300.0, 0.0]))
    lit = np.argwhere(canvas.max(axis=2) > 40)
    assert len(lit) > 0
    assert lit[:, 1].max() <= 399 and lit[:, 0].max() <= 299
