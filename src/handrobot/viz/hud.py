"""The teleoperation interface.

Composed as one frame rather than text stamped over the camera image, because
the operator has to read five things at a glance while their hands are busy:
whether the hand is tracked, whether the clutch is on, whether the episode is
recording, which way to move, and how close they are. Anything that takes a
second look is a second the gripper spends drifting.

Layout::

    +---------------------------+------------------+
    |                           |     TOP VIEW     |
    |   YOUR HAND               +------------------+
    |   webcam + tracking       |   FOLLOW VIEW    |
    |                           |        [ wrist ] |
    +---------------------------+------------------+
    |  status strip: state, guidance, distance bar |
    +----------------------------------------------+

Resolution independence
-----------------------
Every coordinate in this module is *logical*: it describes the interface at a
1280x720 reference, and one ``scale`` factor turns that into whatever the
window actually is -- 720p on a laptop, 8K on a wall. Nothing is written in
device pixels, so no layout can be correct at one size and broken at another.

Text is supersampled by whatever factor keeps strokes at least two pixels wide
(``_ss_factor``): OpenCV's Hershey fonts are stroked vectors with crude
anti-aliasing at small sizes, and supersampling is the difference between text
that looks like a debug tool and text that looks like an interface. Above 2x
scale the strokes are already wide enough, so the supersample is dropped and
the cost with it -- rendering an 8K strip at another 2x would cost four times
the memory bandwidth for no visible gain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A small, deliberately flat palette. BGR, because that is what OpenCV draws in.
INK = (240, 240, 244)
DIM = (176, 176, 182)
FAINT = (110, 110, 118)
PANEL = (22, 22, 26)
EDGE = (44, 44, 50)
GREEN = (120, 225, 145)
AMBER = (70, 190, 250)
RED = (80, 80, 245)
BLUE = (225, 170, 90)

FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
#: cv2.FONT_HERSHEY_DUPLEX. It is 2, not 1 -- 1 is FONT_HERSHEY_PLAIN, the
#: smallest and thinnest face OpenCV has, which is what every headline in this
#: interface used to be drawn in: the emphasis line came out lighter and
#: smaller than the explanation beneath it.
FONT_BOLD = 2

#: The reference interface. Every constant below is in these units.
REFERENCE_WIDTH = 1280
REFERENCE_HEIGHT = 720

#: Height of the status ribbon. It is drawn *over* the stage rather than beside
#: it: a bar that owns its own band takes that band from the picture on every
#: frame, including the ninety-nine per cent of frames where nothing in it has
#: changed.
STRIP_HEIGHT = 96
GAP = 2

#: Width of the tile column, as a fraction of the window. One large stage with
#: tiles for the other sources is how every tool that watches several cameras at
#: once lays them out, and for the same reason: attention is not divisible, so
#: the layout should not divide it either. The tiles still have to be usable
#: though -- a view too small to read is not a view, it is a decoration -- so
#: they take a third of the width and every pixel of the height between them.
TILE_FRACTION = 0.34

#: Kept for callers that still ask for the old split.
CAMERA_FRACTION = 1.0 - TILE_FRACTION

#: Human names for the simulator viewpoints. The panel caption has to say which
#: camera it is actually showing -- captioning every lower view "CHASE VIEW"
#: made the wrist camera look like a broken chase camera.
VIEW_LABELS = {
    "top_cam": "TOP VIEW",
    "front_cam": "FRONT VIEW",
    "chase_cam": "FOLLOW VIEW",
    "hero_cam": "WIDE VIEW",
    "wrist_cam": "WRIST VIEW",
}


def view_label(camera: str) -> str:
    """Caption for a simulator camera."""
    return VIEW_LABELS.get(camera, camera.replace("_", " ").upper())


def _ss_factor(scale: float) -> int:
    """Supersample enough to keep a one-pixel stroke at least two pixels wide."""
    return max(1, int(np.ceil(2.0 / max(scale, 1e-6))))


@dataclass
class HudState:
    """Everything the strip draws, gathered in one place."""

    engaged: bool
    recording: bool
    episode_steps: int
    saved: int
    successes: int
    tracking: float
    fps: float
    sensitivity: float
    message: str
    goal_name: str = "cube"
    goal_distance: float | None = None
    hand_move: np.ndarray | None = None
    saturated: bool = False
    rejection: str | None = None
    holding: bool = False
    hands_seen: int = 0
    followed_hand: str | None = None
    #: Whether *this* frame produced a usable hand pose. A released clutch and a
    #: lost hand both leave the arm still, and the operator has to be able to
    #: tell which is happening.
    tracked_now: bool = True
    #: Steps allowed in one episode, so the strip can warn before the cut-off
    #: rather than announcing it afterwards.
    step_limit: int = 0
    #: 1.0 immediately after a success, fading to 0. Drives the banner.
    flash: float = 0.0
    #: Measured cost of one control period, and the budget it has to fit in.
    #: A loop that has fallen behind feels wrong long before it looks wrong, and
    #: an operator who can see the number stops blaming their own hand for it.
    loop_ms: float | None = None
    loop_budget_ms: float = 1000.0 / 30.0
    #: Per-frame tracking quality over the last few seconds: 1 tracked,
    #: 2 tracked but touching the frame edge, 0 lost. Drawn as a timeline,
    #: because the shape of the loss is the diagnosis -- a steady 3% is a
    #: healthy detector, and the same 3% arriving in one burst is a hand that
    #: left the frame.
    tracking_history: tuple[int, ...] = ()
    #: Commanded jaw gap and the range it is commanded within, in metres, plus
    #: the width of the thing being grasped.
    jaw_gap: float | None = None
    jaw_range: tuple[float, float] = (0.010, 0.075)
    object_width: float | None = None


class _Canvas:
    """A logical drawing surface that renders at ``scale`` device pixels per unit.

    Every drawing call takes *logical* coordinates; the scaling and the
    supersampling live in one place so nothing can be half-scaled.
    """

    def __init__(self, height: int, width: int, fill, scale: float = 1.0) -> None:
        self.scale = float(scale)
        self.ss = _ss_factor(self.scale)
        self.k = self.scale * self.ss
        self.height = int(height)
        self.width = int(width)
        self.canvas = np.full(
            (max(1, round(self.height * self.k)), max(1, round(self.width * self.k)), 3),
            fill,
            np.uint8,
        )

    # -- helpers ------------------------------------------------------------

    def _point(self, point) -> tuple[int, int]:
        return (int(round(point[0] * self.k)), int(round(point[1] * self.k)))

    def _thickness(self, thickness: float) -> int:
        return max(1, int(round(thickness * self.k)))

    # -- primitives ---------------------------------------------------------

    def text(self, text, origin, scale=0.6, colour=INK, thickness=1, font=FONT):
        import cv2

        cv2.putText(self.canvas, text, self._point(origin), font, scale * self.k,
                    colour, self._thickness(thickness), cv2.LINE_AA)

    def line(self, a, b, colour, thickness=1):
        import cv2

        cv2.line(self.canvas, self._point(a), self._point(b), colour,
                 self._thickness(thickness), cv2.LINE_AA)

    def rectangle(self, a, b, colour, thickness=1):
        import cv2

        cv2.rectangle(self.canvas, self._point(a), self._point(b), colour,
                      -1 if thickness < 0 else self._thickness(thickness))

    def circle(self, centre, radius, colour, thickness=-1):
        import cv2

        cv2.circle(self.canvas, self._point(centre), max(1, int(round(radius * self.k))),
                   colour, -1 if thickness < 0 else self._thickness(thickness), cv2.LINE_AA)

    def result(self) -> np.ndarray:
        import cv2

        target = (max(1, round(self.width * self.scale)), max(1, round(self.height * self.scale)))
        if (self.canvas.shape[1], self.canvas.shape[0]) == target:
            return self.canvas
        return cv2.resize(self.canvas, target, interpolation=cv2.INTER_AREA)


def text_width(text: str, scale: float = 0.6, thickness: int = 1, font: int = FONT) -> int:
    """Logical width of a string, measured rather than guessed.

    Guessing it as a fixed number of pixels per character was wrong for every
    string that was not mostly lower-case, which is how captions ended up with
    their backing box cutting through the last letter.
    """
    import cv2

    return int(cv2.getTextSize(text, font, scale, max(1, thickness))[0][0])


def elide(text: str, budget: int, scale: float = 0.6, font: int = FONT) -> str:
    """Trim a string to fit a logical width, ending in an ellipsis if cut."""
    if budget <= 0 or not text:
        return ""
    if text_width(text, scale, 1, font) <= budget:
        return text
    trimmed = text
    while trimmed and text_width(trimmed + "...", scale, 1, font) > budget:
        trimmed = trimmed[:-1]
    return (trimmed + "...") if trimmed else ""


def _text(image, text, origin, scale=0.6, colour=INK, thickness=1, font=FONT):
    import cv2

    cv2.putText(image, text, (int(origin[0]), int(origin[1])), font, scale, colour,
                max(1, int(round(thickness))), cv2.LINE_AA)


def _fit(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize into a box, preserving aspect, padding with the panel colour."""
    import cv2

    height = max(1, int(height))
    width = max(1, int(width))
    if image.shape[0] == height and image.shape[1] == width:
        return image  # already exactly the box: no resample, no letterbox, no copy
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    if resized.shape[0] == height and resized.shape[1] == width:
        return resized
    canvas = np.full((height, width, 3), PANEL, np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _label(image: np.ndarray, text: str, scale: float = 1.0) -> np.ndarray:
    """Caption a panel in its top-left corner, over a soft scrim."""
    import cv2

    caption_scale = 0.48
    box_w = min(image.shape[1], round((16 + text_width(text, caption_scale)) * scale))
    box_h = min(image.shape[0], round(26 * scale))
    if box_w <= 2 or box_h <= 2:
        return image
    band = image[:box_h, :box_w].copy()
    cv2.rectangle(image, (0, 0), (box_w, box_h), PANEL, -1)
    cv2.addWeighted(band, 0.35, image[:box_h, :box_w], 0.65, 0, image[:box_h, :box_w])
    surface = _Canvas(round(box_h / scale), round(box_w / scale), PANEL, scale)
    surface.canvas[:] = cv2.resize(
        image[:box_h, :box_w], (surface.canvas.shape[1], surface.canvas.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    surface.text(text, (8, 18), caption_scale, (205, 205, 210))
    image[:box_h, :box_w] = cv2.resize(
        surface.result(), (box_w, box_h), interpolation=cv2.INTER_AREA
    )
    return image


def draw_jaw(surface: "_Canvas", origin, gap: float, jaw_range: tuple[float, float],
             object_width: float | None) -> None:
    """The commanded jaw opening against the width of the thing being grasped.

    The pinch-to-jaw mapping is absolute, so the operator can be at the rail --
    fingers fully closed, jaws still 10 mm apart -- and have no way of knowing.
    The tick is the object: the fill has to be past it to go around it, and
    back inside it to hold it.
    """
    x, y = origin
    width, height = 130, 14
    low, high = jaw_range
    surface.text("JAW", (x, y - 4), 0.38, FAINT)
    surface.rectangle((x, y), (x + width, y + height), EDGE, 1)

    span = max(high - low, 1e-6)
    fraction = float(np.clip((gap - low) / span, 0.0, 1.0))
    filled = int((width - 2) * fraction)
    if filled > 0:
        near_object = object_width is not None and gap < object_width
        surface.rectangle((x + 1, y + 1), (x + 1 + filled, y + height - 1),
                          AMBER if near_object else GREEN, -1)
    if object_width is not None and low <= object_width <= high:
        tick = x + int((width - 2) * (object_width - low) / span)
        surface.line((tick, y - 4), (tick, y + height + 4), INK, 1)
    surface.text(f"{gap * 1000:.0f} mm", (x + width + 8, y + height - 2), 0.42, DIM)


def _state_words(state: HudState) -> tuple[str, str, tuple[int, int, int], tuple[int, int, int]]:
    """The one line that says what the system is doing, and why.

    Ordered by what stops a demonstration: a lost hand beats a released clutch,
    which beats whether the recorder is running.
    """
    if state.engaged and not state.tracked_now:
        return ("HAND LOST", "show your open palm", AMBER, AMBER)
    if state.recording and state.engaged:
        return ("RECORDING", "", INK, RED)
    if state.engaged:
        return ("FOLLOWING", "not recording - press N", INK, GREEN)
    return ("PAUSED", "press SPACE to engage", DIM, FAINT)


def draw_strip(width: int, state: HudState, scale: float = 1.0) -> np.ndarray:
    """The status ribbon that lies along the bottom of the stage.

    Three rows and three columns, and nothing in it moves between them. What
    the system is doing on the left, what to do next in the middle, what the
    hand is holding on the right; the row a thing appears in never changes, so
    the operator learns where to look rather than reading the whole ribbon.

    ``width`` is in device pixels; everything inside is drawn in logical units.
    """
    logical_width = max(360, round(width / scale))
    surface = _Canvas(STRIP_HEIGHT, logical_width, PANEL, scale)
    surface.line((0, 0), (logical_width, 0), EDGE, 1)

    # Columns as fractions of the ribbon, not fixed pixels: the ribbon is as
    # wide as the stage, and the stage changes with the window.
    left_column = int(logical_width * 0.22)
    middle = int(logical_width * 0.26)
    right = int(logical_width * 0.74)
    middle_width = right - middle - 20

    # --- left: what the system is doing ----------------------------------
    headline, reason, ink, dot = _state_words(state)
    surface.circle((26, 26), 7, dot)
    surface.text(headline, (44, 32), 0.6, ink, 1, FONT_BOLD)

    urgent = bool(reason)
    if not reason and state.engaged:
        reason = f"step {state.episode_steps}"
        if state.step_limit:
            reason += f" of {state.step_limit}"
        if state.hands_seen > 1 and state.followed_hand:
            reason += f"   {state.followed_hand} hand"
    if state.step_limit and state.episode_steps > 0.75 * state.step_limit:
        # The recorder stops silently at the limit, and a demonstration finished
        # after it stopped is a demonstration nobody has.
        reason = f"{max(0, state.step_limit - state.episode_steps)} steps left"
        urgent = True
    if reason:
        surface.text(elide(reason, left_column, 0.42), (44, 54), 0.42,
                     AMBER if urgent else DIM)

    surface.text(elide(f"saved {state.saved}   success {state.successes}", left_column, 0.42),
                 (20, 80), 0.42, DIM)
    surface.line((middle - 22, 14), (middle - 22, STRIP_HEIGHT - 14), EDGE, 1)

    # --- middle: what to do next -----------------------------------------
    if state.saturated:
        surface.text("AT THE EDGE OF REACH", (middle, 32), 0.62, RED, 1, FONT_BOLD)
        detail = "the arm has stopped - move your hand back inside the outline"
    elif state.engaged and not state.tracked_now:
        surface.text("TRACKING LOST", (middle, 32), 0.62, AMBER, 1, FONT_BOLD)
        detail = state.rejection or "hold your open palm towards the camera"
    elif not state.engaged:
        surface.text("PRESS SPACE TO TAKE THE ARM", (middle, 32), 0.58, DIM, 1, FONT_BOLD)
        detail = "hold your hand still first - the clutch anchors to a settled reading"
    else:
        move = None if state.hand_move is None else state.hand_move * 1000
        parts = []
        if move is not None:
            if abs(move[0]) > 5:
                parts.append(("RIGHT" if move[0] > 0 else "LEFT", abs(move[0])))
            if abs(move[1]) > 5:
                parts.append(("DOWN" if move[1] > 0 else "UP", abs(move[1])))
            if abs(move[2]) > 5:
                parts.append(("PULL" if move[2] > 0 else "PUSH", abs(move[2])))
        if parts:
            surface.text("   ".join(f"{word} {value:.0f}" for word, value in parts),
                         (middle, 32), 0.7, INK, 1, FONT_BOLD)
            detail = f"millimetres of hand movement, to reach the {state.goal_name}"
        else:
            surface.text(f"LINED UP WITH THE {state.goal_name.upper()}", (middle, 32),
                         0.7, GREEN, 1, FONT_BOLD)
            detail = ("carry it over, then open your fingers" if state.holding
                      else "lower, then pinch to close")

    # A message reports something that just happened, so it outranks standing
    # advice for the few seconds it is fresh; they share the row rather than
    # fighting over it.
    if state.message:
        surface.text(elide(state.message, middle_width, 0.44), (middle, 54), 0.44, AMBER)
    else:
        surface.text(elide(detail, middle_width, 0.44), (middle, 54), 0.44, FAINT)

    if state.goal_distance is not None:
        bar_width = max(100, min(300, middle_width - 160))
        value = state.goal_distance * 1000
        fraction = float(np.clip(1.0 - value / 350.0, 0.0, 1.0))
        colour = GREEN if value <= 25.0 else (AMBER if value <= 75.0 else BLUE)
        surface.rectangle((middle, 68), (middle + bar_width, 82), EDGE, 1)
        if fraction > 0:
            surface.rectangle((middle + 1, 69),
                              (middle + 1 + int((bar_width - 2) * fraction), 81), colour, -1)
        marker = middle + int((bar_width - 2) * (1.0 - 25.0 / 350.0))
        surface.line((marker, 65), (marker, 85), DIM, 1)
        surface.text(f"{value:.0f} mm", (middle + bar_width + 12, 80), 0.44, DIM)

    # --- right: the hand, and the keys -----------------------------------
    if state.jaw_gap is not None:
        draw_jaw(surface, (right, 22), state.jaw_gap, state.jaw_range, state.object_width)
    right_width = logical_width - right - 12
    slow = state.loop_ms is not None and state.loop_ms > 1.25 * state.loop_budget_ms
    if state.rejection and state.tracked_now:
        surface.text(elide(f"losing frames: {state.rejection}", right_width, 0.4),
                     (right, 56), 0.4, AMBER)
    elif slow:
        surface.text(f"slow loop: {state.loop_ms:.0f} ms", (right, 56), 0.4, AMBER)
    else:
        health = f"{state.tracking * 100:.0f}% tracked   {state.fps:.0f} fps"
        if state.sensitivity:
            health += f"   {state.sensitivity:.1f}x"
        surface.text(elide(health, right_width, 0.4), (right, 56), 0.4,
                     DIM if state.tracking > 0.7 else AMBER)

    keys = "SPACE clutch   N next   ? keys"
    surface.text(elide(keys, right_width, 0.4), (right, 80), 0.4, FAINT)

    strip = surface.result()
    target = (int(width), max(1, round(STRIP_HEIGHT * scale)))
    if (strip.shape[1], strip.shape[0]) != target:
        import cv2

        strip = cv2.resize(strip, target, interpolation=cv2.INTER_AREA)
    return strip


def draw_quality_bar(image: np.ndarray, history, scale: float = 1.0) -> np.ndarray:
    """Tracking quality as a hairline along the top of the stage, in place.

    Shown only when there is something to show. A bar that is solid green
    whatever happens teaches the operator to ignore it, and then it is not
    there when it matters.
    """
    import cv2

    samples = list(history)
    if not samples or all(q == 1 for q in samples):
        return image
    height, width = image.shape[:2]
    band = max(3, round(5 * scale))
    step = width / len(samples)
    for index, quality in enumerate(samples):
        if quality == 1:
            continue
        x = int(index * step)
        cv2.rectangle(image, (x, 0), (x + max(1, int(step)), band),
                      AMBER if quality == 2 else RED, -1)
    cv2.line(image, (0, band), (width, band), EDGE, 1)
    return image


def draw_guidance_arrow(image: np.ndarray, origin, move_mm, scale: float = 1.0,
                        bottom_inset: float = 0.0) -> np.ndarray:
    """One arrow, at the hand, pointing where the hand should go.

    The correction used to be a compass in the corner of the ribbon, which asks
    the operator to look away from the thing they are aiming, translate a
    diagram, and look back. Drawn at the hand it needs no translating at all;
    depth, which no flat arrow can honestly show, stays a word.
    """
    import cv2

    if move_mm is None:
        return image
    planar = np.asarray(move_mm[:2], dtype=float)
    magnitude = float(np.linalg.norm(planar))
    x, y = int(origin[0]), int(origin[1])

    if magnitude > 6:
        direction = planar / magnitude
        # Bounded, and it starts clear of the hand: an arrow that grows with the
        # error ends up crossing the fingers it is meant to be guiding, and a
        # long line over a hand reads as part of the skeleton.
        height, width = image.shape[:2]
        height = max(1, int(height - bottom_inset))
        margin = round(12 * scale)
        start = (int(x + direction[0] * 26 * scale), int(y + direction[1] * 26 * scale))
        length = min(110.0, 34.0 + magnitude * 0.8) * scale
        end = [x + direction[0] * (26 * scale + length),
               y + direction[1] * (26 * scale + length)]
        # Shorten rather than run off: an arrowhead the viewer cannot see is an
        # arrow that only says "somewhere over there".
        for axis, limit in ((0, width), (1, height)):
            if end[axis] < margin or end[axis] > limit - margin:
                bound = margin if end[axis] < margin else limit - margin
                travel = end[axis] - (x if axis == 0 else y)
                if abs(travel) > 1e-6:
                    ratio = (bound - (x if axis == 0 else y)) / travel
                    end = [x + (end[0] - x) * ratio, y + (end[1] - y) * ratio]
        end = (int(end[0]), int(end[1]))
        colour = GREEN if magnitude < 15 else AMBER
        if abs(end[0] - start[0]) + abs(end[1] - start[1]) > 6 * scale:
            cv2.arrowedLine(image, start, end, colour, max(2, round(3 * scale)),
                            cv2.LINE_AA, tipLength=0.3)
    # No number on the arrow and no word for depth: the ribbon already gives
    # both in millimetres, and the same figure printed twice is one more thing
    # to read over the hand the operator is trying to watch.
    return image


def draw_scene_overlays(panel: np.ndarray, env, camera: str,
                        tcp: np.ndarray | None, goal: np.ndarray | None,
                        show_arrow: bool = True) -> np.ndarray:
    """Draw the gripper crosshair and the path to the goal onto a rendered view.

    Projected with the same verified pinhole model the tests check against the
    renderer, so the arrow points at the goal's actual pixels, not at a guess.
    Drawn on a copy: the underlying render is refreshed less often than the
    overlays, and arrows must never accumulate.
    """
    import cv2

    from handrobot.viz.project import project_point

    panel = panel.copy()
    height, width = panel.shape[:2]
    # The crosshair is a fixed fraction of the panel, so it stays the same size
    # on the operator's screen whatever resolution the panel is rendered at.
    unit = max(1.0, height / 295.0)

    tcp_px = None if tcp is None else project_point(env, camera, tcp, height, width)
    goal_px = None if goal is None else project_point(env, camera, goal, height, width)

    if show_arrow and goal_px is not None and tcp_px is not None:
        distance = float(np.hypot(goal_px[0] - tcp_px[0], goal_px[1] - tcp_px[1]))
        if distance > 18 * unit:
            colour = GREEN if distance < 60 * unit else AMBER
            cv2.arrowedLine(panel, (int(tcp_px[0]), int(tcp_px[1])),
                            (int(goal_px[0]), int(goal_px[1])), colour,
                            max(1, round(2 * unit)), cv2.LINE_AA,
                            tipLength=min(0.35, 14.0 * unit / distance))

    # A goal outside the panel is the case the operator most needs help with,
    # and the one a plain arrow handles worst: it points off the edge at
    # nothing. Pin a marker to the border instead, on the line to the goal.
    if goal_px is not None and not (0 <= goal_px[0] < width and 0 <= goal_px[1] < height):
        edge = max(round(10 * unit), 6)
        u = int(np.clip(goal_px[0], edge, width - edge))
        v = int(np.clip(goal_px[1], edge, height - edge))
        cv2.circle(panel, (u, v), edge, AMBER, max(1, round(2 * unit)), cv2.LINE_AA)
        cv2.line(panel, (u - edge // 2, v), (u + edge // 2, v), AMBER,
                 max(1, round(unit)), cv2.LINE_AA)
        cv2.line(panel, (u, v - edge // 2), (u, v + edge // 2), AMBER,
                 max(1, round(unit)), cv2.LINE_AA)

    if tcp_px is not None:
        u, v = int(tcp_px[0]), int(tcp_px[1])
        white = (245, 245, 245)
        thin = max(1, round(unit))
        cv2.circle(panel, (u, v), max(2, round(9 * unit)), white, thin, cv2.LINE_AA)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cv2.line(panel, (u + int(dx * 14 * unit), v + int(dy * 14 * unit)),
                     (u + int(dx * 5 * unit), v + int(dy * 5 * unit)), white, thin, cv2.LINE_AA)
    return panel


def draw_banner(frame: np.ndarray, text: str, intensity: float, scale: float = 1.0,
                colour=GREEN) -> np.ndarray:
    """A fading full-frame banner. Used for the one event worth interrupting for.

    A saved success is the only thing that happens without the operator asking
    for it, and it is exactly what they need to know: keep going, that one
    counted.
    """
    import cv2

    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity <= 0.01:
        return frame
    height, width = frame.shape[:2]
    border = max(2, round(8 * scale * intensity))
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), colour, border)

    size = 1.1 * scale
    thickness = max(1, round(2 * scale))
    (tw, th), _ = cv2.getTextSize(text, FONT_BOLD, size, thickness)
    x, y = (width - tw) // 2, round(height * 0.16)
    pad = round(18 * scale)
    box = frame[max(0, y - th - pad) : y + pad, max(0, x - pad) : x + tw + pad]
    if box.size:
        box[:] = cv2.addWeighted(box, 1.0 - 0.65 * intensity,
                                 np.full_like(box, PANEL), 0.65 * intensity, 0)
    cv2.putText(frame, text, (x, y), FONT_BOLD, size, colour, thickness, cv2.LINE_AA)
    return frame


HELP_LINES = (
    ("SPACE", "clutch on and off - the arm only follows while it is on"),
    ("N", "new episode: fresh layout, recording starts"),
    ("S", "save this episode by hand (a success saves itself)"),
    ("D", "discard this episode and reset"),
    ("H", "send the arm home"),
    ("[ ]", "less / more hand sensitivity"),
    ("V", "put the next view on the stage"),
    ("W", "put the wrist view on the stage"),
    ("T", "hide the tile column - the stage takes the whole window"),
    ("?", "this list"),
    ("Q", "quit"),
)

HELP_TIPS = (
    "Stay inside the green outline: that is every place the arm can reach.",
    "The hairline along the top marks frames the tracker lost.",
    "Pause a beat before you descend and before you pinch - stillness is precision.",
    "Use the clutch like lifting a mouse: release, reposition, engage.",
)


def draw_help(panel: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """The key list, drawn over the camera panel while ``?`` is held on."""
    import cv2

    height, width = panel.shape[:2]
    panel = panel.copy()
    panel[:] = cv2.addWeighted(panel, 0.18, np.full_like(panel, PANEL), 0.82, 0)

    logical_h, logical_w = round(height / scale), round(width / scale)
    surface = _Canvas(logical_h, logical_w, PANEL, scale)
    surface.canvas[:] = cv2.resize(panel, (surface.canvas.shape[1], surface.canvas.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
    surface.text("KEYS", (28, 44), 0.62, INK, 1, FONT_BOLD)
    y = 76
    for key, meaning in HELP_LINES:
        surface.text(key, (28, y), 0.52, GREEN, 1, FONT_BOLD)
        surface.text(elide(meaning, logical_w - 130, 0.46), (100, y), 0.46, DIM)
        y += 26
    y += 10
    for tip in HELP_TIPS:
        surface.text(elide(tip, logical_w - 56, 0.44), (28, y), 0.44, FAINT)
        y += 22
    return surface.result()


def panel_geometry(width: int = REFERENCE_WIDTH, height: int = REFERENCE_HEIGHT,
                   scale: float = 1.0, tiles: int = 3) -> dict:
    """Where every piece of the interface goes, in device pixels.

    The single source of truth for the layout: the compositor draws from it and
    the renderer sizes its images from it, so a panel can never be rendered at a
    size the layout does not use.
    """
    ribbon = max(1, round(STRIP_HEIGHT * scale))
    column = max(round(200 * scale), int(width * TILE_FRACTION))
    stage_width = max(1, width - column)
    tiles = max(1, int(tiles))
    return {
        "width": width,
        "height": height,
        "scale": scale,
        "gap": max(1, round(GAP * scale)),
        "strip_height": ribbon,
        "ribbon_height": ribbon,
        "stage_width": stage_width,
        "stage_height": height,
        "column_width": column,
        "tile_height": max(1, height // tiles),
        # Retained names, so callers that still think in terms of the old
        # side-by-side split get the same answer the compositor uses.
        "body_height": height,
        "camera_width": stage_width,
        "sim_width": column,
        "view_height": max(1, height // tiles),
    }


def compose(stage: np.ndarray, tiles, state: HudState,
            width: int = REFERENCE_WIDTH, height: int = REFERENCE_HEIGHT,
            stage_label: str = "YOUR HAND", help_open: bool = False) -> np.ndarray:
    """Assemble the whole interface: one stage, a column of tiles, one ribbon.

    ``tiles`` is a sequence of ``(image, label)`` for the secondary views, top
    to bottom.
    """
    import cv2

    scale = max(0.25, height / REFERENCE_HEIGHT)
    # Not ``tiles or []``: a caller passing a bare image gets a numpy array here,
    # and testing an array for truth raises rather than being falsy.
    tiles = [] if tiles is None else [t for t in tiles if t is not None and t[0] is not None]
    layout = panel_geometry(width, height, scale, tiles=max(1, len(tiles)))
    stage_width = layout["stage_width"] if tiles else width
    column_x = stage_width
    ribbon = layout["ribbon_height"]

    frame = np.empty((height, width, 3), np.uint8)

    # --- the stage -------------------------------------------------------
    view = frame[:, :stage_width]
    view[:] = _fit(stage, height, stage_width)
    if help_open:
        view[:] = draw_help(view, scale)
    else:
        _label(view, stage_label, scale)
    # Last, so the hairline is never hidden under the caption: it is the one
    # mark on the stage that reports a fault.
    if state.tracking_history:
        draw_quality_bar(view, state.tracking_history, scale)

    # --- the tiles -------------------------------------------------------
    if tiles:
        each = height // len(tiles)
        y = 0
        for index, (image, label) in enumerate(tiles):
            bottom = height if index == len(tiles) - 1 else y + each
            tile = frame[y:bottom, column_x:]
            tile[:] = _fit(image, bottom - y, width - column_x)
            _label(tile, label, scale)
            y = bottom
    else:
        frame[:, column_x:] = PANEL

    # --- the ribbon, over the stage --------------------------------------
    band = frame[height - ribbon :, :stage_width]
    cv2.addWeighted(draw_strip(stage_width, state, scale), 0.86, band, 0.14, 0, band)

    if state.flash > 0.01:
        draw_banner(frame, "SUCCESS - EPISODE SAVED", state.flash, scale)
    return frame
