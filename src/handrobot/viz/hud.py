"""The teleoperation interface.

Composed as one frame rather than text stamped over the camera image, because
the operator has to read four things at a glance while their hands are busy:
whether the clutch is on, whether the episode is recording, which way to move,
and how close they are. Anything that takes a second look is a second the
gripper spends drifting.

Layout::

    +---------------------------+------------------+
    |                           |     TOP VIEW     |
    |   YOUR HAND               +------------------+
    |   webcam + tracking       |    CHASE VIEW    |
    +---------------------------+------------------+
    |  status strip: state, guidance, distance bar |
    +----------------------------------------------+

All text is supersampled: drawn at twice the size and averaged down once.
OpenCV's Hershey fonts are stroked vectors with crude anti-aliasing at small
sizes, and this is the difference between text that looks like a debug tool and
text that looks like an interface.
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
FONT_BOLD = 1  # cv2.FONT_HERSHEY_DUPLEX

STRIP_HEIGHT = 128
GAP = 2

#: Text is rendered at this multiple and downsampled once; see the module note.
SS = 2


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


def _text(image, text, origin, scale=0.6, colour=INK, thickness=1, font=FONT):
    import cv2

    cv2.putText(image, text, origin, font, scale, colour, thickness, cv2.LINE_AA)


class _Supersampled:
    """Draw onto a canvas at ``SS`` times the size, then hand back the downscale.

    Every drawing call takes *logical* pixel coordinates; the scaling lives in
    one place so nothing can be half-scaled.
    """

    def __init__(self, height: int, width: int, fill) -> None:
        self.canvas = np.full((height * SS, width * SS, 3), fill, np.uint8)

    def text(self, text, origin, scale=0.6, colour=INK, thickness=1, font=FONT):
        import cv2

        cv2.putText(self.canvas, text, (origin[0] * SS, origin[1] * SS), font,
                    scale * SS, colour, max(1, thickness * SS), cv2.LINE_AA)

    def line(self, a, b, colour, thickness=1):
        import cv2

        cv2.line(self.canvas, (a[0] * SS, a[1] * SS), (b[0] * SS, b[1] * SS),
                 colour, thickness * SS, cv2.LINE_AA)

    def rectangle(self, a, b, colour, thickness=1):
        import cv2

        cv2.rectangle(self.canvas, (a[0] * SS, a[1] * SS), (b[0] * SS, b[1] * SS),
                      colour, -1 if thickness < 0 else thickness * SS)

    def circle(self, centre, radius, colour, thickness=-1):
        import cv2

        cv2.circle(self.canvas, (centre[0] * SS, centre[1] * SS), radius * SS,
                   colour, -1 if thickness < 0 else thickness * SS, cv2.LINE_AA)

    def result(self, height: int, width: int) -> np.ndarray:
        import cv2

        return cv2.resize(self.canvas, (width, height), interpolation=cv2.INTER_AREA)


def _fit(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize into a box, preserving aspect, padding with the panel colour."""
    import cv2

    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.full((height, width, 3), PANEL, np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _label(image: np.ndarray, text: str) -> np.ndarray:
    """Caption a panel in its top-left corner, over a soft scrim."""
    import cv2

    box_w, box_h = 16 + 9 * len(text), 26
    band = image[:box_h, :box_w].copy()
    cv2.rectangle(image, (0, 0), (box_w, box_h), PANEL, -1)
    cv2.addWeighted(band, 0.35, image[:box_h, :box_w], 0.65, 0, image[:box_h, :box_w])
    caption = _Supersampled(box_h, box_w, PANEL)
    caption.canvas[:] = image[:box_h, :box_w].repeat(SS, axis=0).repeat(SS, axis=1)
    caption.text(text, (8, 18), 0.48, (205, 205, 210))
    image[:box_h, :box_w] = caption.result(box_h, box_w)
    return image


def draw_direction(image, origin, move_mm: np.ndarray, box: int = 92) -> None:
    """A compass showing which way the operator should move their hand.

    Camera axes are (right, down, forward). The preview is mirrored, so right on
    screen is the operator's right; depth is drawn as a separate marker because
    there is no honest way to show it on a flat compass.
    """
    import cv2

    cx, cy = origin
    half = box // 2
    cv2.rectangle(image, (cx - half, cy - half), (cx + half, cy + half), EDGE, 1)
    cv2.line(image, (cx - 6, cy), (cx + 6, cy), FAINT, 1)
    cv2.line(image, (cx, cy - 6), (cx, cy + 6), FAINT, 1)

    planar = np.array([move_mm[0], move_mm[1]], dtype=float)
    magnitude = float(np.linalg.norm(planar))
    if magnitude > 4:
        scale = min(half - 10, 14 + magnitude * 0.45) / magnitude
        end = (int(cx + planar[0] * scale), int(cy + planar[1] * scale))
        colour = GREEN if magnitude < 15 else (AMBER if magnitude < 60 else INK)
        cv2.arrowedLine(image, (cx, cy), end, colour, 3, cv2.LINE_AA, tipLength=0.3)

    depth = float(move_mm[2])
    if abs(depth) > 4:
        colour = GREEN if abs(depth) < 15 else (AMBER if abs(depth) < 60 else INK)
        # Camera-forward points at the operator, so a positive depth correction
        # means bring the hand back towards yourself.
        word = "PULL" if depth > 0 else "PUSH"
        _text(image, f"{word} {abs(depth):.0f}", (cx - half, cy + half + 18), 0.46, colour)


def draw_bar(image, origin, size, value: float, limit: float, good: float) -> None:
    """A horizontal distance bar that fills as the gripper closes on the target."""
    import cv2

    x, y = origin
    w, h = size
    cv2.rectangle(image, (x, y), (x + w, y + h), EDGE, 1)
    fraction = float(np.clip(1.0 - value / max(limit, 1e-6), 0.0, 1.0))
    colour = GREEN if value <= good else (AMBER if value <= good * 3 else BLUE)
    if fraction > 0:
        cv2.rectangle(image, (x + 1, y + 1), (x + 1 + int((w - 2) * fraction), y + h - 1),
                      colour, -1)
    marker = x + int((w - 2) * (1.0 - good / max(limit, 1e-6)))
    cv2.line(image, (marker, y - 3), (marker, y + h + 3), DIM, 1)


def draw_strip(width: int, state: HudState) -> np.ndarray:
    """The status strip along the bottom."""
    surface = _Supersampled(STRIP_HEIGHT, width, PANEL)
    surface.line((0, 0), (width, 0), EDGE, 1)

    # --- left: state -----------------------------------------------------
    if state.recording and state.engaged:
        surface.circle((28, 30), 8, RED)
        surface.text("RECORDING", (46, 36), 0.62, INK, 1, FONT_BOLD)
    elif state.engaged:
        surface.circle((28, 30), 8, GREEN)
        surface.text("FOLLOWING", (46, 36), 0.62, INK, 1, FONT_BOLD)
    else:
        surface.circle((28, 30), 8, FAINT)
        surface.text("PAUSED", (46, 36), 0.62, DIM, 1, FONT_BOLD)
        surface.text("press SPACE", (46, 58), 0.46, AMBER)

    if state.engaged:
        step_line = f"step {state.episode_steps}"
        if state.hands_seen > 1 and state.followed_hand:
            step_line += f"  -  {state.followed_hand} hand"
        surface.text(step_line, (46, 58), 0.46, DIM)
    surface.text(f"saved {state.saved}   success {state.successes}", (20, 84), 0.48, DIM)
    surface.text(
        f"{state.tracking * 100:.0f}% tracked   {state.fps:.0f} fps   "
        f"{state.sensitivity:.1f}x",
        (20, 106), 0.44, DIM if state.tracking > 0.7 else AMBER,
    )

    # --- middle: what to do next -----------------------------------------
    x0 = 260
    surface.line((x0 - 24, 12), (x0 - 24, STRIP_HEIGHT - 12), EDGE, 1)

    if state.saturated:
        surface.text("AT THE EDGE OF REACH", (x0, 34), 0.66, RED, 1, FONT_BOLD)
        surface.text("bring your hand back towards the middle", (x0, 60), 0.5, AMBER)
    elif not state.engaged:
        surface.text("hold your hand still, then press SPACE", (x0, 34), 0.6, DIM)
        surface.text("the clutch needs a settled reading to anchor to", (x0, 60), 0.46, FAINT)
    elif state.hand_move is not None:
        move = state.hand_move * 1000
        parts = []
        if abs(move[0]) > 5:
            parts.append(("RIGHT" if move[0] > 0 else "LEFT", abs(move[0])))
        if abs(move[1]) > 5:
            parts.append(("DOWN" if move[1] > 0 else "UP", abs(move[1])))
        if abs(move[2]) > 5:
            parts.append(("PULL" if move[2] > 0 else "PUSH", abs(move[2])))
        if parts:
            text = "   ".join(f"{word} {value:.0f}" for word, value in parts)
            surface.text(text, (x0, 36), 0.72, INK, 1, FONT_BOLD)
            surface.text(f"millimetres, to reach the {state.goal_name}", (x0, 60), 0.46, FAINT)
        else:
            surface.text(f"LINED UP WITH THE {state.goal_name.upper()}", (x0, 36),
                         0.72, GREEN, 1, FONT_BOLD)
            surface.text("lower, then pinch to close" if not state.holding
                         else "open your fingers to release", (x0, 60), 0.5, GREEN)

    if state.goal_distance is not None:
        bar_width = min(420, width - x0 - 170)
        value = state.goal_distance * 1000
        fraction = float(np.clip(1.0 - value / 350.0, 0.0, 1.0))
        colour = GREEN if value <= 25.0 else (AMBER if value <= 75.0 else BLUE)
        surface.rectangle((x0, 74), (x0 + bar_width, 88), EDGE, 1)
        if fraction > 0:
            surface.rectangle((x0 + 1, 75), (x0 + 1 + int((bar_width - 2) * fraction), 87),
                              colour, -1)
        marker = x0 + int((bar_width - 2) * (1.0 - 25.0 / 350.0))
        surface.line((marker, 71), (marker, 91), DIM, 1)
        surface.text(f"{value:.0f} mm", (x0, 106), 0.5, DIM)

    if state.message:
        surface.text(state.message[:64], (x0 + 130, 106), 0.44, AMBER)

    # --- right: compass and keys -----------------------------------------
    strip = surface.result(STRIP_HEIGHT, width)
    if state.hand_move is not None and state.engaged:
        draw_direction(strip, (width - 82, 50), state.hand_move * 1000)
    _text(strip, "SPACE clutch   N next   S save   D drop   [ ] speed   V view   Q quit",
          (width - 560, STRIP_HEIGHT - 8), 0.42, FAINT)
    if state.rejection:
        _text(strip, f"losing frames: {state.rejection}", (width - 560, STRIP_HEIGHT - 26),
              0.42, AMBER)
    return strip


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

    tcp_px = None if tcp is None else project_point(env, camera, tcp, height, width)
    goal_px = None if goal is None else project_point(env, camera, goal, height, width)

    if show_arrow and goal_px is not None and tcp_px is not None:
        distance = float(np.hypot(goal_px[0] - tcp_px[0], goal_px[1] - tcp_px[1]))
        if distance > 18:
            colour = GREEN if distance < 60 else AMBER
            cv2.arrowedLine(panel, (int(tcp_px[0]), int(tcp_px[1])),
                            (int(goal_px[0]), int(goal_px[1])), colour, 2,
                            cv2.LINE_AA, tipLength=min(0.35, 14.0 / distance))

    if tcp_px is not None:
        u, v = int(tcp_px[0]), int(tcp_px[1])
        white = (245, 245, 245)
        cv2.circle(panel, (u, v), 9, white, 1, cv2.LINE_AA)
        cv2.line(panel, (u - 14, v), (u - 5, v), white, 1, cv2.LINE_AA)
        cv2.line(panel, (u + 5, v), (u + 14, v), white, 1, cv2.LINE_AA)
        cv2.line(panel, (u, v - 14), (u, v - 5), white, 1, cv2.LINE_AA)
        cv2.line(panel, (u, v + 5), (u, v + 14), white, 1, cv2.LINE_AA)
    return panel


def compose(camera: np.ndarray, top: np.ndarray | None, front: np.ndarray | None,
            state: HudState, width: int = 1280, height: int = 720) -> np.ndarray:
    """Assemble the whole interface into one frame."""

    body_height = height - STRIP_HEIGHT
    camera_width = int(width * 0.52)
    sim_width = width - camera_width - GAP

    frame = np.full((height, width, 3), PANEL, np.uint8)
    frame[:body_height, :camera_width] = _label(
        _fit(camera, body_height, camera_width), "YOUR HAND"
    )

    views = [v for v in (top, front) if v is not None]
    if views:
        each = (body_height - GAP * (len(views) - 1)) // len(views)
        y = 0
        for view, name in zip(views, ("TOP VIEW", "CHASE VIEW")):
            frame[y : y + each, camera_width + GAP :] = _label(
                _fit(view, each, sim_width), name
            )
            y += each + GAP

    frame[body_height:] = draw_strip(width, state)
    return frame
