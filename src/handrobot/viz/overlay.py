"""Operator feedback drawn onto the webcam preview.

The overlay is not decoration. It shows the operator the two things that decide
whether a demonstration is any good: whether the hand is being tracked at all,
and where the system thinks the pinch point and hand frame are.
"""

from __future__ import annotations

import numpy as np

from handrobot.hands.types import HandPose, INDEX_TIP, Landmarks, THUMB_TIP

SKELETON = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

GREEN = (120, 230, 120)
GREY = (150, 150, 150)
AMBER = (60, 190, 250)
RED = (70, 70, 240)
WHITE = (245, 245, 245)
AXIS_COLORS = ((80, 80, 250), (80, 220, 80), (250, 160, 60))  # x, y, z in BGR


def _pixels(landmarks: Landmarks, width: int, height: int) -> np.ndarray:
    return (landmarks.image[:, :2] * np.array([width, height])).astype(int)


def hand_is_clipped(landmarks: Landmarks, margin: float = 0.04) -> bool:
    """Whether the hand is touching the edge of the frame.

    The detector degrades badly once part of the hand leaves the picture, and a
    hand held close enough to fill the frame is the commonest cause of poor
    tracking. Worth saying out loud, because the fix is simply to sit back.
    """
    xy = landmarks.image[:, :2]
    return bool(np.any(xy < margin) or np.any(xy > 1.0 - margin))


def draw_hand_overlay(
    image: np.ndarray,
    landmarks: Landmarks | None,
    pose: HandPose | None,
    engaged: bool = False,
) -> np.ndarray:
    """Draw the skeleton, the pinch span and the derived hand frame, in place."""
    import cv2

    height, width = image.shape[:2]
    if landmarks is None:
        cv2.putText(image, "no hand detected", (16, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
        return image

    points = _pixels(landmarks, width, height)
    colour = GREEN if (pose is not None and engaged) else (AMBER if pose is not None else GREY)
    for a, b in SKELETON:
        cv2.line(image, tuple(points[a]), tuple(points[b]), colour, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, tuple(point), 3, colour, -1, cv2.LINE_AA)

    thumb, index = tuple(points[THUMB_TIP]), tuple(points[INDEX_TIP])
    cv2.line(image, thumb, index, WHITE, 2, cv2.LINE_AA)
    midpoint = ((thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2)
    cv2.circle(image, midpoint, 7, WHITE, 2, cv2.LINE_AA)

    if hand_is_clipped(landmarks):
        cv2.putText(image, "hand is leaving the frame - sit back a little",
                    (16, height - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, AMBER, 2, cv2.LINE_AA)

    if pose is not None:
        gap_mm = pose.pinch_distance * 1000
        cv2.putText(image, f"{gap_mm:.0f} mm", (midpoint[0] + 12, midpoint[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        _draw_frame_axes(image, midpoint, pose.rotation)
    else:
        cv2.putText(image, "hand seen, pose unresolved", (16, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, AMBER, 2, cv2.LINE_AA)
    return image


def _draw_frame_axes(image: np.ndarray, origin: tuple[int, int], rotation: np.ndarray,
                     length: int = 48) -> None:
    """Project the hand frame onto the image by dropping the depth component."""
    import cv2

    for axis in range(3):
        direction = rotation[:, axis]
        end = (int(origin[0] + direction[0] * length), int(origin[1] + direction[1] * length))
        cv2.arrowedLine(image, origin, end, AXIS_COLORS[axis], 2, cv2.LINE_AA, tipLength=0.25)


def draw_status_panel(image: np.ndarray, session, info: dict) -> np.ndarray:
    """Draw the teleop heads-up display, in place."""
    import cv2

    stats = session.stats
    command = info.get("command")
    ik = info.get("ik")

    engaged = bool(command is not None and command.engaged)
    lines = [
        ("CLUTCH ENGAGED" if engaged else "clutch released", GREEN if engaged else GREY),
        (f"recording: {'yes' if session.recording else 'no'}  "
         f"steps: {session.episode_steps}", WHITE),
        (f"saved {stats.episodes_saved}  success {stats.successes}  "
         f"discarded {stats.episodes_discarded}", WHITE),
        (f"tracking {100 * stats.tracking_rate:.0f}%   {stats.fps:.0f} fps   "
         f"sensitivity {session.mapper.position_gain:.1f}x   "
         f"regrabs {session.mapper.tracking_gaps}",
         WHITE if stats.tracking_rate > 0.7 else AMBER),
    ]
    if command is not None:
        lines.append((
            f"target xyz [{command.position[0]:+.3f} {command.position[1]:+.3f} "
            f"{command.position[2]:+.3f}]  jaw {np.degrees(command.jaw_azimuth):+.0f} deg "
            f"/ {command.jaw_gap * 1000:.0f} mm", WHITE))
    if ik is not None:
        colour = WHITE if ik.ok else RED
        lines.append((f"ik {ik.position_error * 1000:.1f} mm / {ik.orientation_error:.2f} rad"
                      f"{'' if ik.ok else '  UNREACHABLE'}", colour))

    if getattr(session.mapper, "saturated", False):
        lines.append((
            "AT THE EDGE OF REACH - move your hand back towards the middle", RED))
    alignment = info.get("alignment")
    if alignment is not None:
        planar_mm = alignment["cube_planar"] * 1000
        # Green once the gripper is close enough that the jaws will actually
        # close on the cube rather than beside it.
        colour = GREEN if planar_mm < 12 else (AMBER if planar_mm < 30 else GREY)
        lines.append((
            f"cube: {planar_mm:5.0f} mm across, {alignment['cube_height'] * 1000:+5.0f} mm up"
            f"   bin: {alignment['bin_planar'] * 1000:5.0f} mm", colour))

        # Turn the distance into an instruction. Camera axes are
        # (right, down, forward); the preview is mirrored, so "right" on screen
        # is the operator's right.
        move = alignment["hand_move"] * 1000
        parts = []
        if abs(move[0]) > 5:
            parts.append(f"{'RIGHT' if move[0] > 0 else 'LEFT'} {abs(move[0]):3.0f}")
        if abs(move[1]) > 5:
            parts.append(f"{'DOWN' if move[1] > 0 else 'UP'} {abs(move[1]):3.0f}")
        if abs(move[2]) > 5:
            parts.append(f"{'FORWARD' if move[2] > 0 else 'BACK'} {abs(move[2]):3.0f}")
        instruction = ("move hand: " + "  ".join(parts) + " mm") if parts else \
            f"hand is lined up with the {alignment['goal_name']}"
        lines.append((instruction, GREEN if not parts else WHITE))

    top = stats.top_rejection
    if top is not None and stats.tracking_rate < 0.9:
        reason, count = top
        lines.append((f"most frames lost to: {reason} ({count})", AMBER))

    lines.append((session.message, AMBER))
    lines.append((
        "space clutch | n new | s save | d discard | h home | [ ] speed | v view | q quit",
        GREY))

    box_height = 22 * len(lines) + 14
    panel = image[: box_height, : 560].copy()
    cv2.rectangle(image, (0, 0), (560, box_height), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.25, image[:box_height, :560], 0.75, 0, image[:box_height, :560])
    for i, (text, colour) in enumerate(lines):
        cv2.putText(image, text, (12, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, colour, 1, cv2.LINE_AA)
    return image
