"""Operator feedback drawn onto the webcam preview.

The overlay is not decoration. It shows the operator the two things that decide
whether a demonstration is any good: whether the hand is being tracked at all,
and where the system thinks the pinch point and hand frame are.

Everything else the operator reads lives in :mod:`handrobot.viz.hud`, which
composes the whole interface. Two places drawing the same numbers is two places
to update and one of them will be wrong.
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
    axes: bool = False,
) -> np.ndarray:
    """Draw the skeleton, the pinch span and the derived hand frame, in place.

    Every size here is a multiple of ``unit``, the image width against the
    640-pixel webcam frame it was designed for. The same overlay is drawn on a
    diagnostic preview and on a panel of an 8K interface; a skeleton stroked two
    device pixels wide is emphatic on one and invisible on the other.
    """
    import cv2

    height, width = image.shape[:2]
    unit = max(1.0, width / 640.0)
    stroke = max(1, round(2 * unit))
    if landmarks is None:
        cv2.putText(image, "no hand detected", (round(16 * unit), round(height - 20 * unit)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * unit, RED, stroke, cv2.LINE_AA)
        return image

    points = _pixels(landmarks, width, height)
    colour = GREEN if (pose is not None and engaged) else (AMBER if pose is not None else GREY)
    for a, b in SKELETON:
        cv2.line(image, tuple(points[a]), tuple(points[b]), colour, stroke, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, tuple(point), max(1, round(3 * unit)), colour, -1, cv2.LINE_AA)

    thumb, index = tuple(points[THUMB_TIP]), tuple(points[INDEX_TIP])
    cv2.line(image, thumb, index, WHITE, stroke, cv2.LINE_AA)
    midpoint = ((thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2)
    cv2.circle(image, midpoint, max(2, round(7 * unit)), WHITE, stroke, cv2.LINE_AA)

    if hand_is_clipped(landmarks):
        cv2.putText(image, "your hand is leaving the frame - move back to the middle",
                    (round(16 * unit), round(height - 46 * unit)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * unit, AMBER, stroke, cv2.LINE_AA)

    if pose is not None:
        if axes:
            # Only the diagnostic wants the raw number over the hand; the
            # interface has a jaw gauge, and a figure printed in both places is
            # one more thing to read while aiming.
            cv2.putText(image, f"{pose.pinch_distance * 1000:.0f} mm",
                        (round(midpoint[0] + 12 * unit), round(midpoint[1] - 8 * unit)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * unit, WHITE, max(1, round(unit)),
                        cv2.LINE_AA)
        # The palm frame's three axes used to be drawn here. They are debugging
        # output -- the operator cannot act on them, and three more arrows over
        # a hand that already carries a skeleton, a pinch marker and a
        # correction arrow is three fewer things anyone reads. `handcheck` asks
        # for them, which is where the frame itself is what you are checking.
        if axes:
            _draw_frame_axes(image, midpoint, pose.rotation, length=round(48 * unit), unit=unit)
    else:
        cv2.putText(image, "hand seen, but its pose could not be resolved",
                    (round(16 * unit), round(height - 20 * unit)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * unit, AMBER, stroke, cv2.LINE_AA)
    return image


def _draw_frame_axes(image: np.ndarray, origin: tuple[int, int], rotation: np.ndarray,
                     length: int = 48, unit: float = 1.0) -> None:
    """Project the hand frame onto the image by dropping the depth component."""
    import cv2

    for axis in range(3):
        direction = rotation[:, axis]
        end = (int(origin[0] + direction[0] * length), int(origin[1] + direction[1] * length))
        cv2.arrowedLine(image, origin, end, AXIS_COLORS[axis], max(1, round(2 * unit)),
                        cv2.LINE_AA, tipLength=0.25)
