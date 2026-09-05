"""The operator's region of interest, drawn where the operator is looking.

Teleoperation through a webcam fails in three ways that all look identical from
the operator's side -- the arm simply stops responding:

1. the hand has moved somewhere the arm cannot follow (the command saturates);
2. the hand has drifted to the edge of the frame, where the detector degrades;
3. the hand is too close to or too far from the camera for the depth fit.

None of these are visible in the camera image, so this module draws them into
it. The reach envelope is not an illustration: it is the exact set of hand
positions that map inside the arm's reachable region under the mapping that is
running right now, projected through the same pinhole model that produced the
hand pose in the first place. Standing inside the outline is a guarantee, not
a hint.

Geometry
--------
The mapping from hand to robot is affine while the clutch holds::

    r = r0 + M (h - h0) g

with ``M`` the camera-to-robot map, ``g`` the position gain, and ``(h0, r0)``
the anchors captured when the clutch engaged. Inverting it turns the robot's
reachable set into a set of hand positions::

    h = h0 + M^-1 (r - r0) / g

``M`` sends camera-forward to robot-x, so a slice of the workspace at constant
robot x is a set at constant hand depth -- exactly a plane parallel to the
image, which is why the envelope can be drawn as a flat outline at all rather
than as a perspective cage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GREEN = (120, 225, 145)
AMBER = (70, 190, 250)
RED = (80, 80, 245)
FAINT = (110, 110, 118)

#: Grid used to trace the envelope. Fine enough that the outline is smooth on
#: screen, coarse enough to rebuild in well under a millisecond.
GRID_Y = 64
GRID_Z = 40


def envelope_polygons(
    workspace,
    camera_to_robot: np.ndarray,
    hand_anchor: np.ndarray,
    robot_anchor: np.ndarray,
    gain: float,
    plane_x: float,
    intrinsics,
) -> list[np.ndarray]:
    """Pixel outlines of every hand position that maps inside the workspace.

    Returns one polygon per connected region -- a cylindrical sector sliced
    close to the shoulder genuinely does have a hole in the middle, and drawing
    it as one blob would promise reach that is not there.
    """
    import cv2

    gain = float(max(gain, 1e-6))
    inverse = np.linalg.inv(np.asarray(camera_to_robot, dtype=float))
    hand_anchor = np.asarray(hand_anchor, dtype=float)
    robot_anchor = np.asarray(robot_anchor, dtype=float)

    low, high = np.asarray(workspace.low, dtype=float), np.asarray(workspace.high, dtype=float)
    # ``low``/``high`` are corners of the sector, not of its bounding box: the
    # widest part of a sector is at its outer radius, on both sides of centre.
    # Scanning between the corners as given would miss most of the left half.
    span_y = max(abs(low[1]), abs(high[1]))
    y_edges = np.linspace(-span_y, span_y, GRID_Y)
    z_edges = np.linspace(min(low[2], high[2]), max(low[2], high[2]), GRID_Z)

    mask = np.zeros((GRID_Z, GRID_Y), np.uint8)
    for row, z in enumerate(z_edges):
        for column, y in enumerate(y_edges):
            if workspace.contains(np.array([plane_x, y, z])):
                mask[row, column] = 255
    if not mask.any():
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[np.ndarray] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        points = contour.reshape(-1, 2).astype(float)
        # Grid indices back to metres, then through the inverse mapping.
        y = np.interp(points[:, 0], np.arange(GRID_Y), y_edges)
        z = np.interp(points[:, 1], np.arange(GRID_Z), z_edges)
        robot = np.stack([np.full_like(y, plane_x), y, z], axis=1)
        hand = hand_anchor + (robot - robot_anchor) @ inverse.T / gain

        depth = hand[:, 2]
        if np.any(depth <= 1e-3):
            continue
        u = intrinsics.cx + intrinsics.fx * hand[:, 0] / depth
        v = intrinsics.cy + intrinsics.fy * hand[:, 1] / depth
        polygons.append(np.stack([u, v], axis=1))
    return polygons


@dataclass
class ReachEnvelope:
    """Caches the envelope, which only changes when the mapping does.

    Rebuilding it costs a few thousand membership tests. The mapping it depends
    on changes when the operator engages the clutch, changes sensitivity, or
    pushes into the boundary -- a few times a second at worst, not thirty.
    """

    workspace: object
    camera_to_robot: np.ndarray
    intrinsics: object
    _key: tuple | None = None
    _polygons: list[np.ndarray] | None = None

    def polygons(self, hand_anchor, robot_anchor, gain: float, plane_x: float):
        if hand_anchor is None or robot_anchor is None:
            return []
        key = (
            tuple(np.round(np.asarray(hand_anchor, dtype=float), 3)),
            tuple(np.round(np.asarray(robot_anchor, dtype=float), 3)),
            round(float(gain), 2),
            round(float(plane_x), 3),
            (self.intrinsics.width, self.intrinsics.height,
             round(self.intrinsics.fx, 3), round(self.intrinsics.fy, 3)),
        )
        if key != self._key:
            self._key = key
            self._polygons = envelope_polygons(
                self.workspace, self.camera_to_robot, hand_anchor, robot_anchor,
                gain, plane_x, self.intrinsics,
            )
        return self._polygons or []


def clip_to_frame(polygon: np.ndarray, width: int, height: int,
                  inset: float = 2.0, bottom_inset: float = 0.0) -> np.ndarray:
    """Cut a polygon down to the picture it is drawn on (Sutherland-Hodgman).

    The rasteriser would clip the *lines* anyway, but what the operator sees
    then is an outline that runs off the edge of the image and stops, which
    reads as a broken drawing rather than as a region continuing out of view.
    Clipping the shape instead keeps it a closed figure inside the frame, and
    the edges that lie along the border are exactly the sides where the region
    carries on past what the camera can see.
    """
    left, top = inset, inset
    right = width - 1 - inset
    # ``bottom_inset`` is the strip the status ribbon lies over. Drawing under
    # it is drawing where the operator cannot see, and an outline that vanishes
    # into the ribbon looks like an outline that stopped.
    bottom = height - 1 - inset - bottom_inset
    planes = (
        (lambda p: p[0] >= left, (1.0, 0.0, left)),
        (lambda p: p[0] <= right, (1.0, 0.0, right)),
        (lambda p: p[1] >= top, (0.0, 1.0, top)),
        (lambda p: p[1] <= bottom, (0.0, 1.0, bottom)),
    )

    points = [np.asarray(p, dtype=float) for p in np.asarray(polygon, dtype=float)]
    for inside, (ax, ay, value) in planes:
        if not points:
            return np.empty((0, 2))
        output = []
        for index, current in enumerate(points):
            previous = points[index - 1]
            if inside(current):
                if not inside(previous):
                    output.append(_cross(previous, current, ax, ay, value))
                output.append(current)
            elif inside(previous):
                output.append(_cross(previous, current, ax, ay, value))
        points = output
    return np.array(points) if points else np.empty((0, 2))


def _cross(a: np.ndarray, b: np.ndarray, ax: float, ay: float, value: float) -> np.ndarray:
    """Where the segment a-b meets the axis-aligned line ``ax*x + ay*y = value``."""
    denominator = ax * (b[0] - a[0]) + ay * (b[1] - a[1])
    if abs(denominator) < 1e-9:
        return b.copy()
    t = (value - (ax * a[0] + ay * a[1])) / denominator
    return a + (b - a) * float(np.clip(t, 0.0, 1.0))


def draw_envelope(image: np.ndarray, polygons, saturated: bool = False,
                  engaged: bool = True, hide_above: float = 0.92,
                  bottom_inset: float = 0.0) -> np.ndarray:
    """Outline the reachable region on the webcam preview, in place.

    Saturation changes the line *and* the dash pattern. Colour alone carries
    this warning badly: green against amber is the commonest form of colour
    blindness, and the operator has to be able to tell at a glance whether the
    arm has stopped because they left the region or for some other reason.
    """
    import cv2

    if not polygons:
        return image
    height, width = image.shape[:2]
    colour = AMBER if saturated else (GREEN if engaged else FAINT)
    thickness = 2 if engaged else 1
    for polygon in polygons:
        clipped = clip_to_frame(polygon, width, height, bottom_inset=bottom_inset)
        if len(clipped) < 3:
            continue
        # A region that fills the view is not telling the operator anything: the
        # answer to "where may I move" is "anywhere you can be seen", and an
        # outline hugging the border only crowds the picture.
        area = 0.5 * abs(float(np.dot(clipped[:, 0], np.roll(clipped[:, 1], -1))
                               - np.dot(clipped[:, 1], np.roll(clipped[:, 0], -1))))
        if area > hide_above * width * max(1.0, height - bottom_inset):
            continue
        points = np.round(clipped).astype(np.int32)
        if not saturated:
            cv2.polylines(image, [points], True, colour, thickness, cv2.LINE_AA)
            continue
        closed = np.vstack([points, points[:1]])
        for start, end in zip(closed[:-1], closed[1:]):
            length = float(np.hypot(*(end - start)))
            steps = max(1, int(length // 10))
            for step in range(0, steps, 2):
                a = start + (end - start) * (step / steps)
                b = start + (end - start) * (min(step + 1, steps) / steps)
                cv2.line(image, tuple(a.astype(int)), tuple(b.astype(int)),
                         colour, thickness, cv2.LINE_AA)
    return image


def draw_frame_margin(image: np.ndarray, margin: float = 0.04,
                      clipped: bool = False) -> np.ndarray:
    """Mark the border where the detector starts to degrade, in place.

    The same margin :func:`handrobot.viz.overlay.hand_is_clipped` warns about,
    drawn before it is crossed rather than after.
    """
    import cv2

    height, width = image.shape[:2]
    x, y = int(width * margin), int(height * margin)
    cv2.rectangle(image, (x, y), (width - x, height - y),
                  RED if clipped else (60, 60, 66), 1, cv2.LINE_AA)
    return image


def draw_depth_band(image: np.ndarray, distance: float | None,
                    band: tuple[float, float], limits: tuple[float, float]) -> np.ndarray:
    """A vertical gauge of how far the hand is from the camera, in place.

    Depth is the weakest axis of a single camera, and its error grows with the
    square of the distance while the hand shrinks in the frame. There is a band
    where both are tolerable; this says whether the operator is in it.
    """
    import cv2

    height, width = image.shape[:2]
    x = width - 34
    top, bottom = int(height * 0.18), int(height * 0.72)
    cv2.rectangle(image, (x, top), (x + 12, bottom), (60, 60, 66), 1, cv2.LINE_AA)

    def position(value: float) -> int:
        span = max(limits[1] - limits[0], 1e-6)
        t = float(np.clip((value - limits[0]) / span, 0.0, 1.0))
        return int(bottom - t * (bottom - top))

    good_low, good_high = position(band[0]), position(band[1])
    cv2.rectangle(image, (x + 1, good_high), (x + 11, good_low), (40, 70, 50), -1)

    if distance is None:
        cv2.putText(image, "?", (x - 4, bottom + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, FAINT, 1, cv2.LINE_AA)
        return image

    y = position(distance)
    inside = band[0] <= distance <= band[1]
    colour = GREEN if inside else AMBER
    cv2.line(image, (x - 5, y), (x + 17, y), colour, 2, cv2.LINE_AA)
    cv2.putText(image, f"{distance:.2f} m", (x - 62, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, colour, 1, cv2.LINE_AA)
    if not inside:
        word = "MOVE BACK" if distance < band[0] else "COME CLOSER"
        cv2.putText(image, word, (x - 108, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, AMBER, 1, cv2.LINE_AA)
    return image
