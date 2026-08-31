"""Two-webcam stereo depth for the hand.

Monocular depth is the one genuinely weak signal in the pipeline: it comes
from apparent hand size, so it wanders by centimetres. Two cameras a known
distance apart turn depth into geometry instead: the same wrist seen from two
positions shifts horizontally by a disparity, and

    depth = focal_length_px * baseline / disparity

For webcams sitting side by side on the same edge of a screen (parallel
optical axes, no toe-in), this is the standard rectified-stereo relation. It
is exact for ideally parallel cameras and degrades gracefully with small
misalignment; there is deliberately no calibration step, because a second
webcam and a ruler is the entire setup cost this project permits.

The stereo estimate replaces only the depth axis. Left/right and up/down come
from the primary camera exactly as before -- they were never the problem.
"""

from __future__ import annotations

import math


from handrobot.config import HandConfig
from handrobot.hands.types import HandPose


def focal_length_px(width: int, hfov_deg: float) -> float:
    """Horizontal focal length in pixels from the field of view."""
    return width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def triangulate_depth(
    x_primary_px: float,
    x_secondary_px: float,
    fx: float,
    baseline_m: float,
    min_disparity_px: float = 1.0,
) -> float | None:
    """Depth from horizontal disparity between two parallel cameras.

    Returns ``None`` when the disparity is too small to be trustworthy --
    at webcam resolution a sub-pixel disparity would put the hand metres away,
    and a wrong-but-confident depth is worse than falling back to monocular.
    """
    disparity = abs(float(x_primary_px) - float(x_secondary_px))
    if disparity < min_disparity_px:
        return None
    return fx * baseline_m / disparity


class StereoRig:
    """Owns the second camera and tracker, and refines poses from the first.

    The primary pipeline is untouched: frames, tracking, filtering and the
    mapping all run exactly as in the monocular setup. This class only watches
    the same hand from the side and, when both cameras agree they see it,
    replaces the pose's depth with the triangulated one (scaling the camera-
    frame position so the ray direction is preserved).
    """

    #: The wrist landmark: the most stably detected point on the hand.
    ANCHOR = 0

    def __init__(
        self,
        device: int,
        baseline_m: float,
        config: HandConfig | None = None,
    ) -> None:
        from handrobot.hands.tracker import HandTracker, Webcam

        if baseline_m <= 0:
            raise ValueError("the camera baseline must be positive, in metres")
        self.config = config or HandConfig()
        self.baseline_m = float(baseline_m)
        self.camera = Webcam(device)
        self.tracker = HandTracker(self.camera.width, self.camera.height, self.config)
        #: How many poses were refined vs. passed through, for the HUD.
        self.refined = 0
        self.passed_through = 0

    def close(self) -> None:
        self.tracker.close()
        self.camera.close()

    def __enter__(self) -> "StereoRig":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def refine(self, pose: HandPose | None, primary_width: int) -> HandPose | None:
        """Return ``pose`` with a triangulated depth where possible."""
        if pose is None:
            return None
        frame = self.camera.read()
        secondary = None
        if frame is not None:
            secondary, _ = self.tracker.detect(frame)
        if secondary is None:
            self.passed_through += 1
            return pose

        fx = focal_length_px(primary_width, self.config.assumed_hfov_deg)
        depth = triangulate_depth(
            pose.landmarks.image[self.ANCHOR, 0] * primary_width,
            secondary.landmarks.image[self.ANCHOR, 0] * self.camera.width,
            fx,
            self.baseline_m,
        )
        if depth is None or not (self.config.depth_min < depth < self.config.depth_max):
            self.passed_through += 1
            return pose

        # Rescale along the existing camera ray: bearing from the primary
        # camera is trusted, only the distance along it is replaced.
        old_depth = float(pose.palm_position[2])
        if old_depth <= 1e-6:
            self.passed_through += 1
            return pose
        scale = depth / old_depth
        self.refined += 1
        return HandPose(
            palm_position=pose.palm_position * scale,
            rotation=pose.rotation,
            pinch_distance=pose.pinch_distance,
            depth=depth,
            landmarks=pose.landmarks,
            timestamp=pose.timestamp,
        )
