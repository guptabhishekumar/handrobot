"""Live checks that tell the operator whether hand tracking will actually work.

The important one is the sign of the world-landmark depth axis. MediaPipe's
normalised landmarks define z as depth relative to the wrist, increasing away
from the camera. The metric world landmarks use the same axis convention, so the
two should be positively correlated across the twenty-one joints. If they are
not, the depth axis is inverted and every orientation the retargeter produces
would be mirrored. Rather than assume, this measures it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from handrobot.config import HandConfig
from handrobot.hands.types import Landmarks


@dataclass
class DepthSignReport:
    """Result of comparing image-space depth against world-space depth."""

    correlation: float
    samples: int

    @property
    def recommended_sign(self) -> float:
        return 1.0 if self.correlation >= 0 else -1.0

    @property
    def confident(self) -> bool:
        return abs(self.correlation) > 0.5 and self.samples >= 15


def depth_axis_correlation(landmark_history: list[Landmarks]) -> DepthSignReport:
    """Correlate normalised-landmark depth with world-landmark depth.

    Both are measured relative to their own mean, so only the *direction* of the
    axis is being compared, not its scale or origin.
    """
    correlations = []
    for landmarks in landmark_history:
        image_z = landmarks.image[:, 2] - landmarks.image[:, 2].mean()
        world_z = landmarks.world[:, 2] - landmarks.world[:, 2].mean()
        if image_z.std() < 1e-6 or world_z.std() < 1e-6:
            continue
        correlations.append(float(np.corrcoef(image_z, world_z)[0, 1]))
    if not correlations:
        return DepthSignReport(correlation=0.0, samples=0)
    return DepthSignReport(correlation=float(np.mean(correlations)), samples=len(correlations))


def run_handcheck(device: int = 0, seconds: float = 0.0, config: HandConfig | None = None) -> int:
    """Open the camera, overlay the tracking, and report whether it is usable."""
    import cv2

    from handrobot.hands.tracker import HandTracker, Webcam
    from handrobot.viz.overlay import draw_hand_overlay

    config = config or HandConfig()
    history: list[Landmarks] = []
    depths: list[float] = []
    pinches: list[float] = []
    seen = tracked = 0
    started = time.perf_counter()

    print("handcheck: move your hand around the frame; open and close your pinch.")
    print("press q to finish.\n")

    with Webcam(device) as camera, HandTracker(camera.width, camera.height, config) as tracker:
        while True:
            frame = camera.read()
            if frame is None:
                break
            seen += 1
            pose, landmarks = tracker.detect(frame)
            if landmarks is not None:
                history.append(landmarks)
            if pose is not None:
                tracked += 1
                depths.append(pose.depth)
                pinches.append(pose.pinch_distance)

            preview = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            draw_hand_overlay(preview, landmarks, pose, engaged=pose is not None)
            status = f"tracked {tracked}/{seen}"
            if pose is not None:
                status += f"  depth {pose.depth:.2f} m  pinch {pose.pinch_distance * 1000:.0f} mm"
            cv2.putText(preview, status, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (245, 245, 245), 2, cv2.LINE_AA)
            cv2.imshow("handrobot handcheck", preview)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            if seconds and (time.perf_counter() - started) > seconds:
                break

    cv2.destroyAllWindows()

    print(f"frames            {seen}")
    print(f"hand tracked      {tracked} ({100 * tracked / max(seen, 1):.0f}%)")
    if depths:
        print(f"depth             {np.mean(depths):.2f} m "
              f"(min {np.min(depths):.2f}, max {np.max(depths):.2f})")
        print(f"pinch distance    {np.min(pinches) * 1000:.0f}-{np.max(pinches) * 1000:.0f} mm")

    report = depth_axis_correlation(history)
    print(f"depth-axis check  correlation {report.correlation:+.2f} over {report.samples} frames")
    if not report.confident:
        print("  inconclusive - move your hand toward and away from the camera and rerun")
    elif report.recommended_sign > 0:
        print("  depth axis is upright: run teleop without --flip-z")
    else:
        print("  depth axis is inverted: run teleop with --flip-z")

    if tracked < 0.5 * max(seen, 1):
        print("\nTracking is unreliable. More light on your hand, a plainer background, "
              "and keeping your whole hand in frame all help.")
        return 1
    return 0
