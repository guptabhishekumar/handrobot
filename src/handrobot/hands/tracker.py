"""MediaPipe hand tracking wrapped in something the rest of the code can use.

MediaPipe 1.x removed the old ``mp.solutions`` façade, so this talks to the
Tasks API directly. The tracker owns exactly one concern: frames in, poses out.
"""

from __future__ import annotations

import sys
import time
from typing import Iterator

import numpy as np

from handrobot.config import HandConfig
from handrobot.hands.geometry import CameraIntrinsics, resolve_hand_pose
from handrobot.hands.types import HandPose, Landmarks
from handrobot.paths import HAND_LANDMARKER_TASK


class HandTracker:
    """Detect one hand per frame and convert it to a :class:`HandPose`."""

    def __init__(
        self,
        width: int,
        height: int,
        config: HandConfig | None = None,
        world_z_sign: float = 1.0,
        model_path: str | None = None,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self.config = config or HandConfig()
        self.world_z_sign = world_z_sign
        self.prefer_hand = self.config.prefer_hand
        self._followed: Landmarks | None = None
        #: How many hands the last frame contained, for the on-screen readout.
        self.hands_seen = 0
        self.intrinsics = CameraIntrinsics.from_hfov(width, height, self.config.assumed_hfov_deg)

        path = model_path or str(HAND_LANDMARKER_TASK)
        if not HAND_LANDMARKER_TASK.exists() and model_path is None:
            raise FileNotFoundError(
                f"hand landmarker model missing at {HAND_LANDMARKER_TASK}. "
                "Run: scripts/fetch_models.sh"
            )

        # Force the CPU delegate: MediaPipe's macOS wheels abort inside
        # DrishtiMetalHelper when the Metal graph service is unavailable, which
        # it is in a plain Python process.
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=path,
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.config.num_hands,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._start = time.perf_counter()
        self._last_timestamp_ms = -1
        #: Why the last frame produced no pose, or ``None``. Shown on screen.
        self.last_rejection: str | None = None

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._landmarker.close()

    # -- detection ----------------------------------------------------------

    def _choose(self, candidates: list[Landmarks]) -> Landmarks:
        """Pick which detected hand to follow.

        The operator's other hand is usually on the keyboard, and switching to
        it mid-episode throws the arm across the workspace. So the choice is
        made once and then held: an explicit preference if one was given,
        otherwise the hand nearest the camera when tracking starts -- the one
        deliberately raised -- and after that whichever hand is closest to
        where the followed one was last seen.
        """
        if len(candidates) == 1:
            self._followed = candidates[0]
            return candidates[0]

        if self.prefer_hand is not None:
            wanted = self._mediapipe_label(self.prefer_hand)
            preferred = [c for c in candidates if c.handedness == wanted]
            if preferred:
                chosen = max(preferred, key=lambda c: c.score)
                self._followed = chosen
                return chosen

        if self._followed is not None:
            previous = self._followed.image[:, :2].mean(axis=0)
            nearest = min(
                candidates,
                key=lambda c: float(
                    np.linalg.norm(c.image[:, :2].mean(axis=0) - previous)
                ),
            )
            self._followed = nearest
            return nearest

        # Nothing to go on yet: take the hand that fills more of the frame,
        # which is the one being held up rather than resting on the keyboard.
        chosen = max(
            candidates,
            key=lambda c: float(np.ptp(c.image[:, :2], axis=0).prod()),
        )
        self._followed = chosen
        return chosen

    @staticmethod
    def _mediapipe_label(preference: str) -> str:
        """Translate the operator's own left or right into MediaPipe's label.

        The preview is mirrored so that moving right looks like moving right,
        and MediaPipe labels the hand it sees in that mirrored image. The
        operator's right hand therefore comes back labelled "Left".
        """
        return {"right": "Left", "left": "Right"}[preference.lower()]

    def forget_hand(self) -> None:
        """Stop following whichever hand was being followed."""
        self._followed = None

    def detect(self, frame_rgb: np.ndarray) -> tuple[HandPose | None, Landmarks | None]:
        """Run the detector on one RGB frame.

        Returns:
            ``(pose, landmarks)``. ``landmarks`` is non-``None`` whenever a hand
            was seen at all, even if the pose could not be resolved -- the
            preview overlay uses it to show the operator what went wrong.
        """
        # Cameras do not always deliver the resolution they were asked for, and
        # the pinhole model must describe the frame that actually arrived.
        height, width = frame_rgb.shape[:2]
        if (width, height) != (self.intrinsics.width, self.intrinsics.height):
            self.intrinsics = CameraIntrinsics.from_hfov(
                width, height, self.config.assumed_hfov_deg
            )

        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        now = time.perf_counter() - self._start
        timestamp_ms = int(now * 1000)
        # MediaPipe's video mode requires strictly increasing timestamps.
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            self.last_rejection = "no hand in frame"
            return None, None

        candidates = [
            Landmarks(
                image=np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[i]], dtype=float),
                world=np.array(
                    [[p.x, p.y, p.z] for p in result.hand_world_landmarks[i]], dtype=float
                ),
                handedness=(
                    result.handedness[i][0].category_name
                    if result.handedness and i < len(result.handedness)
                    else "Unknown"
                ),
                score=(
                    float(result.handedness[i][0].score)
                    if result.handedness and i < len(result.handedness)
                    else 0.0
                ),
            )
            for i in range(len(result.hand_landmarks))
        ]
        self.hands_seen = len(candidates)
        landmarks = self._choose(candidates)
        pose, self.last_rejection = resolve_hand_pose(
            landmarks,
            self.intrinsics,
            timestamp=now,
            world_z_sign=self.world_z_sign,
            depth_range=(self.config.depth_min, self.config.depth_max),
        )
        return pose, landmarks


class Webcam:
    """Minimal OpenCV capture that yields mirrored RGB frames.

    The preview is mirrored so that moving your hand right moves it right on
    screen. Landmarks are computed on the mirrored image, so the whole pipeline
    stays in one consistent, intuitive frame.

    Opening the camera is retried, because on macOS the first attempt is what
    triggers the permission prompt: it fails immediately while the dialog is
    still on screen, and succeeds a moment after the user clicks Allow.
    """

    def __init__(
        self,
        device: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        attempts: int = 4,
        retry_delay: float = 1.5,
    ) -> None:
        import cv2

        self._cv2 = cv2
        # AVFoundation is the only backend that works on modern macOS; asking
        # for it by name avoids a slow fall-through and a misleading FFMPEG
        # warning about failing to list devices.
        backends = (
            [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
            if sys.platform == "darwin"
            else [cv2.CAP_ANY]
        )

        self.capture = None
        for attempt in range(attempts):
            for backend in backends:
                capture = cv2.VideoCapture(device, backend)
                if capture.isOpened():
                    ok, _ = capture.read()
                    if ok:
                        self.capture = capture
                        break
                capture.release()
            if self.capture is not None:
                break
            if attempt < attempts - 1:
                print(
                    "waiting for camera access"
                    f" (attempt {attempt + 1} of {attempts})..."
                )
                time.sleep(retry_delay)

        if self.capture is None:
            raise RuntimeError(
                f"could not open camera {device}.\n"
                "  On macOS: System Settings > Privacy & Security > Camera, switch on\n"
                "  your terminal, then QUIT the terminal completely (Cmd+Q) and reopen\n"
                "  it -- macOS only applies camera permission on relaunch.\n"
                "  If the terminal is not listed, run:  tccutil reset Camera\n"
                "  If another app is using the camera, close it first."
            )

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)

        # Trust the first real frame over the driver's reported size: on macOS
        # the two often disagree, and the frame is what the pinhole model needs.
        frame = self.read()
        if frame is None:
            raise RuntimeError(f"camera {device} opened but returned no frames")
        self.height, self.width = frame.shape[:2]

    def read(self) -> np.ndarray | None:
        ok, frame_bgr = self.capture.read()
        if not ok:
            return None
        frame_bgr = self._cv2.flip(frame_bgr, 1)
        return self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def __enter__(self) -> "Webcam":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
