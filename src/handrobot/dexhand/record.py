"""Recording the operator's own hand, and training the retargeter on it.

The synthetic hand generator is a cartoon of a hand. It gets the network into
the right neighbourhood, but the poses a particular person makes, through their
particular camera, are a distribution no generator matches -- and any residual
mismatch shows up live as fingers doing almost-but-not-quite what the operator
did. The cure is direct: record the operator for a minute, moving through
everything, and train on exactly what came through the pipeline. Whatever
quirks the camera, the tracker or the frame construction have, the training
data has them too, so the network cannot be surprised by them.

The recording stores the keypoints *after* the live preprocessing, which is
what makes that guarantee airtight.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from handrobot.config import HandConfig
from handrobot.dexhand.synth import landmarks_to_keypoints
from handrobot.hands.tracker import HandTracker, Webcam

from handrobot.paths import DATA_DIR

RECORDING = DATA_DIR / "hand_poses.npz"

#: The guided sequence. Each prompt holds for a few seconds; together they
#: cover the pose space the network needs.
PROMPTS = (
    "open your hand wide, fingers spread",
    "slowly close into a fist ... and open again",
    "pinch thumb and index, open, pinch again",
    "point with your index finger",
    "curl just your middle and ring fingers",
    "sweep your thumb across your palm",
    "half-close, like holding a cup",
    "wiggle each finger in turn",
    "slowly roll your wrist while half open",
    "anything else your hand does often",
)


def record_hand(seconds_per_prompt: float = 6.0, device: int = 0,
                hand: str | None = "right", out: Path | str = RECORDING) -> dict:
    """Run the guided capture and save the keypoint stream."""
    import cv2

    from handrobot.viz.overlay import draw_hand_overlay

    config = HandConfig(prefer_hand=hand)
    frames: list[np.ndarray] = []
    total = seconds_per_prompt * len(PROMPTS)

    with Webcam(device) as camera, HandTracker(camera.width, camera.height, config) as tracker:
        # The clock only runs while a hand is actually tracked. Camera and
        # model start-up, glancing away, or the tracker losing the hand must
        # not eat the capture budget -- an operator who follows every prompt
        # gets the full minute of usable poses, not whatever was left over.
        tracked = 0.0
        last = time.perf_counter()
        while True:
            frame = camera.read()
            if frame is None:
                break
            if tracked >= total:
                break
            now = time.perf_counter()
            step = min(now - last, 0.1)
            last = now
            prompt = PROMPTS[min(int(tracked / seconds_per_prompt), len(PROMPTS) - 1)]
            remaining = seconds_per_prompt - (tracked % seconds_per_prompt)

            pose, landmarks = tracker.detect(frame)
            if pose is not None:
                try:
                    keypoints = landmarks_to_keypoints(
                        pose.landmarks.world, pose.landmarks.handedness
                    )
                except ValueError:
                    pose = None      # glitch frame: no pose, clock pauses
                else:
                    tracked += step
                    frames.append(keypoints)

            preview = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            draw_hand_overlay(preview, landmarks, pose, engaged=pose is not None)
            cv2.rectangle(preview, (0, 0), (preview.shape[1], 64), (18, 18, 22), -1)
            cv2.putText(preview, prompt, (16, 28), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                        (240, 240, 244), 1, cv2.LINE_AA)
            status = (f"{len(frames)} poses captured   next prompt in {remaining:.0f}s"
                      if pose is not None else
                      f"{len(frames)} poses captured   paused -- show your hand")
            cv2.putText(preview,
                        status + "   q stops early",
                        (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (176, 176, 182), 1,
                        cv2.LINE_AA)
            cv2.imshow("handrobot hand recording", preview)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    cv2.destroyAllWindows()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.stack(frames) if frames else np.zeros((0, 16, 3))
    np.savez_compressed(out, keypoints=data)
    print(f"captured {len(frames)} hand poses -> {out}")
    return {"poses": len(frames), "path": str(out),
            "tracking": tracked / total if total else 0.0}


def load_recording(path: Path | str = RECORDING) -> np.ndarray:
    with np.load(Path(path)) as raw:
        return raw["keypoints"]
