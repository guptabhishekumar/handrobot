"""Live neural retargeting: your fingers driving the LEAP hand at 30 Hz.

Webcam landmarks in, sixteen joint angles out of the network, straight into the
simulated hand's position actuators. The window shows you and the robot hand
side by side, mirror-fashion.

The palm frame is built from the landmarks' own anatomy inside
:func:`handrobot.dexhand.synth.landmarks_to_keypoints`, so the mirrored preview
cannot flip or swap any axis: the same definition produced the training data.
"""

from __future__ import annotations

import time

import numpy as np

from handrobot.config import HandConfig
from handrobot.dexhand.retarget_net import PERSONAL_CHECKPOINT, load_retargeter
from handrobot.dexhand.synth import landmarks_to_keypoints
from handrobot.hands.tracker import HandTracker, Webcam
from handrobot.filters import OneEuroFilter
from handrobot.paths import ASSETS_DIR

MIRROR_XML = ASSETS_DIR / "leap" / "scene_mirror.xml"


def run_dexhand(device: int = 0, hand: str | None = "right") -> int:
    """Open the camera and mirror the operator's fingers on the LEAP hand."""
    import cv2
    import mujoco

    from handrobot.viz.overlay import draw_hand_overlay

    retargeter = load_retargeter()
    which = "trained on YOUR hand" if PERSONAL_CHECKPOINT.exists() else \
        "base network - run with --record to train it on your hand"
    model = mujoco.MjModel.from_xml_path(str(MIRROR_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=560, width=560)

    config = HandConfig(prefer_hand=hand)
    smoother = OneEuroFilter(min_cutoff=1.5, beta=0.6, d_cutoff=1.0)
    last_q = np.zeros(16)
    last_time = time.perf_counter()
    tracked = seen = 0

    with Webcam(device) as camera, HandTracker(
        camera.width, camera.height, config
    ) as tracker:
        while True:
            frame = camera.read()
            if frame is None:
                break
            seen += 1
            now = time.perf_counter()
            dt = min(max(now - last_time, 1e-3), 0.1)
            last_time = now

            pose, landmarks = tracker.detect(frame)
            if pose is not None:
                try:
                    keypoints = landmarks_to_keypoints(
                        pose.landmarks.world, pose.landmarks.handedness
                    )
                except ValueError:
                    pass                 # glitch frame: hold the last pose
                else:
                    tracked += 1
                    target = retargeter(keypoints)
                    last_q = smoother(target, dt)

            data.ctrl[:] = last_q
            for _ in range(6):
                mujoco.mj_step(model, data)
            renderer.update_scene(data, camera="mirror_cam")
            # Flipped so the robot presents as your mirror twin: your thumb and
            # its thumb on the same side of the screen. Without this the hand
            # reads as someone else's, and every motion looks subtly wrong even
            # when the joints are perfectly copied.
            hand_view = cv2.flip(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR), 1)

            preview = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            draw_hand_overlay(preview, landmarks, pose, engaged=pose is not None)
            scale = hand_view.shape[0] / preview.shape[0]
            preview = cv2.resize(preview, (int(preview.shape[1] * scale), hand_view.shape[0]))
            canvas = np.hstack([preview, hand_view])
            rate = 100 * tracked / max(seen, 1)
            cv2.putText(canvas, f"neural retargeting  -  {which}",
                        (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 244), 1,
                        cv2.LINE_AA)
            cv2.putText(canvas, f"{rate:.0f}% tracked  -  q quits",
                        (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (176, 176, 182), 1,
                        cv2.LINE_AA)
            cv2.imshow("handrobot dexhand", canvas)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cv2.destroyAllWindows()
    print(f"tracked {tracked}/{seen} frames ({100 * tracked / max(seen, 1):.0f}%)")
    return 0
