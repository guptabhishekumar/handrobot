"""Compose the three-panel film: your hands, the robot copying, the robot alone.

The three panels answer the three questions a viewer has, in order:

1. **What did you do?** The operator's webcam, with the tracking overlay, so it
   is visible that the only input is a bare hand in front of a laptop.
2. **What did the robot do at the time?** The same episode replayed in the
   simulator. Replay is exact: MuJoCo is deterministic, so re-running the
   recorded actions from the recorded seed reproduces the episode frame for
   frame, which means this panel can be re-rendered at any resolution.
3. **What can it do now, by itself?** A trained policy attempting the same task,
   with no hand anywhere.

Panels two and three are rendered fresh rather than upscaled from the 128-pixel
images the policy sees, so the film looks like the task and not like the tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from handrobot.config import Config
from handrobot.data.dataset import Episode, list_episodes, load_episode
from handrobot.sim.env import PickPlaceEnv

PANEL_WIDTH = 640
PANEL_HEIGHT = 720
CAPTION_HEIGHT = 96

BACKGROUND = (14, 14, 16)
CAPTION_COLOR = (238, 238, 240)
SUBTITLE_COLOR = (150, 150, 156)
ACCENT = (90, 200, 130)


@dataclass
class Panel:
    """One column of the film."""

    frames: list[np.ndarray]
    title: str
    subtitle: str


def replay_episode(
    episode: Episode,
    config: Config | None = None,
    camera: str = "hero_cam",
    size: tuple[int, int] = (PANEL_HEIGHT - CAPTION_HEIGHT, PANEL_WIDTH),
) -> list[np.ndarray]:
    """Re-render a recorded episode at an arbitrary resolution.

    Requires the episode metadata to carry the seed it was recorded with; the
    simulator is deterministic, so replaying the recorded actions from that seed
    reproduces the episode exactly.
    """
    seed = episode.metadata.get("seed")
    if seed is None:
        raise ValueError(
            "this episode has no recorded seed, so it cannot be replayed; "
            "re-record with a version that stores one"
        )
    env = PickPlaceEnv(config=config or Config(), render_cameras=())
    try:
        env.reset(seed=int(seed))
        frames = []
        for action in episode.actions:
            frames.append(env.render(camera, *size))
            env.step(action)
        frames.append(env.render(camera, *size))
        return frames
    finally:
        env.close()


def policy_attempt(
    checkpoint: Path | str,
    seed: int,
    config: Config | None = None,
    camera: str = "hero_cam",
    size: tuple[int, int] = (PANEL_HEIGHT - CAPTION_HEIGHT, PANEL_WIDTH),
    device_name: str | None = None,
) -> tuple[list[np.ndarray], bool]:
    """Run a trained policy on one layout and return its frames and outcome."""
    from handrobot.policy.inference import ChunkedActor, load_checkpoint
    from handrobot.policy.train import resolve_device
    from handrobot.rollout import PolicyController, run_episode

    config = config or Config()
    device = resolve_device(device_name)
    policy, stats, _ = load_checkpoint(checkpoint, device)
    actor = ChunkedActor(
        policy, stats, device,
        image_size=config.train.image_size,
        temporal_ensemble_coeff=config.train.temporal_ensemble_coeff,
    )
    env = PickPlaceEnv(config=config)
    try:
        result = run_episode(
            env, PolicyController(actor), seed=seed,
            render_camera=camera, render_size=size,
        )
        return result.frames, result.success
    finally:
        env.close()


def _fit(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize into a box, preserving aspect ratio, padding with the background."""
    import cv2

    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    canvas = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _caption(width: int, index: int, title: str, subtitle: str) -> np.ndarray:
    """Render one panel's caption strip."""
    import cv2

    strip = np.full((CAPTION_HEIGHT, width, 3), BACKGROUND, dtype=np.uint8)
    cv2.putText(strip, f"{index}", (22, 42), cv2.FONT_HERSHEY_DUPLEX, 0.7, ACCENT, 1, cv2.LINE_AA)
    cv2.putText(strip, title, (52, 42), cv2.FONT_HERSHEY_DUPLEX, 0.72,
                CAPTION_COLOR, 1, cv2.LINE_AA)
    cv2.putText(strip, subtitle, (52, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                SUBTITLE_COLOR, 1, cv2.LINE_AA)
    return strip


def compose(panels: list[Panel], fps: int = 30) -> list[np.ndarray]:
    """Lay the panels side by side, holding short ones on their final frame."""
    import cv2

    if not panels:
        raise ValueError("at least one panel is required")
    length = max(len(p.frames) for p in panels)
    if length == 0:
        raise ValueError("panels contain no frames")

    body_height = PANEL_HEIGHT - CAPTION_HEIGHT
    captions = [
        _caption(PANEL_WIDTH, i + 1, panel.title, panel.subtitle)
        for i, panel in enumerate(panels)
    ]

    composed = []
    for t in range(length):
        columns = []
        for panel, caption in zip(panels, captions):
            frame = panel.frames[min(t, len(panel.frames) - 1)] if panel.frames else None
            body = (
                _fit(frame, body_height, PANEL_WIDTH)
                if frame is not None
                else np.full((body_height, PANEL_WIDTH, 3), BACKGROUND, np.uint8)
            )
            columns.append(np.vstack([caption, body]))
        frame = np.hstack(columns)
        # Thin separators between panels.
        for i in range(1, len(panels)):
            cv2.line(frame, (i * PANEL_WIDTH, 0), (i * PANEL_WIDTH, PANEL_HEIGHT),
                     (38, 38, 42), 2)
        composed.append(frame)
    return composed


def build_film(
    data_root: Path | str,
    episode_index: int,
    checkpoint: Path | str | None,
    output: Path | str,
    config: Config | None = None,
    device_name: str | None = None,
    operator_camera: str = "operator_cam",
) -> dict:
    """Build the three-panel film from one recorded episode.

    The third panel is omitted when no checkpoint is given, and the first when
    the episode was recorded without an operator camera (a scripted episode, for
    instance), so the same command works at every stage of the project.
    """
    config = config or Config()
    paths = list_episodes(data_root)
    if not paths:
        raise FileNotFoundError(f"no episodes in {data_root}")
    if not 0 <= episode_index < len(paths):
        raise IndexError(f"episode {episode_index} out of range (0..{len(paths) - 1})")

    episode = load_episode(paths[episode_index])
    panels: list[Panel] = []

    if operator_camera in episode.images:
        panels.append(
            Panel(
                frames=episode.images[operator_camera],
                title="The operator",
                subtitle="one webcam, one bare hand, no controller",
            )
        )

    panels.append(
        Panel(
            frames=replay_episode(episode, config),
            title="The robot copying",
            subtitle=f"{len(episode)} steps at {config.sim.control_hz:.0f} Hz"
            + ("  ·  succeeded" if episode.success else "  ·  did not finish"),
        )
    )

    policy_success = None
    if checkpoint is not None:
        seed = int(episode.metadata.get("seed", 0))
        frames, policy_success = policy_attempt(
            checkpoint, seed, config, device_name=device_name
        )
        panels.append(
            Panel(
                frames=frames,
                title="The robot alone",
                subtitle="same layout, learned policy, no hand"
                + ("  ·  succeeded" if policy_success else "  ·  failed"),
            )
        )

    frames = compose(panels, fps=int(config.sim.control_hz))
    # Hold the last frame so the ending reads.
    frames.extend([frames[-1]] * int(config.sim.control_hz))

    import imageio.v2 as imageio

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(output, frames, fps=int(config.sim.control_hz), quality=9,
                     macro_block_size=1)
    return {
        "output": str(output),
        "panels": [p.title for p in panels],
        "frames": len(frames),
        "episode_success": episode.success,
        "policy_success": policy_success,
    }
