"""Export a handrobot dataset to the Hugging Face LeRobot format.

Written through LeRobot's own ``LeRobotDataset.create`` writer rather than by
imitating its on-disk layout, so the result is valid by construction for
whatever format version the installed ``lerobot`` package speaks, and loads
directly in the LeRobot training stack (or can be pushed to the Hub with
``dataset.push_to_hub()``).

``lerobot`` is an optional dependency: nothing else in handrobot needs it, and
it brings a heavy dependency tree of its own. Install it only for exporting::

    uv pip install lerobot
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from handrobot.data.dataset import list_episodes, load_episode
from handrobot.tasks import TASK_NAMES, get_task


def export_lerobot(
    data_root: Path | str,
    out_root: Path | str,
    repo_id: str,
    fps: float = 30.0,
    robot_type: str = "panda",
    successful_only: bool = True,
    log: bool = True,
) -> dict:
    """Convert every episode under ``data_root`` into a LeRobot dataset."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:  # pragma: no cover - exercised via CLI message
        raise ImportError(
            "the LeRobot export needs the optional 'lerobot' package: "
            "uv pip install lerobot"
        ) from error

    paths = list_episodes(data_root)
    if not paths:
        raise FileNotFoundError(f"no episodes in {data_root}")

    first = load_episode(paths[0])
    cameras = list(first.policy_cameras or first.images)
    state_dim = int(first.states.shape[1])
    action_dim = int(first.actions.shape[1])
    height, width = first.images[cameras[0]][0].shape[:2]

    features = {
        "observation.state": {
            "dtype": "float32", "shape": (state_dim,),
            "names": [f"joint_{i}" for i in range(state_dim)],
        },
        "action": {
            "dtype": "float32", "shape": (action_dim,),
            "names": [f"joint_{i}" for i in range(action_dim)],
        },
    }
    for camera in cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": "video", "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(round(fps)),
        features=features,
        root=Path(out_root),
        robot_type=robot_type,
        use_videos=True,
    )

    exported = skipped = 0
    for path in paths:
        episode = load_episode(path)
        if successful_only and not episode.success:
            skipped += 1
            continue
        task_name = episode.metadata.get("task", TASK_NAMES[0])
        instruction = episode.metadata.get(
            "instruction", get_task(task_name).instruction
        )
        import inspect

        task_is_argument = "task" in inspect.signature(dataset.add_frame).parameters
        for t in range(len(episode)):
            frame = {
                "observation.state": episode.states[t].astype(np.float32),
                "action": episode.actions[t].astype(np.float32),
            }
            for camera in cameras:
                frame[f"observation.images.{camera}"] = episode.images[camera][t]
            # The writer API moved the task between releases: older versions
            # take it inside the frame dict, newer ones as an argument.
            if task_is_argument:
                dataset.add_frame(frame, task=instruction)
            else:
                frame["task"] = instruction
                dataset.add_frame(frame)
        dataset.save_episode()
        exported += 1
        if log:
            print(f"  exported {path.name}  ({len(episode)} frames, task: {task_name})")

    if hasattr(dataset, "finalize"):
        dataset.finalize()
    if log:
        print(f"\n{exported} episodes exported to {out_root} "
              f"({skipped} skipped), repo id {repo_id}")
        print("push with: LeRobotDataset(repo_id, root=...).push_to_hub()")
    return {"episodes": exported, "skipped": skipped, "root": str(out_root)}
