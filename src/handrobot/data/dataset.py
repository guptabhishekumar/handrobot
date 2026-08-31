"""On-disk demonstration storage and the chunked dataset ACT trains on.

Each episode is a single ``.npz`` file. Images are JPEG-encoded per frame before
being stored, which keeps a 200-step two-camera episode around 1.5 MB instead of
20 MB while avoiding a video-codec dependency in the training path.

Layout::

    data/<name>/
        meta.json                  # cameras, control rate, per-episode index
        episodes/episode_000000.npz
        episodes/episode_000001.npz
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

JPEG_QUALITY = 92


def _encode(image: np.ndarray) -> np.ndarray:
    import cv2

    ok, buffer = cv2.imencode(
        ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.reshape(-1)


def _decode(buffer: np.ndarray) -> np.ndarray:
    import cv2

    image = cv2.imdecode(np.asarray(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("JPEG decoding failed")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@dataclass
class Episode:
    """One demonstration, fully decoded."""

    states: np.ndarray
    """(T, 6) measured joint positions."""

    actions: np.ndarray
    """(T, 6) commanded joint positions."""

    images: dict[str, list[np.ndarray]]
    """Camera name to a list of T RGB frames."""

    success: bool
    source: str
    policy_cameras: list[str] = field(default_factory=list)
    """Subset of :attr:`images` a policy is allowed to see.

    Teleoperated episodes also carry the operator's webcam, which is there for
    the film and must never reach the network -- a policy that could see the
    human's hand would learn to read it instead of the scene.
    """

    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.states.shape[0])


class EpisodeWriter:
    """Accumulates one episode in memory and writes it on :meth:`finish`."""

    def __init__(
        self,
        root: Path | str,
        cameras: Sequence[str],
        source: str,
        policy_cameras: Sequence[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.episodes_dir = self.root / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.cameras = list(cameras)
        self.policy_cameras = list(policy_cameras) if policy_cameras is not None else list(cameras)
        unknown = set(self.policy_cameras) - set(self.cameras)
        if unknown:
            raise ValueError(f"policy cameras not being recorded: {sorted(unknown)}")
        self.source = source
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._images: dict[str, list[np.ndarray]] = {c: [] for c in self.cameras}

    def add(self, state: np.ndarray, action: np.ndarray, images: dict[str, np.ndarray]) -> None:
        """Record one control step: what was measured, and what was commanded."""
        missing = set(self.cameras) - set(images)
        if missing:
            raise ValueError(f"missing camera images: {sorted(missing)}")
        self._states.append(np.asarray(state, dtype=np.float32).copy())
        self._actions.append(np.asarray(action, dtype=np.float32).copy())
        for camera in self.cameras:
            self._images[camera].append(_encode(images[camera]))

    def __len__(self) -> int:
        return len(self._states)

    def discard(self) -> None:
        self._states.clear()
        self._actions.clear()
        for frames in self._images.values():
            frames.clear()

    def finish(self, success: bool, metadata: dict | None = None) -> Path:
        """Write the episode to disk and return its path."""
        if not self._states:
            raise RuntimeError("cannot write an empty episode")

        index = _next_index(self.episodes_dir)
        path = self.episodes_dir / f"episode_{index:06d}.npz"
        payload: dict[str, np.ndarray] = {
            "states": np.stack(self._states),
            "actions": np.stack(self._actions),
        }
        for camera in self.cameras:
            frames = self._images[camera]
            payload[f"image__{camera}"] = np.concatenate(frames)
            payload[f"offsets__{camera}"] = np.cumsum([0] + [len(f) for f in frames])
        np.savez_compressed(
            path,
            **payload,
            success=np.array(bool(success)),
            source=np.array(self.source),
            cameras=np.array(self.cameras),
            policy_cameras=np.array(self.policy_cameras),
            metadata=np.array(json.dumps(metadata or {})),
        )
        self.discard()
        _refresh_meta(self.root)
        return path


def _next_index(episodes_dir: Path) -> int:
    existing = sorted(episodes_dir.glob("episode_*.npz"))
    if not existing:
        return 0
    return int(existing[-1].stem.split("_")[-1]) + 1


def list_episodes(root: Path | str) -> list[Path]:
    """All episode files under a dataset root, in recording order."""
    return sorted((Path(root) / "episodes").glob("episode_*.npz"))


def load_episode(path: Path | str, decode_images: bool = True) -> Episode:
    """Read one episode from disk."""
    with np.load(path, allow_pickle=False) as raw:
        cameras = [str(c) for c in raw["cameras"]]
        images: dict[str, list[np.ndarray]] = {}
        if decode_images:
            for camera in cameras:
                blob = raw[f"image__{camera}"]
                offsets = raw[f"offsets__{camera}"]
                images[camera] = [
                    _decode(blob[offsets[i] : offsets[i + 1]]) for i in range(len(offsets) - 1)
                ]
        policy_cameras = (
            [str(c) for c in raw["policy_cameras"]]
            if "policy_cameras" in raw.files
            else list(cameras)
        )
        return Episode(
            states=raw["states"],
            actions=raw["actions"],
            images=images,
            success=bool(raw["success"]),
            source=str(raw["source"]),
            policy_cameras=policy_cameras,
            metadata=json.loads(str(raw["metadata"])),
        )


def _refresh_meta(root: Path | str) -> dict:
    """Rewrite ``meta.json`` from the episodes actually present on disk."""
    root = Path(root)
    entries = []
    for path in list_episodes(root):
        with np.load(path, allow_pickle=False) as raw:
            entries.append(
                {
                    "file": path.name,
                    "length": int(raw["states"].shape[0]),
                    "success": bool(raw["success"]),
                    "source": str(raw["source"]),
                    "cameras": [str(c) for c in raw["cameras"]],
                    "policy_cameras": (
                        [str(c) for c in raw["policy_cameras"]]
                        if "policy_cameras" in raw.files
                        else [str(c) for c in raw["cameras"]]
                    ),
                }
            )
    meta = {
        "num_episodes": len(entries),
        "num_frames": sum(e["length"] for e in entries),
        "num_successful": sum(1 for e in entries if e["success"]),
        "cameras": entries[0]["cameras"] if entries else [],
        "policy_cameras": entries[0]["policy_cameras"] if entries else [],
        "episodes": entries,
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def read_meta(root: Path | str) -> dict:
    """Load ``meta.json``, regenerating it if it is missing."""
    path = Path(root) / "meta.json"
    if not path.exists():
        return _refresh_meta(root)
    return json.loads(path.read_text())


@dataclass
class NormalizationStats:
    """Per-dimension mean and standard deviation for states and actions."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {k: np.asarray(v).tolist() for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, payload: dict) -> "NormalizationStats":
        return cls(**{k: np.asarray(v, dtype=np.float32) for k, v in payload.items()})


class DemoDataset:
    """Chunked action-prediction dataset.

    Item ``i`` is a single timestep: the observation at that step, plus the next
    ``chunk_size`` actions. Chunks that run past the end of an episode are padded
    with the final action and flagged, so the loss can ignore them.

    Episodes are decoded once and held in memory. A hundred 200-step episodes at
    two 128x128 cameras is roughly 2 GB decoded, which fits comfortably.
    """

    def __init__(
        self,
        root: Path | str,
        chunk_size: int,
        cameras: Sequence[str] | None = None,
        successful_only: bool = True,
        episode_indices: Sequence[int] | None = None,
    ) -> None:
        self.root = Path(root)
        self.chunk_size = int(chunk_size)

        paths = list_episodes(self.root)
        if not paths:
            raise FileNotFoundError(f"no episodes in {self.root}")

        self.episodes: list[Episode] = []
        for i, path in enumerate(paths):
            if episode_indices is not None and i not in episode_indices:
                continue
            episode = load_episode(path)
            if successful_only and not episode.success:
                continue
            self.episodes.append(episode)

        if not self.episodes:
            raise ValueError(
                f"no usable episodes in {self.root} "
                f"(successful_only={successful_only}, {len(paths)} on disk)"
            )

        # Default to the cameras the recording marked as policy inputs, never to
        # every stream present -- see Episode.policy_cameras.
        self.cameras = (
            list(cameras)
            if cameras
            else sorted(self.episodes[0].policy_cameras or self.episodes[0].images)
        )
        for episode in self.episodes:
            missing = set(self.cameras) - set(episode.images)
            if missing:
                raise ValueError(f"an episode is missing cameras {sorted(missing)}")
        self._index: list[tuple[int, int]] = [
            (e, t) for e, episode in enumerate(self.episodes) for t in range(len(episode))
        ]
        # Task conditioning. Episodes recorded before tasks existed carry no
        # task metadata and default to the first task; a dataset is multi-task
        # only if any episode says so explicitly.
        from handrobot.tasks import TASK_NAMES, task_id

        self.episode_tasks = [
            task_id(e.metadata.get("task", TASK_NAMES[0])) for e in self.episodes
        ]
        self.n_tasks = len(TASK_NAMES) if len(set(self.episode_tasks)) > 1 else 1
        self.stats = self.compute_stats()

    def __len__(self) -> int:
        return len(self._index)

    def compute_stats(self) -> NormalizationStats:
        """Mean and standard deviation over every frame in the dataset."""
        states = np.concatenate([e.states for e in self.episodes])
        actions = np.concatenate([e.actions for e in self.episodes])
        return NormalizationStats(
            state_mean=states.mean(0),
            state_std=np.maximum(states.std(0), 1e-3),
            action_mean=actions.mean(0),
            action_std=np.maximum(actions.std(0), 1e-3),
        )

    def episode_boundaries(self) -> list[tuple[int, int]]:
        """``(start, end)`` index ranges of each episode within the flat index."""
        bounds, start = [], 0
        for episode in self.episodes:
            bounds.append((start, start + len(episode)))
            start += len(episode)
        return bounds

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        episode_index, t = self._index[i]
        episode = self.episodes[episode_index]
        length = len(episode)

        end = min(t + self.chunk_size, length)
        chunk = episode.actions[t:end]
        pad = self.chunk_size - chunk.shape[0]
        is_pad = np.zeros(self.chunk_size, dtype=bool)
        if pad > 0:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, axis=0)])
            is_pad[self.chunk_size - pad :] = True

        sample: dict[str, np.ndarray] = {
            "state": episode.states[t],
            "action": chunk.astype(np.float32),
            "is_pad": is_pad,
            "task": np.int64(self.episode_tasks[episode_index]),
        }
        for camera in self.cameras:
            sample[f"image.{camera}"] = episode.images[camera][t]
        return sample

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        for i in range(len(self)):
            yield self[i]
