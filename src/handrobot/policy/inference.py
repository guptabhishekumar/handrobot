"""Running a trained ACT policy in the loop.

Two things sit between the network and the robot:

* **Normalisation.** The network sees zero-mean unit-variance states and actions.
  The statistics come from the training set and travel with the checkpoint.
* **Temporal ensembling.** The policy predicts a chunk of future actions at every
  timestep, so each moment is covered by several overlapping predictions made at
  different times. Averaging them with an exponential weight, as in the ACT
  paper, removes the visible jerk that appears when a policy switches from one
  open-loop chunk to the next.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from handrobot.data.dataset import NormalizationStats
from handrobot.policy.act import ACTConfig, ACTPolicy


def prepare_image(image: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    """uint8 HWC RGB -> float (1, 3, size, size) in [0, 1]."""
    import cv2

    if image.shape[0] != size or image.shape[1] != size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(image)).to(device)
    return tensor.permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


class ChunkedActor:
    """Wraps a policy so it can be called once per control step."""

    def __init__(
        self,
        policy: ACTPolicy,
        stats: NormalizationStats,
        device: torch.device,
        image_size: int = 128,
        temporal_ensemble_coeff: float | None = 0.01,
        query_every: int = 1,
    ) -> None:
        self.policy = policy.to(device).eval()
        self.config: ACTConfig = policy.config
        self.stats = stats
        self.device = device
        self.image_size = image_size
        self.temporal_ensemble_coeff = temporal_ensemble_coeff
        self.query_every = max(1, int(query_every))

        self._state_mean = torch.tensor(stats.state_mean, dtype=torch.float32, device=device)
        self._state_std = torch.tensor(stats.state_std, dtype=torch.float32, device=device)
        self._action_mean = np.asarray(stats.action_mean, dtype=np.float64)
        self._action_std = np.asarray(stats.action_std, dtype=np.float64)

        self._pending: list[tuple[int, np.ndarray]] = []
        self._step = 0
        #: Task id fed to a multi-task policy; single-task policies ignore it.
        self.task_id: int = 0

    def reset(self) -> None:
        """Clear the ensembling buffer between episodes."""
        self._pending.clear()
        self._step = 0

    @torch.no_grad()
    def _predict_chunk(self, images: dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        tensors = [
            prepare_image(images[camera], self.image_size, self.device)
            for camera in self.config.cameras
        ]
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        normalized = (state_tensor - self._state_mean) / self._state_std
        task = (
            torch.tensor([self.task_id], device=self.device)
            if self.config.n_tasks > 1
            else None
        )
        predicted = self.policy(tensors, normalized, task=task)["actions"][0].float().cpu().numpy()
        return predicted.astype(np.float64) * self._action_std + self._action_mean

    def act(self, images: dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """Return the joint position command for this control step."""
        if self.temporal_ensemble_coeff is None:
            return self._act_open_loop(images, state)

        chunk = self._predict_chunk(images, state)
        self._pending.append((self._step, chunk))
        # Drop chunks whose horizon no longer covers the current step.
        self._pending = [
            (start, actions)
            for start, actions in self._pending
            if self._step - start < actions.shape[0]
        ]

        candidates = np.stack(
            [actions[self._step - start] for start, actions in self._pending]
        )
        ages = np.arange(len(candidates))
        weights = np.exp(-self.temporal_ensemble_coeff * ages)
        weights /= weights.sum()
        self._step += 1
        return (candidates * weights[:, None]).sum(axis=0)

    def _act_open_loop(self, images: dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """Re-plan every ``query_every`` steps and replay the chunk in between."""
        offset = self._step % self.query_every
        if offset == 0:
            self._pending = [(self._step, self._predict_chunk(images, state))]
        start, chunk = self._pending[-1]
        index = min(self._step - start, chunk.shape[0] - 1)
        self._step += 1
        return chunk[index]


def save_checkpoint(
    path: Path | str,
    policy: ACTPolicy,
    stats: NormalizationStats,
    extra: dict | None = None,
) -> Path:
    """Write weights, architecture and normalisation together.

    Keeping them in one file means a checkpoint can never be loaded with the
    wrong statistics, which would silently produce plausible-looking but wrong
    joint commands.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from handrobot.policy.diffusion import DiffusionChunkPolicy

    torch.save(
        {
            "state_dict": policy.state_dict(),
            "config": policy.config.to_dict(),
            "model_type": (
                "diffusion" if isinstance(policy, DiffusionChunkPolicy) else "act"
            ),
            "stats": stats.to_dict(),
            "extra": extra or {},
        },
        path,
    )
    return path


def load_checkpoint(
    path: Path | str, device: torch.device | str = "cpu"
) -> tuple[ACTPolicy, NormalizationStats, dict]:
    """Rebuild a policy from a checkpoint written by :func:`save_checkpoint`."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("model_type", "act") == "diffusion":
        from handrobot.policy.diffusion import DiffusionChunkPolicy, DiffusionConfig

        config = DiffusionConfig.from_dict(payload["config"])
        config.pretrained_backbone = False
        policy = DiffusionChunkPolicy(config)
    else:
        config = ACTConfig.from_dict(payload["config"])
        # The backbone weights come from the checkpoint, so skip the download.
        config.pretrained_backbone = False
        policy = ACTPolicy(config)
    policy.load_state_dict(payload["state_dict"])
    policy.to(device)
    stats = NormalizationStats.from_dict(payload["stats"])
    return policy, stats, payload.get("extra", {})
