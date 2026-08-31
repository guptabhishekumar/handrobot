"""Training loop for the ACT policy.

Design notes:

* Episodes are split into train and validation sets *by episode*, never by
  frame. Frames within an episode are highly correlated, so a frame-level split
  would leak and report a validation loss that means nothing.
* The pretrained ResNet backbone gets a lower learning rate than the freshly
  initialised transformer, which otherwise destroys the features it inherited.
* Augmentation runs on the accelerator in batch: decoded images already live in
  memory, so there is nothing for dataloader workers to do except copy them.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from handrobot.config import TrainConfig
from handrobot.data.dataset import DemoDataset
from handrobot.policy.act import ACTConfig, ACTPolicy
from handrobot.policy.inference import save_checkpoint


def resolve_device(name: str | None = None) -> torch.device:
    """Pick the best available accelerator unless one is named explicitly."""
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BatchSampler:
    """Yields batches of decoded frames as device tensors.

    Deliberately not a ``torch.utils.data.DataLoader``: the whole dataset is
    already decoded in this process, so worker processes would only add a copy
    of every image per worker.
    """

    def __init__(
        self,
        dataset: DemoDataset,
        indices: np.ndarray,
        batch_size: int,
        device: torch.device,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices)
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        stats = dataset.stats
        self.state_mean = torch.tensor(stats.state_mean, dtype=torch.float32, device=device)
        self.state_std = torch.tensor(stats.state_std, dtype=torch.float32, device=device)
        self.action_mean = torch.tensor(stats.action_mean, dtype=torch.float32, device=device)
        self.action_std = torch.tensor(stats.action_std, dtype=torch.float32, device=device)

    def __len__(self) -> int:
        return max(1, len(self.indices) // self.batch_size)

    def batch(self, chosen: np.ndarray) -> dict:
        samples = [self.dataset[int(i)] for i in chosen]
        images = {
            # .contiguous() is not cosmetic: a permuted view makes the backward
            # pass raise "view size is not compatible with input tensor's size
            # and stride". Augmentation happened to hide this by producing a
            # fresh contiguous tensor, so it only ever bit --no-augment runs.
            camera: torch.from_numpy(
                np.stack([s[f"image.{camera}"] for s in samples])
            ).to(self.device).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
            for camera in self.dataset.cameras
        }
        state = torch.from_numpy(np.stack([s["state"] for s in samples])).to(self.device)
        action = torch.from_numpy(np.stack([s["action"] for s in samples])).to(self.device)
        is_pad = torch.from_numpy(np.stack([s["is_pad"] for s in samples])).to(self.device)
        return {
            "images": images,
            "state": (state - self.state_mean) / self.state_std,
            "action": (action - self.action_mean) / self.action_std,
            "is_pad": is_pad,
        }

    def epochs(self) -> Iterator[dict]:
        """Infinite stream of batches."""
        while True:
            order = self.rng.permutation(self.indices) if self.shuffle else self.indices
            for start in range(0, len(order) - self.batch_size + 1, self.batch_size):
                yield self.batch(order[start : start + self.batch_size])

    def all_batches(self) -> Iterator[dict]:
        """One pass, for validation."""
        for start in range(0, len(self.indices), self.batch_size):
            chunk = self.indices[start : start + self.batch_size]
            if len(chunk):
                yield self.batch(chunk)


def augment(images: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Mild photometric and spatial jitter, applied per camera per batch.

    Kept mild on purpose: the wrist camera's framing carries real information
    about where the gripper is, so aggressive cropping would destroy signal
    rather than add invariance.

    Randomness comes from the global torch generator, which :func:`train` seeds;
    MPS tensors cannot draw from an explicitly passed CPU generator.
    """
    out = {}
    for name, batch in images.items():
        b = batch.shape[0]
        device = batch.device
        brightness = 1.0 + 0.18 * (torch.rand(b, 1, 1, 1, device=device) * 2 - 1)
        contrast = 1.0 + 0.18 * (torch.rand(b, 1, 1, 1, device=device) * 2 - 1)
        mean = batch.mean(dim=(1, 2, 3), keepdim=True)
        jittered = ((batch - mean) * contrast + mean) * brightness

        # Random translation of up to 4% of the frame, via an affine grid.
        shift = 0.04 * (torch.rand(b, 2, device=device) * 2 - 1)
        theta = torch.zeros(b, 2, 3, device=device)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, :, 2] = shift
        grid = F.affine_grid(theta, jittered.shape, align_corners=False)
        shifted = F.grid_sample(jittered, grid, padding_mode="border", align_corners=False)
        out[name] = shifted.clamp_(0.0, 1.0)
    return out


def learning_rate_at(step: int, config: TrainConfig) -> float:
    """Linear warmup then cosine decay to a tenth of the peak."""
    if step < config.warmup_steps:
        return (step + 1) / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def split_episodes(
    dataset: DemoDataset, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split the flat frame index by episode, never within one."""
    bounds = dataset.episode_boundaries()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(bounds))
    n_val = int(round(val_fraction * len(bounds)))
    # Always keep at least one episode on each side when there is more than one.
    if len(bounds) > 1:
        n_val = max(1, min(n_val, len(bounds) - 1))
    else:
        n_val = 0
    val_episodes = set(order[:n_val].tolist())

    train_idx, val_idx = [], []
    for episode, (start, end) in enumerate(bounds):
        target = val_idx if episode in val_episodes else train_idx
        target.extend(range(start, end))
    return np.array(train_idx), np.array(val_idx)


def train(
    data_root: Path | str,
    output_dir: Path | str,
    config: TrainConfig | None = None,
    device_name: str | None = None,
    cameras: tuple[str, ...] | None = None,
    resume: Path | str | None = None,
    log: bool = True,
) -> dict:
    """Train ACT on a demonstration dataset and return the training summary."""
    config = config or TrainConfig()
    device = resolve_device(device_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    dataset = DemoDataset(data_root, chunk_size=config.chunk_size, cameras=cameras)
    train_idx, val_idx = split_episodes(dataset, config.val_fraction, config.seed)
    if log:
        print(
            f"dataset: {len(dataset.episodes)} episodes, {len(dataset)} frames "
            f"({len(train_idx)} train / {len(val_idx)} val), cameras={dataset.cameras}"
        )
        print(f"device: {device}")

    policy_config = ACTConfig(
        state_dim=dataset.episodes[0].states.shape[1],
        action_dim=dataset.episodes[0].actions.shape[1],
        chunk_size=config.chunk_size,
        cameras=tuple(dataset.cameras),
        hidden_dim=config.hidden_dim,
        dim_feedforward=config.dim_feedforward,
        n_heads=config.n_heads,
        n_encoder_layers=config.n_encoder_layers,
        n_decoder_layers=config.n_decoder_layers,
        latent_dim=config.latent_dim,
        dropout=config.dropout,
    )
    policy = ACTPolicy(policy_config).to(device)
    if resume is not None:
        # Weights only. The optimiser state is not carried over, so a resumed run
        # restarts the schedule; that is fine for fine-tuning and wrong for
        # continuing an interrupted run to the same total step count.
        payload = torch.load(Path(resume), map_location=device, weights_only=False)
        policy.load_state_dict(payload["state_dict"])
        if log:
            print(f"resumed weights from {resume} (optimiser state is not restored)")

    backbone_params = [p for n, p in policy.named_parameters() if n.startswith("backbones")]
    other_params = [p for n, p in policy.named_parameters() if not n.startswith("backbones")]
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": config.learning_rate},
            {"params": backbone_params, "lr": config.backbone_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: learning_rate_at(step, config)
    )

    train_sampler = BatchSampler(dataset, train_idx, config.batch_size, device, config.seed)
    val_sampler = (
        BatchSampler(dataset, val_idx, config.batch_size, device, config.seed, shuffle=False)
        if len(val_idx) >= config.batch_size
        else None
    )
    history: list[dict] = []
    stream = train_sampler.epochs()
    best_val = math.inf
    started = time.perf_counter()

    for step in range(config.steps):
        policy.train()
        batch = next(stream)
        images = augment(batch["images"]) if config.augment else batch["images"]
        ordered = [images[c] for c in dataset.cameras]
        losses = policy.compute_loss(
            ordered, batch["state"], batch["action"], batch["is_pad"], config.kl_weight
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % config.log_every == 0 or step == 0:
            entry = {
                "step": step + 1,
                "loss": float(losses["loss"].detach()),
                "l1": float(losses["l1"].detach()),
                "kl": float(losses["kl"].detach()),
                "grad_norm": float(grad_norm),
                "lr": scheduler.get_last_lr()[0],
                "elapsed": time.perf_counter() - started,
            }
            history.append(entry)
            if log:
                print(
                    f"step {entry['step']:6d}/{config.steps}  loss {entry['loss']:8.4f}  "
                    f"l1 {entry['l1']:.4f}  kl {entry['kl']:.4f}  "
                    f"lr {entry['lr']:.2e}  {entry['elapsed']:6.1f}s"
                )

        if (step + 1) % config.save_every == 0 or (step + 1) == config.steps:
            val_l1 = (
                evaluate(policy, val_sampler, dataset.cameras, config.kl_weight)
                if val_sampler is not None
                else None
            )
            if log and val_l1 is not None:
                print(f"    validation L1 {val_l1:.4f}")
            save_checkpoint(
                output_dir / "last.pt", policy, dataset.stats,
                extra={"step": step + 1, "val_l1": val_l1, "train": asdict(config)},
            )
            # With too few episodes to hold any out, there is no validation
            # signal to select on; keep "best" meaning "the one to load" rather
            # than leaving it absent and breaking every downstream command.
            improved = val_l1 is not None and val_l1 < best_val
            if improved or (val_l1 is None and (step + 1) == config.steps):
                best_val = val_l1 if val_l1 is not None else best_val
                save_checkpoint(
                    output_dir / "best.pt", policy, dataset.stats,
                    extra={"step": step + 1, "val_l1": val_l1, "train": asdict(config)},
                )

    summary = {
        "steps": config.steps,
        "episodes": len(dataset.episodes),
        "frames": len(dataset),
        "best_val_l1": None if math.isinf(best_val) else best_val,
        "final_loss": history[-1]["loss"] if history else None,
        "wall_seconds": time.perf_counter() - started,
        "device": str(device),
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


@torch.no_grad()
def evaluate(
    policy: ACTPolicy, sampler: BatchSampler, cameras: list[str], kl_weight: float
) -> float:
    """Mean unpadded L1 action error over the validation frames."""
    policy.eval()
    total, count = 0.0, 0
    for batch in sampler.all_batches():
        ordered = [batch["images"][c] for c in cameras]
        losses = policy.compute_loss(
            ordered, batch["state"], batch["action"], batch["is_pad"], kl_weight
        )
        total += float(losses["l1"]) * batch["state"].shape[0]
        count += batch["state"].shape[0]
    return total / max(count, 1)
