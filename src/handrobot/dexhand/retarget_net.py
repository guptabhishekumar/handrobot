"""The retargeting network, trained end to end through the hand's kinematics.

Input: sixteen human keypoints in the palm frame, size-normalised.
Output: the LEAP hand's sixteen joint angles.

There is no labelled data anywhere in this. The loss says only "put your
fingertips where the human's are, scaled to your own proportions, without
straining your joints" -- and the differentiable FK lets that instruction flow
back through the network as gradients. The classic optimisation-based
retargeters (DexPilot and its descendants) solve that same objective per frame
at runtime; training a network once amortises the whole thing into a
sub-millisecond forward pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from handrobot.dexhand.fk import KEYPOINT_NAMES, LeapFK
from handrobot.dexhand.synth import sample_hands
from handrobot.paths import ASSETS_DIR

CHECKPOINT = ASSETS_DIR / "cache" / "leap_retargeter.pt"

#: Written by :func:`train_personal`; preferred at load time when present.
PERSONAL_CHECKPOINT = ASSETS_DIR / "cache" / "leap_retargeter_personal.pt"

#: Keypoint indices of the three finger knuckles, identical in the human and
#: robot orderings. They anchor the shared frame both hands are matched in:
#: human landmarks are measured from the wrist, the LEAP palm frame from its
#: knuckle line, and hands come in every size -- so both keypoint sets are
#: centred on their own knuckle centroid and scaled by their own knuckle-row
#: width before being compared. The robot's knuckles are bolted to the palm,
#: which makes its normalisation constants genuinely constant: the network
#: cannot game the frame, only move the fingers inside it.
KNUCKLE_INDICES = (0, 4, 8)

#: Each digit's keypoint indices, base to tip, shared by both orderings.
DIGITS = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))


def digit_lengths(keypoints: torch.Tensor) -> torch.Tensor:
    """(B, 4) summed bone lengths of each digit's keypoint chain."""
    lengths = []
    for digit in DIGITS:
        chain = keypoints[:, list(digit)]
        lengths.append(torch.linalg.norm(chain[:, 1:] - chain[:, :-1], dim=-1).sum(-1))
    return torch.stack(lengths, dim=-1)


def build_targets(human: torch.Tensor, robot_anchor: torch.Tensor,
                  robot_digit_length: torch.Tensor) -> torch.Tensor:
    """Human keypoints to reachable robot-frame targets, per digit.

    A human finger and a LEAP finger have different proportions, so asking the
    robot to place its fingertips exactly on scaled human positions is asking
    for the impossible -- training against it plateaus at whatever the
    proportion gap is. The standard retargeting move (DexPilot and family) is
    per-digit: keep each digit's *shape* -- every keypoint's offset from its own
    knuckle -- and rescale it by the ratio of robot digit length to this
    person's digit length, anchored at the robot's own knuckle. Direction and
    curl transfer; proportions become the robot's; every target is reachable.
    """
    human_length = digit_lengths(human)  # (B, 4)
    targets = torch.empty(human.shape[0], 16, 3, dtype=human.dtype, device=human.device)
    for d, digit in enumerate(DIGITS):
        scale = (robot_digit_length[d] / human_length[:, d].clamp(min=1e-4)).view(-1, 1, 1)
        shape = human[:, list(digit)] - human[:, digit[0] : digit[0] + 1]
        targets[:, list(digit)] = robot_anchor[d].view(1, 1, 3) + shape * scale
    return targets


class RetargetNet(nn.Module):
    """A small MLP: 48 numbers of hand in, 16 joint angles out."""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(48, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 16),
        )
        # Start near the middle of every joint's range.
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        return self.body(keypoints.reshape(keypoints.shape[0], -1))


def normalise(keypoints: torch.Tensor) -> torch.Tensor:
    """Map keypoints into the shared knuckle-anchored, size-free frame."""
    anchors = keypoints[:, list(KNUCKLE_INDICES)]
    centre = anchors.mean(dim=1, keepdim=True)
    width = torch.linalg.norm(anchors[:, 0] - anchors[:, 2], dim=-1)
    return (keypoints - centre) / width.clamp(min=1e-4).view(-1, 1, 1)


def train_retargeter(
    steps: int = 3000,
    batch: int = 512,
    seed: int = 0,
    log: bool = True,
    path: Path | str = CHECKPOINT,
) -> dict:
    """Train the network on synthetic hands and cache the result."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fk = LeapFK()
    net = RetargetNet()

    mid = (fk.joint_low + fk.joint_high) / 2
    half = (fk.joint_high - fk.joint_low) / 2

    # The robot's own digit anchors and straight lengths, measured once.
    straight = fk(torch.zeros(1, 16))
    robot_anchor = torch.stack([straight[0, d[0]] for d in DIGITS])
    robot_digit_length = digit_lengths(straight)[0]

    #: Fingertips matter most; knuckles anchor the pose; mid points shape the curl.
    weights = torch.tensor(
        [1.0 if n.endswith("_knuckle") or n.endswith("_base") else
         0.6 if n.endswith("_mid") else
         0.8 if n.endswith("_distal") else 2.0
         for n in KEYPOINT_NAMES]
    ).view(1, -1, 1)

    optimiser = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)
    history = []
    for step in range(steps):
        human = sample_hands(batch, rng)
        target = build_targets(human, robot_anchor, robot_digit_length)

        raw = net(normalise(human))
        q = mid + half * torch.tanh(raw)          # joint limits by construction
        keypoints = fk(q)

        match = ((keypoints - target) ** 2 * weights).sum(-1).mean()
        strain = (raw ** 2).mean()                # keep away from tanh saturation
        loss = match + 1e-3 * strain
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()

        if log and (step % 300 == 0 or step == steps - 1):
            error = torch.sqrt(((keypoints - target) ** 2).sum(-1)).mean().detach()
            tips = torch.sqrt(((keypoints[:, 3::4] - target[:, 3::4]) ** 2).sum(-1)).mean().detach()
            history.append(float(error))
            print(f"  step {step:5d}  keypoint error {float(error) * 1000:6.2f} mm   "
                  f"fingertips {float(tips) * 1000:6.2f} mm")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net.state_dict()}, path)
    final = history[-1] if history else float("nan")
    return {"path": str(path), "final_error": final, "steps": steps}


class Retargeter:
    """Runtime wrapper: palm-frame keypoints in, clamped joint targets out."""

    def __init__(self, net: RetargetNet, fk: LeapFK) -> None:
        self.net = net.eval()
        self.fk = fk
        self.mid = (fk.joint_low + fk.joint_high) / 2
        self.half = (fk.joint_high - fk.joint_low) / 2

    @torch.no_grad()
    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)
        raw = self.net(normalise(tensor))
        q = self.mid + self.half * torch.tanh(raw)
        return q[0].numpy()


def train_personal(
    recording_path=None,
    steps: int = 2500,
    synthetic_fraction: float = 0.3,
    seed: int = 0,
    log: bool = True,
    min_curl_span: float = 0.03,
    require_improvement: bool = True,
) -> dict:
    """Fine-tune on the operator's own recorded hand.

    Recorded poses form most of each batch; a synthetic minority keeps coverage
    of poses the minute of recording missed. Same objective, same FK -- only the
    hand distribution changes, to the one that will actually be used.
    """
    from handrobot.dexhand.record import RECORDING, load_recording

    everything = torch.tensor(
        load_recording(recording_path or RECORDING), dtype=torch.float32
    )
    if everything.shape[0] < 200:
        raise ValueError(
            f"only {everything.shape[0]} recorded poses; record for longer first"
        )
    curl = everything[:, [3, 7, 11], 2].mean(-1)
    if float(curl.max() - curl.min()) < min_curl_span:
        raise ValueError(
            "recording barely varies -- no real fists or open hands captured; "
            "record again and follow the on-screen prompts"
        )
    # Hold out a slice the training never sees, so the final number is a
    # measurement of generalisation to this hand, not a memorisation score.
    held = max(32, everything.shape[0] // 10)
    holdout, recorded = everything[:held], everything[held:]

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fk = LeapFK()
    net = RetargetNet()
    if CHECKPOINT.exists():
        net.load_state_dict(
            torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["net"]
        )

    mid = (fk.joint_low + fk.joint_high) / 2
    half = (fk.joint_high - fk.joint_low) / 2
    straight = fk(torch.zeros(1, 16))
    robot_anchor = torch.stack([straight[0, d[0]] for d in DIGITS])
    robot_digit_length = digit_lengths(straight)[0]
    weights = torch.tensor(
        [1.0 if n.endswith("_knuckle") or n.endswith("_base") else
         0.6 if n.endswith("_mid") else
         0.8 if n.endswith("_distal") else 2.0
         for n in KEYPOINT_NAMES]
    ).view(1, -1, 1)

    optimiser = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)
    batch = 256
    n_synthetic = int(batch * synthetic_fraction)
    final = float("nan")
    for step in range(steps):
        picks = torch.tensor(rng.integers(0, recorded.shape[0], batch - n_synthetic))
        human = torch.cat([
            recorded[picks] * (1 + 0.03 * torch.randn(batch - n_synthetic, 1, 1))
            + 0.002 * torch.randn(batch - n_synthetic, 16, 3),
            sample_hands(n_synthetic, rng),
        ])
        target = build_targets(human, robot_anchor, robot_digit_length)
        raw = net(normalise(human))
        q = mid + half * torch.tanh(raw)
        keypoints = fk(q)
        loss = ((keypoints - target) ** 2 * weights).sum(-1).mean() + 1e-3 * (raw ** 2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        scheduler.step()
        if log and (step % 250 == 0 or step == steps - 1):
            error = torch.sqrt(((keypoints - target) ** 2).sum(-1)).mean().detach()
            final = float(error)
            print(f"  step {step:5d}  keypoint error {final * 1000:6.2f} mm")

    def holdout_metrics(candidate: RetargetNet) -> tuple[float, float]:
        with torch.no_grad():
            target = build_targets(holdout, robot_anchor, robot_digit_length)
            keypoints = fk(mid + half * torch.tanh(candidate(normalise(holdout))))
            mean = float(torch.sqrt(((keypoints - target) ** 2).sum(-1)).mean())
            tips = float(torch.sqrt(
                ((keypoints[:, 3::4] - target[:, 3::4]) ** 2).sum(-1)).mean())
        return mean, tips

    holdout_error, holdout_tips = holdout_metrics(net)
    base = RetargetNet()
    if CHECKPOINT.exists():
        base.load_state_dict(
            torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["net"]
        )
    _, base_tips = holdout_metrics(base)
    if log:
        print(f"  held-out check on {held} unseen poses of this hand: "
              f"{holdout_error * 1000:.1f} mm mean, fingertips {holdout_tips * 1000:.1f} mm "
              f"(base network: {base_tips * 1000:.1f} mm)")
    # A personal checkpoint is preferred by the loader forever after, so a run
    # that is not measurably better than the base network must install nothing.
    if require_improvement and holdout_tips >= base_tips:
        raise ValueError(
            f"training did not beat the base network on your own poses "
            f"({holdout_tips * 1000:.1f} vs {base_tips * 1000:.1f} mm fingertips); "
            "nothing installed -- record again with more finger variety"
        )

    PERSONAL_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net.state_dict()}, PERSONAL_CHECKPOINT)
    return {"path": str(PERSONAL_CHECKPOINT), "final_error": final,
            "holdout_error": holdout_error, "holdout_tips": holdout_tips,
            "holdout_tips_mm": holdout_tips * 1000, "base_tips_mm": base_tips * 1000,
            "recorded_poses": int(everything.shape[0])}


def load_retargeter(path: Path | str | None = None, train_if_missing: bool = True) -> Retargeter:
    if path is None:
        path = PERSONAL_CHECKPOINT if PERSONAL_CHECKPOINT.exists() else CHECKPOINT
    path = Path(path)
    if not path.exists():
        if not train_if_missing:
            raise FileNotFoundError(path)
        train_retargeter(path=path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    net = RetargetNet()
    net.load_state_dict(payload["net"])
    return Retargeter(net, LeapFK())
