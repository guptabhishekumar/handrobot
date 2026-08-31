"""Why is this policy failing?

A success rate tells you that a policy is bad, not what is wrong with it. These
three measurements separate the common causes, which need opposite fixes:

* **Grasp accuracy.** The cube is 25 mm wide, so the gripper must arrive within
  roughly 8 mm or the jaws close on air. A policy can look like it is doing the
  whole task -- approach, descend, close, carry, release -- and still score zero
  because it is a centimetre off at one moment. This is by far the most common
  failure, and the fix is more training or more demonstrations.
* **Whether the policy uses its cameras.** Run the same policy on two different
  cube positions. If the trajectories are nearly identical, the policy has
  learned one average motion and ignores what it sees. More training will not
  fix that; more varied demonstrations will.
* **Whether it moves at all.** A policy that has collapsed onto a constant
  output looks like a frozen arm, and means the run diverged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from handrobot.config import Config

#: The gripper must land within this of the cube's centre for the jaws to close
#: on it rather than beside it.
GRASP_TOLERANCE = 0.008

#: Below this peak difference between two very different layouts, the policy is
#: replaying one memorised trajectory rather than reacting to the scene.
VISION_THRESHOLD = 0.15


@dataclass
class Diagnosis:
    """What the three probes found."""

    grasp_errors: list[float]
    """Planar distance from gripper to cube at the moment of the grasp, in metres."""

    vision_sensitivity: float
    """Peak per-joint difference between two very different layouts, in radians."""

    action_range: float
    """Peak-to-peak joint motion over a rollout, in radians."""

    success_rate: float
    episodes: int

    @property
    def mean_grasp_error(self) -> float:
        return float(np.mean(self.grasp_errors)) if self.grasp_errors else float("nan")

    @property
    def grasp_is_accurate(self) -> bool:
        return self.mean_grasp_error <= GRASP_TOLERANCE

    @property
    def uses_vision(self) -> bool:
        return self.vision_sensitivity > VISION_THRESHOLD

    @property
    def moves(self) -> bool:
        return self.action_range > 0.1

    def verdict(self) -> list[str]:
        """Plain-language reading of the numbers, most important first."""
        lines = []
        if not self.moves:
            lines.append(
                "The arm barely moves. The policy has collapsed onto a constant output, "
                "which usually means training diverged. Check the training log for a loss "
                "that stopped falling or went to NaN, and retrain with a lower learning rate."
            )
            return lines

        if not self.uses_vision:
            lines.append(
                "The policy ignores its cameras: it produces almost the same motion "
                "wherever the cube is. It has memorised one average trajectory. More "
                "training will not help. Record demonstrations with the cube in more "
                "varied positions."
            )
        else:
            lines.append("The policy is using its cameras: the motion changes with the scene.")

        if self.grasp_is_accurate:
            lines.append(
                f"Grasp accuracy is good ({self.mean_grasp_error * 1000:.1f} mm, "
                f"needs under {GRASP_TOLERANCE * 1000:.0f} mm)."
            )
        else:
            lines.append(
                f"The grasp misses by {self.mean_grasp_error * 1000:.1f} mm on average, and "
                f"the cube is only 25 mm wide, so the jaws are closing beside it. This is "
                f"the usual cause of a low score and it is fixed by training longer "
                f"(--steps) or recording more demonstrations."
            )

        if self.uses_vision and self.grasp_is_accurate and self.success_rate < 0.5:
            lines.append(
                "The policy sees the scene and reaches the cube accurately, yet still fails. "
                "Look at a rendered rollout -- 'handrobot demo --checkpoint ...' -- because "
                "the problem is later in the motion: the carry or the release."
            )
        return lines


def _actor(checkpoint: Path | str, config: Config, device_name: str | None):
    from handrobot.policy.inference import ChunkedActor, load_checkpoint
    from handrobot.policy.train import resolve_device

    device = resolve_device(device_name)
    policy, stats, extra = load_checkpoint(checkpoint, device)
    return (
        ChunkedActor(
            policy, stats, device,
            image_size=config.train.image_size,
            temporal_ensemble_coeff=config.train.temporal_ensemble_coeff,
        ),
        extra,
    )


def measure_grasp_accuracy(
    checkpoint: Path | str,
    config: Config,
    device_name: str | None = None,
    grasp_step: int = 62,
) -> list[float]:
    """How close the gripper gets to the cube, at five points across the workspace.

    ``grasp_step`` is when the scripted plan closes the jaws; the learned policy
    keeps roughly the same timing because it was trained on that plan.
    """
    from handrobot.sim import PickPlaceEnv

    actor, _ = _actor(checkpoint, config, device_name)
    rng = np.random.default_rng(0)
    layouts = [
        config.spec.layout.sample(rng, config.spec.cube_half_extent) for _ in range(5)
    ]
    errors = []
    env = PickPlaceEnv(config=config)
    try:
        for cube, bin_pos, _ in layouts:
            observation = env.reset(seed=1, cube_position=cube, bin_position=bin_pos)
            actor.reset()
            for _ in range(grasp_step):
                action = actor.act(observation.images, observation.joint_positions)
                observation = env.step(action).observation
            gripper, _ = env.gripper_pose
            errors.append(float(np.linalg.norm(gripper[:2] - cube[:2])))
    finally:
        env.close()
    return errors


def measure_vision_sensitivity(
    checkpoint: Path | str,
    config: Config,
    device_name: str | None = None,
    steps: int = 60,
) -> tuple[float, float]:
    """Run the policy on two very different layouts and compare the trajectories.

    Returns ``(peak difference between layouts, peak motion within one layout)``.
    The second is the control: a frozen policy would also show no difference
    between layouts, for an entirely different reason.
    """
    from handrobot.sim import PickPlaceEnv

    rng = np.random.default_rng(0)
    first, bin_pos, _ = config.spec.layout.sample(rng, config.spec.cube_half_extent)
    second, _, _ = config.spec.layout.sample(rng, config.spec.cube_half_extent)
    layouts = [first, second]

    trajectories = []
    env = PickPlaceEnv(config=config)
    try:
        for cube in layouts:
            actor, _ = _actor(checkpoint, config, device_name)
            observation = env.reset(seed=1, cube_position=cube, bin_position=bin_pos)
            actor.reset()
            actions = []
            for _ in range(steps):
                action = actor.act(observation.images, observation.joint_positions)
                actions.append(action)
                observation = env.step(action).observation
            trajectories.append(np.array(actions))
    finally:
        env.close()

    # Arm joints only. The gripper's command is not in radians on every robot,
    # and a 0-255 tendon command would swamp both figures.
    n = config.spec.n_arm_joints
    difference = float(np.max(np.abs(trajectories[0][:, :n] - trajectories[1][:, :n])))
    motion = float(np.max(np.ptp(trajectories[0][:, :n], axis=0)))
    return difference, motion


def diagnose(
    checkpoint: Path | str,
    episodes: int = 10,
    config: Config | None = None,
    device_name: str | None = None,
) -> Diagnosis:
    """Run all three probes plus a short success evaluation."""
    from handrobot.rollout import PolicyController, evaluate_controller
    from handrobot.sim import PickPlaceEnv

    config = config or Config()
    grasp_errors = measure_grasp_accuracy(checkpoint, config, device_name)
    sensitivity, motion = measure_vision_sensitivity(checkpoint, config, device_name)

    actor, _ = _actor(checkpoint, config, device_name)
    env = PickPlaceEnv(config=config)
    try:
        results = evaluate_controller(
            env, PolicyController(actor), episodes, seed=970000, verbose=False
        )
    finally:
        env.close()

    return Diagnosis(
        grasp_errors=grasp_errors,
        vision_sensitivity=sensitivity,
        action_range=motion,
        success_rate=results["success_rate"],
        episodes=episodes,
    )
