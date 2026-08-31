"""Running a controller in the environment, with optional recording.

One function serves the scripted expert, the teleoperator and the trained
policy, so all three produce datasets with identical structure and are scored by
identical rules. If evaluation and data collection diverged, the numbers would
not be comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from handrobot.data.dataset import EpisodeWriter
from handrobot.sim.env import Observation, PickPlaceEnv


class Controller(Protocol):
    """Anything that can drive the arm."""

    def reset(self, env: PickPlaceEnv) -> None: ...

    def act(self, env: PickPlaceEnv, observation: Observation) -> np.ndarray: ...


@dataclass
class RolloutResult:
    """Outcome of one episode."""

    success: bool
    steps: int
    frames: list[np.ndarray]
    final_cube_position: np.ndarray
    final_bin_position: np.ndarray


class ScriptedController:
    """Adapts :class:`~handrobot.scripted.expert.ScriptedExpert` to the protocol."""

    def __init__(self, expert) -> None:
        self.expert = expert

    def reset(self, env: PickPlaceEnv) -> None:
        self.expert.reset(env)

    def act(self, env: PickPlaceEnv, observation: Observation) -> np.ndarray:
        return self.expert.act(env)


class PolicyController:
    """Adapts a :class:`~handrobot.policy.inference.ChunkedActor` to the protocol."""

    def __init__(self, actor) -> None:
        self.actor = actor

    def reset(self, env: PickPlaceEnv) -> None:
        self.actor.reset()

    def act(self, env: PickPlaceEnv, observation: Observation) -> np.ndarray:
        return self.actor.act(observation.images, observation.joint_positions)


def run_episode(
    env: PickPlaceEnv,
    controller: Controller,
    max_steps: int | None = None,
    writer: EpisodeWriter | None = None,
    render_camera: str | None = None,
    render_size: tuple[int, int] = (720, 1280),
    seed: int | None = None,
    on_step: Callable[[int, Observation, np.ndarray], None] | None = None,
    settle_steps: int = 20,
) -> RolloutResult:
    """Run one episode and optionally record it.

    Args:
        env: the task environment.
        controller: the thing being evaluated.
        max_steps: overrides the configured episode limit.
        writer: when given, every step is appended to it. The caller decides
            whether to keep the episode.
        render_camera: extra camera rendered into :attr:`RolloutResult.frames`.
        render_size: ``(height, width)`` for that camera.
        seed: passed to :meth:`PickPlaceEnv.reset`.
        on_step: callback receiving ``(step, observation, action)``.
        settle_steps: extra steps run after the controller finishes, so a cube
            that is still falling gets a chance to land in the bin.

    Returns:
        A :class:`RolloutResult`. Success is judged by the environment, not the
        controller.
    """
    observation = env.reset(seed=seed)
    controller.reset(env)

    limit = max_steps if max_steps is not None else env.config.sim.max_episode_steps
    frames: list[np.ndarray] = []
    success = False
    step = 0

    while step < limit:
        action = controller.act(env, observation)
        if writer is not None:
            writer.add(observation.joint_positions, action, observation.images)
        if render_camera is not None:
            frames.append(env.render(render_camera, *render_size))
        if on_step is not None:
            on_step(step, observation, action)

        result = env.step(action)
        observation = result.observation
        step += 1
        if result.success:
            success = True
            break
        if result.done:
            break

    # Hold the last command while the scene settles.
    if not success and settle_steps:
        hold = env.commanded_positions
        for _ in range(settle_steps):
            if render_camera is not None:
                frames.append(env.render(render_camera, *render_size))
            result = env.step(hold)
            if result.success:
                success = True
                break

    return RolloutResult(
        success=success,
        steps=step,
        frames=frames,
        final_cube_position=env.cube_position,
        final_bin_position=env.bin_position,
    )


def evaluate_controller(
    env: PickPlaceEnv,
    controller: Controller,
    episodes: int,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Success rate over a fixed, reproducible set of layouts."""
    successes, lengths = 0, []
    per_episode = []
    for i in range(episodes):
        result = run_episode(env, controller, seed=seed + i)
        successes += int(result.success)
        lengths.append(result.steps)
        per_episode.append({"seed": seed + i, "success": result.success, "steps": result.steps})
        if verbose:
            print(f"  episode {i:3d} (seed {seed + i}): "
                  f"{'success' if result.success else 'FAIL   '}  {result.steps:3d} steps")
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / max(episodes, 1),
        "mean_steps": float(np.mean(lengths)) if lengths else 0.0,
        "per_episode": per_episode,
    }
