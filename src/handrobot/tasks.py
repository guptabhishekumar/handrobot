"""The task registry: what the operator can ask the robot to do.

One scene, several objectives. Every task shares the same puck, bin and
cameras, so a multi-task policy cannot cheat by recognising the furniture --
the only thing that distinguishes "put it in the bin" from "lift it up" is the
task conditioning, which is exactly the ability being trained.

Each task defines its success predicate, the scripted expert's plan for it,
and the natural-language phrasings that name it. Success is judged by the
same simulator state the single-task pipeline always used; nothing here can
inflate a score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from handrobot.sim.env import PickPlaceEnv


@dataclass(frozen=True)
class TaskSpec:
    """One objective over the shared scene."""

    name: str
    instruction: str
    """The canonical phrasing, stored with every episode."""

    aliases: tuple[str, ...]
    """Other phrasings that resolve to this task."""

    success: Callable[["PickPlaceEnv"], bool]
    uses_zone: bool = False
    """Whether the task needs the sampled target zone (drawn as the ring)."""

    hold_steps: int | None = None
    """Consecutive steps the predicate must hold; None uses the sim default."""

    max_steps: int | None = None
    """Episode step budget override; None uses the sim default. Pushing is
    legitimately slower than picking -- the puck can only be moved at sliding
    speed, and a veer costs a full replan -- so it gets more clock, not a
    lower bar."""


def _bin_success(env: "PickPlaceEnv") -> bool:
    return env.cube_in_bin()


def _push_success(env: "PickPlaceEnv") -> bool:
    """The puck rests inside the ring, on the table."""
    if env.zone_position is None:
        return False
    cube = env.cube_position
    planar = float(np.linalg.norm(cube[:2] - env.zone_position[:2]))
    on_table = cube[2] < env.cube_half_extent + 0.02
    return planar <= 0.05 and bool(on_table)


def _lift_success(env: "PickPlaceEnv") -> bool:
    """The puck is held well clear of the table."""
    return float(env.cube_position[2]) > 0.15


def _touch_success(env: "PickPlaceEnv") -> bool:
    """The gripper rests against the puck without having knocked it away."""
    cube = env.cube_position
    tcp, _ = env.gripper_pose
    close = float(np.linalg.norm(tcp - cube)) < 0.055
    undisturbed = (
        env.cube_start_position is not None
        and float(np.linalg.norm(cube[:2] - env.cube_start_position[:2])) < 0.03
    )
    return close and bool(undisturbed)


TASKS: dict[str, TaskSpec] = {
    "bin": TaskSpec(
        name="bin",
        instruction="put the puck in the bin",
        aliases=("put it in the bin", "pick and place", "drop it in the box",
                 "put the object in the bin"),
        success=_bin_success,
    ),
    "push": TaskSpec(
        name="push",
        instruction="push the puck to the green ring",
        aliases=("push it to the ring", "slide the puck to the target",
                 "move it to the marker"),
        success=_push_success,
        uses_zone=True,
        max_steps=700,
    ),
    "lift": TaskSpec(
        name="lift",
        instruction="lift the puck up high",
        aliases=("pick it up and hold it", "lift it", "hold the puck in the air"),
        success=_lift_success,
    ),
    "touch": TaskSpec(
        name="touch",
        instruction="touch the puck without moving it",
        aliases=("gently touch it", "tap the puck", "touch the object"),
        success=_touch_success,
    ),
}

#: Stable task ordering; the policy's task ids index into this.
TASK_NAMES: tuple[str, ...] = tuple(TASKS)


def get_task(name: str) -> TaskSpec:
    """Resolve a task by name or by any natural-language alias."""
    key = name.strip().lower()
    if key in TASKS:
        return TASKS[key]
    for task in TASKS.values():
        if key == task.instruction or key in task.aliases:
            return task
    raise KeyError(
        f"unknown task {name!r}; known tasks: {', '.join(TASK_NAMES)} "
        f"(or any of their phrasings)"
    )


def task_id(name: str) -> int:
    return TASK_NAMES.index(get_task(name).name)
