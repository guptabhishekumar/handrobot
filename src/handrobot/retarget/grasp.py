"""Choosing the gripper orientation for a target position.

For an arm with enough joints this is trivial: point straight down, open the
jaws whichever way was asked for, everywhere in the workspace. The Panda solves
that to under a tenth of a millimetre anywhere.

For an arm that runs out of joints it is not trivial at all. The SO-101 can only
point straight down near the table; higher up it has to lean outward by an angle
that depends on how far out and how high the target is. That coupling is
measured rather than guessed -- see :mod:`handrobot.retarget.reach` -- and this
module is what hides the difference from everything else.
"""

from __future__ import annotations

import numpy as np

from handrobot.robots import RobotSpec, get_robot

DOWN = np.array([0.0, 0.0, -1.0])


class GraspFrames:
    """Maps a target position and jaw angle to a gripper orientation."""

    def __init__(self, spec: RobotSpec | None = None, reach=None) -> None:
        self.spec = spec or get_robot()
        self._reach = None
        if self.spec.needs_reach_table:
            from handrobot.retarget.reach import ReachTable

            self._reach = reach or ReachTable.cached()

    @property
    def reach(self):
        """The measured reach table, or ``None`` for arms that do not need one."""
        return self._reach

    def approach_pitch(self, position: np.ndarray) -> float:
        """Radians the approach must lean away from straight down."""
        if self._reach is None:
            return 0.0
        return self._reach.approach_pitch(position)

    def reachable(self, position: np.ndarray) -> bool:
        if self._reach is None:
            return self.spec.workspace.contains(position)
        return bool(self._reach.reachable(position))

    def frame_for(self, position: np.ndarray, jaw_azimuth: float = 0.0) -> np.ndarray:
        """Gripper site orientation to command at this target."""
        position = np.asarray(position, dtype=float)
        pitch = self.approach_pitch(position)
        if pitch == 0.0:
            approach = DOWN
        else:
            radial = np.array([position[0], position[1], 0.0])
            norm = np.linalg.norm(radial)
            radial = np.array([1.0, 0.0, 0.0]) if norm < 1e-9 else radial / norm
            approach = -np.cos(pitch) * np.array([0.0, 0.0, 1.0]) + np.sin(pitch) * radial
        jaw = np.array([np.cos(jaw_azimuth), np.sin(jaw_azimuth), 0.0])
        return self.spec.grasp_rotation(approach, jaw)
