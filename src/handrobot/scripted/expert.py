"""Waypoint-following scripted expert for cube-into-bin.

The expert plans in gripper space -- a short list of Cartesian waypoints with a
jaw rotation and a jaw opening -- and converts each interpolated pose into joint
targets using the same inverse kinematics and the same reach table the human
teleoperator drives. Scripted and human demonstrations therefore land in the
same action distribution, and a policy can be trained on either or on a mix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from handrobot.config import Config
from handrobot.gripper import GripperCalibration
from handrobot.retarget.grasp import GraspFrames
from handrobot.retarget.ik import ArmIK
from handrobot.sim.env import PickPlaceEnv


@dataclass(frozen=True)
class Waypoint:
    """A gripper pose to reach, a jaw opening to hold, and how long to take."""

    position: np.ndarray
    jaw_azimuth: float
    jaw_gap: float
    duration: float
    label: str


from handrobot.retarget.mapper import wrap_to_pi


class ScriptedExpert:
    """Executes a pick-and-place plan, replanned at every reset."""

    def __init__(
        self,
        config: Config | None = None,
        ik: ArmIK | None = None,
        grasp: GraspFrames | None = None,
        calibration: GripperCalibration | None = None,
    ) -> None:
        self.config = config or Config()
        self.spec = self.config.spec
        self.ik = ik or ArmIK(self.config.ik, self.spec)
        self.grasp = grasp or GraspFrames(self.spec)
        self.calibration = calibration or GripperCalibration.cached(self.spec)

        # Every height comes from the robot, because where the tool site sits
        # relative to the fingers differs between arms.
        self.grasp_height = self.spec.cube_half_extent + self.spec.grasp_clearance
        self.hover_height = self.spec.hover_height
        self.release_height = self.spec.release_clearance + 2 * self.spec.cube_half_extent
        self.gripper_index = self.spec.gripper_index

        self._waypoints: list[Waypoint] = []
        self._timeline = np.zeros(0)
        self._q = np.zeros(6)
        self._t = 0.0
        self._start = (np.zeros(3), 0.0, 0.0)
        self._home = None

    # -- planning -----------------------------------------------------------

    def grasp_azimuth(self, cube_yaw: float, cube_position: np.ndarray) -> float:
        """Jaw direction that grips a pair of the cube's opposite faces.

        A cube has four-fold symmetry about the vertical, and a parallel jaw is
        unchanged by a half turn, so any of ``cube_yaw + k * pi / 2`` grips it.
        The one closest to the arm's natural wrist angle is chosen, which keeps
        the wrist roll joint away from its limits.
        """
        natural = float(np.arctan2(cube_position[1], cube_position[0])) + np.pi / 2
        candidates = [cube_yaw + k * np.pi / 2 for k in range(-2, 3)]
        return wrap_to_pi(min(candidates, key=lambda a: abs(wrap_to_pi(a - natural))))

    @staticmethod
    def _nearest_equivalent(azimuth: float, reference: float) -> float:
        """The half-turn equivalent of ``azimuth`` closest to ``reference``."""
        return wrap_to_pi(
            min(
                (azimuth + k * np.pi for k in (-1, 0, 1)),
                key=lambda a: abs(wrap_to_pi(a - reference)),
            )
        )

    def plan(self, env: PickPlaceEnv) -> list[Waypoint]:
        """Build the waypoint list for the current object layout."""
        workspace = self.config.workspace
        cube = env.cube_position.copy()
        bin_pos = env.bin_position.copy()

        grasp_azimuth = self.grasp_azimuth(_cube_yaw(env), cube)
        # The jaw angle must follow the arm as it pans across, not stay fixed in
        # world coordinates: holding a fixed world angle while swinging from one
        # side of the base to the other drives the wrist roll joint into its
        # limit, and the arm stops short of the bin.
        release_azimuth = self._nearest_equivalent(
            float(np.arctan2(bin_pos[1], bin_pos[0])) + np.pi / 2, grasp_azimuth
        )

        # Open wide enough to clear the cube comfortably, and close narrower
        # than it so the jaws actually load against it.
        width = 2 * self.spec.cube_half_extent
        open_gap = min(self.calibration.gap_max * 0.94, width + 0.030)
        grip_gap = max(self.calibration.gap_min, width - 0.008)

        at_cube = workspace.clip(np.array([cube[0], cube[1], self.grasp_height]))
        above_cube = workspace.clip(np.array([cube[0], cube[1], self.hover_height]))
        above_bin = workspace.clip(np.array([bin_pos[0], bin_pos[1], self.hover_height]))
        release = workspace.clip(
            np.array([bin_pos[0], bin_pos[1], bin_pos[2] + self.release_height])
        )

        start = self.config.workspace.clip(env.gripper_pose[0])

        def travel(a: np.ndarray, b: np.ndarray) -> float:
            """Seconds to move between two points at the arm's travel speed."""
            distance = float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
            return max(self.spec.min_segment, distance / self.spec.travel_speed)

        return [
            Waypoint(above_cube, grasp_azimuth, open_gap, travel(start, above_cube), "approach"),
            Waypoint(at_cube, grasp_azimuth, open_gap, travel(above_cube, at_cube), "descend"),
            Waypoint(at_cube, grasp_azimuth, grip_gap, self.spec.close_duration, "close"),
            Waypoint(above_cube, grasp_azimuth, grip_gap, travel(at_cube, above_cube), "lift"),
            Waypoint(above_bin, release_azimuth, grip_gap,
                     travel(above_cube, above_bin), "transfer"),
            Waypoint(release, release_azimuth, grip_gap, travel(above_bin, release), "lower"),
            Waypoint(release, release_azimuth, open_gap,
                     0.8 * self.spec.close_duration, "release"),
            Waypoint(above_bin, release_azimuth, open_gap, travel(release, above_bin), "retreat"),
        ]

    def reset(self, env: PickPlaceEnv) -> None:
        """Plan for the current episode and rewind the clock."""
        self._waypoints = self.plan(env)
        self._timeline = np.cumsum([w.duration for w in self._waypoints])
        self._q = env.commanded_positions.copy()
        self._home = env._home_ctrl.copy()
        self._t = 0.0

        position, _ = env.gripper_pose
        self._start = (
            self.config.workspace.clip(position),
            self._waypoints[0].jaw_azimuth,
            self.calibration.command_to_gap(self._q[self.gripper_index]),
        )

    @property
    def duration(self) -> float:
        return float(self._timeline[-1]) if len(self._timeline) else 0.0

    @property
    def finished(self) -> bool:
        return self._t >= self.duration

    # -- execution ----------------------------------------------------------

    def target_at(self, t: float) -> tuple[np.ndarray, float, float, str]:
        """Interpolated (position, jaw azimuth, jaw gap, label) at time ``t``."""
        index = min(int(np.searchsorted(self._timeline, t, side="right")),
                    len(self._waypoints) - 1)
        waypoint = self._waypoints[index]

        segment_start = self._timeline[index - 1] if index > 0 else 0.0
        alpha = float(np.clip((t - segment_start) / max(waypoint.duration, 1e-6), 0.0, 1.0))
        # Smoothstep keeps the commanded velocity continuous at every waypoint.
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)

        if index == 0:
            p0, a0, g0 = self._start
        else:
            previous = self._waypoints[index - 1]
            p0, a0, g0 = previous.position, previous.jaw_azimuth, previous.jaw_gap

        position = (1 - alpha) * p0 + alpha * waypoint.position
        azimuth = a0 + alpha * wrap_to_pi(waypoint.jaw_azimuth - a0)
        gap = (1 - alpha) * g0 + alpha * waypoint.jaw_gap
        return position, azimuth, gap, waypoint.label

    def act(self, env: PickPlaceEnv) -> np.ndarray:
        """Next joint-position command, advancing the internal clock."""
        position, azimuth, gap, _ = self.target_at(self._t)
        rotation = self.grasp.frame_for(position, azimuth)
        result = self.ik.solve(position, rotation, self._q)
        if not result.ok and self._home is not None:
            # Re-solve from the home pose rather than a stale one. This matters
            # when the arm has to swing right across the workspace.
            retry = self.ik.solve(position, rotation, self._home, iterations=60)
            if retry.position_error < result.position_error:
                result = retry
        self._q = result.q.copy()
        self._q[self.gripper_index] = self.calibration.gap_to_command(gap)
        self._t += self.config.sim.control_dt
        return self._q.copy()


def _cube_yaw(env: PickPlaceEnv) -> float:
    """Yaw of the cube about the world z axis, in radians."""
    rotation = env.data.xmat[env.model.body("cube").id].reshape(3, 3)
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))
