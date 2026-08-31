"""What the arm can actually reach, measured rather than assumed.

The SO-101 has five joints before the gripper, which is one short of a full
six-DoF pose. In practice the shortfall shows up as a coupling between height
and approach angle: near the table the gripper can point straight down, but the
higher it goes the further it has to lean outward. Asking for a straight-down
grasp at 15 cm produces a solution that silently misses the target by centimetres.

So the approach angle is not a free choice. This module measures the best
achievable approach pitch on a grid of (radius, height), caches the result, and
interpolates it at runtime. Both the scripted expert and the teleoperator use it,
which is why neither of them ever commands a pose the arm cannot hold.

The table is azimuth-independent: the shoulder pan axis is vertical, so rotating
a target about the base rotates the whole solution with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from handrobot.geometry import frame_from_axes
from handrobot.paths import ASSETS_DIR

DEFAULT_TABLE_PATH = ASSETS_DIR / "reach_table.npz"

#: Grid the table is built on. Covers everything within the arm's span.
RADII = np.arange(0.12, 0.301, 0.01)
HEIGHTS = np.arange(0.010, 0.2401, 0.01)
PITCHES = np.radians(np.arange(0.0, 76.0, 2.5))

#: A target counts as reachable when the gripper lands this close to it.
REACHABLE_TOLERANCE = 0.006


def approach_frame(x: float, y: float, pitch: float, jaw_azimuth: float | None = None) -> np.ndarray:
    """Gripper frame leaning ``pitch`` radians outward from straight down.

    Args:
        x, y: target position, used for the outward (radial) direction.
        pitch: 0 points the gripper straight at the table; larger values lean it
            outward, away from the base.
        jaw_azimuth: direction the jaws open, in the horizontal plane. Defaults
            to perpendicular to the radial direction, which is the orientation a
            human naturally uses to grasp something in front of them.
    """
    radial = np.array([x, y, 0.0], dtype=float)
    norm = np.linalg.norm(radial)
    radial = np.array([1.0, 0.0, 0.0]) if norm < 1e-9 else radial / norm

    approach = -np.cos(pitch) * np.array([0.0, 0.0, 1.0]) + np.sin(pitch) * radial
    if jaw_azimuth is None:
        jaw = np.cross(np.array([0.0, 0.0, 1.0]), radial)
    else:
        jaw = np.array([np.cos(jaw_azimuth), np.sin(jaw_azimuth), 0.0])
    return frame_from_axes(approach, jaw)


def warm_start(x: float, y: float, rest: np.ndarray | None = None) -> np.ndarray:
    """Seed the solver with the shoulder already turned towards the target.

    Differential IK is local, so a warm start pointing the wrong way costs
    accuracy. The pan joint turns opposite to the target azimuth on this arm.
    """
    q = np.array([0.0, -1.0, 1.0, 0.5, 0.0, 0.5]) if rest is None else np.asarray(rest, float).copy()
    q[0] = -np.arctan2(y, x)
    return q


@dataclass
class ReachTable:
    """Best approach pitch and residual error over a (radius, height) grid."""

    radii: np.ndarray
    heights: np.ndarray
    pitch: np.ndarray
    """(len(radii), len(heights)) best approach pitch in radians."""

    error: np.ndarray
    """(len(radii), len(heights)) position error at that pitch, in metres."""

    def save(self, path: Path | str = DEFAULT_TABLE_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, radii=self.radii, heights=self.heights, pitch=self.pitch, error=self.error
        )
        return path

    @classmethod
    def load(cls, path: Path | str = DEFAULT_TABLE_PATH) -> "ReachTable":
        with np.load(Path(path)) as raw:
            return cls(radii=raw["radii"], heights=raw["heights"],
                       pitch=raw["pitch"], error=raw["error"])

    @classmethod
    def build(cls, ik=None, progress: bool = False) -> "ReachTable":
        """Sweep the grid, solving IK at every candidate pitch. Takes a few seconds."""
        from handrobot.retarget.ik import ArmIK
        from handrobot.robots import get_robot

        # This table describes the SO-101 and nothing else: its grid, warm
        # starts and pitch coupling are that arm's geometry. Building it with
        # the *default* arm broke the day the default became the Panda -- and
        # only on machines without the cached file, which is why CI caught it
        # and no local run ever did.
        ik = ik or ArmIK(spec=get_robot("so101"))
        pitch = np.zeros((len(RADII), len(HEIGHTS)))
        error = np.full((len(RADII), len(HEIGHTS)), np.inf)

        for i, radius in enumerate(RADII):
            for j, height in enumerate(HEIGHTS):
                q0 = warm_start(radius, 0.0)
                target = np.array([radius, 0.0, height])
                for candidate in PITCHES:
                    result = ik.solve(
                        target, approach_frame(radius, 0.0, candidate), q0, iterations=100
                    )
                    if result.position_error < error[i, j]:
                        error[i, j] = result.position_error
                        pitch[i, j] = candidate
                    # Straight down is always preferred when it is achievable.
                    if error[i, j] < 1e-3 and candidate == PITCHES[0]:
                        break
            if progress:
                print(f"  reach table: radius {radius:.2f} m done")
        return cls(radii=RADII.copy(), heights=HEIGHTS.copy(), pitch=pitch, error=error)

    @classmethod
    def cached(cls, path: Path | str = DEFAULT_TABLE_PATH, rebuild: bool = False) -> "ReachTable":
        """Load the table, building and caching it the first time."""
        path = Path(path)
        if path.exists() and not rebuild:
            return cls.load(path)
        table = cls.build(progress=True)
        table.save(path)
        return table

    # -- queries ------------------------------------------------------------

    def _interpolate(self, grid: np.ndarray, radius: float, height: float) -> float:
        """Bilinear interpolation with clamping at the grid edges."""
        r = np.clip(radius, self.radii[0], self.radii[-1])
        h = np.clip(height, self.heights[0], self.heights[-1])
        i = int(np.clip(np.searchsorted(self.radii, r) - 1, 0, len(self.radii) - 2))
        j = int(np.clip(np.searchsorted(self.heights, h) - 1, 0, len(self.heights) - 2))
        tr = (r - self.radii[i]) / (self.radii[i + 1] - self.radii[i])
        th = (h - self.heights[j]) / (self.heights[j + 1] - self.heights[j])
        return float(
            grid[i, j] * (1 - tr) * (1 - th)
            + grid[i + 1, j] * tr * (1 - th)
            + grid[i, j + 1] * (1 - tr) * th
            + grid[i + 1, j + 1] * tr * th
        )

    def approach_pitch(self, position: np.ndarray) -> float:
        """Approach pitch, in radians, for a target in the robot base frame."""
        position = np.asarray(position, dtype=float)
        radius = float(np.hypot(position[0], position[1]))
        return self._interpolate(self.pitch, radius, float(position[2]))

    def expected_error(self, position: np.ndarray) -> float:
        """Position error the arm is expected to have at this target, in metres."""
        position = np.asarray(position, dtype=float)
        radius = float(np.hypot(position[0], position[1]))
        return self._interpolate(self.error, radius, float(position[2]))

    def reachable(self, position: np.ndarray, tolerance: float = REACHABLE_TOLERANCE) -> bool:
        return self.expected_error(position) <= tolerance

    def frame_for(self, position: np.ndarray, jaw_azimuth: float | None = None) -> np.ndarray:
        """Gripper rotation to command at this target."""
        position = np.asarray(position, dtype=float)
        return approach_frame(
            position[0], position[1], self.approach_pitch(position), jaw_azimuth
        )

    def reachable_box(self, tolerance: float = REACHABLE_TOLERANCE) -> dict[str, float]:
        """Largest axis-aligned radius/height rectangle that is entirely reachable.

        Used to keep :class:`~handrobot.config.WorkspaceConfig` honest.
        """
        mask = self.error <= tolerance
        best = None
        for j0 in range(len(self.heights)):
            for j1 in range(j0, len(self.heights)):
                column = mask[:, j0 : j1 + 1].all(axis=1)
                runs = _longest_run(column)
                if runs is None:
                    continue
                i0, i1 = runs
                area = (self.radii[i1] - self.radii[i0]) * (self.heights[j1] - self.heights[j0])
                if best is None or area > best[0]:
                    best = (area, i0, i1, j0, j1)
        if best is None:
            raise RuntimeError("no reachable region found; the table is empty")
        _, i0, i1, j0, j1 = best
        return {
            "radius_min": float(self.radii[i0]),
            "radius_max": float(self.radii[i1]),
            "height_min": float(self.heights[j0]),
            "height_max": float(self.heights[j1]),
        }


def _longest_run(flags: np.ndarray) -> tuple[int, int] | None:
    """Start and end indices of the longest contiguous run of ``True``."""
    best = current_start = None
    best_length = 0
    for index, value in enumerate(flags):
        if value:
            if current_start is None:
                current_start = index
            length = index - current_start + 1
            if length > best_length:
                best_length, best = length, (current_start, index)
        else:
            current_start = None
    return best
