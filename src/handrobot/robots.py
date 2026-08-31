"""Which robot the system is driving, and everything that differs between them.

The rest of the codebase is written against this description rather than against
any particular arm, because the choice of arm turned out to matter enormously.

The SO-101 is a lovely £200 machine, but it has five joints before the gripper
and cannot hold a straight-down grasp above about 9 cm from the table. Every
teleoperation session ran into that ceiling. The Panda has seven joints and
solves a straight-down grasp to under a millimetre anywhere in a workspace forty
times larger, which removes the constraint entirely rather than working around
it.

Both are kept. The SO-101 is the one you can buy and put on a desk; the Panda is
the one that makes teleoperation pleasant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from handrobot.paths import ASSETS_DIR


@dataclass(frozen=True)
class WorkspaceBox:
    """Axis-aligned region the gripper may be commanded into, in the base frame.

    A box, not the cylindrical sector the SO-101 needed: an arm with enough
    joints has no reason to be described in polar coordinates.
    """

    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    @property
    def low(self) -> np.ndarray:
        return np.array([self.x[0], self.y[0], self.z[0]])

    @property
    def high(self) -> np.ndarray:
        return np.array([self.x[1], self.y[1], self.z[1]])

    @property
    def center(self) -> np.ndarray:
        return (self.low + self.high) / 2.0

    @property
    def size(self) -> np.ndarray:
        return self.high - self.low

    def clip(self, p: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(p, dtype=float), self.low, self.high)

    def contains(self, p: np.ndarray, tolerance: float = 1e-9) -> bool:
        p = np.asarray(p, dtype=float)
        return bool(np.all(p >= self.low - tolerance) and np.all(p <= self.high + tolerance))


@dataclass(frozen=True)
class CylinderSector:
    """Region the gripper is allowed to be commanded into.

    It is a cylindrical sector, not a box, because that is the shape the arm
    actually has: the shoulder pan axis is vertical, so reachability depends on
    distance from the base and height, not on x and y separately.

    The height ceiling is low on purpose. Above roughly 10 cm the SO-101 cannot
    hold a useful grasp orientation, and a solver asked to try anyway misses the
    target by centimetres. See :mod:`handrobot.retarget.reach`, and
    ``tests/test_reach.py::test_declared_workspace_is_reachable`` which fails if
    these bounds are widened past what the arm can do.
    """

    radius_min: float = 0.17
    radius_max: float = 0.26
    azimuth_max: float = 0.785
    z_min: float = 0.012
    z_max: float = 0.095

    @property
    def low(self) -> np.ndarray:
        return self.from_polar(self.radius_min, -self.azimuth_max, self.z_min)

    @property
    def high(self) -> np.ndarray:
        return self.from_polar(self.radius_max, self.azimuth_max, self.z_max)

    @property
    def size(self) -> np.ndarray:
        """Rough extent, for reporting only; the region is not a box."""
        return np.array([
            self.radius_max - self.radius_min,
            2 * self.radius_max * np.sin(self.azimuth_max),
            self.z_max - self.z_min,
        ])

    @property
    def center(self) -> np.ndarray:
        radius = (self.radius_min + self.radius_max) / 2.0
        return np.array([radius, 0.0, (self.z_min + self.z_max) / 2.0])

    @staticmethod
    def to_polar(p: np.ndarray) -> tuple[float, float, float]:
        p = np.asarray(p, dtype=float)
        return float(np.hypot(p[0], p[1])), float(np.arctan2(p[1], p[0])), float(p[2])

    @staticmethod
    def from_polar(radius: float, azimuth: float, z: float) -> np.ndarray:
        return np.array([radius * np.cos(azimuth), radius * np.sin(azimuth), z])

    def clip(self, p: np.ndarray) -> np.ndarray:
        """Project a point into the sector, keeping its direction where possible."""
        radius, azimuth, z = self.to_polar(p)
        return self.from_polar(
            float(np.clip(radius, self.radius_min, self.radius_max)),
            float(np.clip(azimuth, -self.azimuth_max, self.azimuth_max)),
            float(np.clip(z, self.z_min, self.z_max)),
        )

    def contains(self, p: np.ndarray, tolerance: float = 1e-9) -> bool:
        radius, azimuth, z = self.to_polar(p)
        return bool(
            self.radius_min - tolerance <= radius <= self.radius_max + tolerance
            and abs(azimuth) <= self.azimuth_max + tolerance
            and self.z_min - tolerance <= z <= self.z_max + tolerance
        )


@dataclass(frozen=True)
class BoxLayout:
    """Where the objects spawn, sampled in Cartesian coordinates.

    Suits an arm whose reachable region is roughly a box.
    """

    cube_x: tuple[float, float]
    cube_y: tuple[float, float]
    bin_x: tuple[float, float]
    bin_y: tuple[float, float]
    cube_yaw: tuple[float, float] = (-0.6, 0.6)
    min_separation: float = 0.22

    def sample(self, rng, cube_height: float) -> tuple[np.ndarray, np.ndarray, float]:
        for _ in range(200):
            cube = np.array([rng.uniform(*self.cube_x), rng.uniform(*self.cube_y), cube_height])
            bin_pos = np.array([rng.uniform(*self.bin_x), rng.uniform(*self.bin_y), 0.0])
            if np.linalg.norm(cube[:2] - bin_pos[:2]) >= self.min_separation:
                return cube, bin_pos, float(rng.uniform(*self.cube_yaw))
        raise RuntimeError("could not sample a layout satisfying min_separation")


@dataclass(frozen=True)
class PolarLayout:
    """Where the objects spawn, sampled around the base.

    Suits an arm whose reachable region is a cylindrical sector, which is what
    happens when the shoulder turns about a vertical axis and the rest of the
    arm has too few joints to compensate.
    """

    radius: tuple[float, float]
    cube_azimuth: tuple[float, float]
    bin_azimuth: tuple[float, float]
    cube_yaw: tuple[float, float] = (-0.6, 0.6)
    min_separation: float = 0.10

    def sample(self, rng, cube_height: float) -> tuple[np.ndarray, np.ndarray, float]:
        for _ in range(200):
            cr, ca = rng.uniform(*self.radius), rng.uniform(*self.cube_azimuth)
            br, ba = rng.uniform(*self.radius), rng.uniform(*self.bin_azimuth)
            cube = np.array([cr * np.cos(ca), cr * np.sin(ca), cube_height])
            bin_pos = np.array([br * np.cos(ba), br * np.sin(ba), 0.0])
            if np.linalg.norm(cube[:2] - bin_pos[:2]) >= self.min_separation:
                return cube, bin_pos, float(rng.uniform(*self.cube_yaw))
        raise RuntimeError("could not sample a layout satisfying min_separation")


@dataclass(frozen=True)
class RobotSpec:
    """Everything about an arm that the rest of the code needs to know."""

    name: str
    arm_xml: Path
    scene_xml: Path

    actuators: tuple[str, ...]
    """Arm actuators in order, then the gripper actuator last."""

    tcp_site: str
    """Site the inverse kinematics targets, between the fingertips."""

    jaw_bodies: tuple[str, str]
    """Two points whose separation is the jaw gap.

    Named sites, geoms or bodies -- whichever an arm puts in the right place.
    """

    gripper_joint: str
    """Joint whose angle sets the jaw gap."""

    site_from_grasp: np.ndarray
    """Maps the canonical grasp frame onto this robot's site frame.

    The canonical frame has x along the approach direction and z along the jaw
    opening. Each arm's tool frame labels its axes differently; this is the
    fixed rotation between the two, so every other module can work in one
    convention.
    """

    workspace: WorkspaceBox | CylinderSector
    """Region the gripper may be commanded into.

    A box for an arm with enough joints; a cylindrical sector for one whose
    reach is dictated by a vertical shoulder axis.
    """

    home_key: str = "home"

    needs_reach_table: bool = False
    """Whether a straight-down grasp is unreachable in part of the workspace.

    True for arms that run out of joints, which then need the measured
    height-to-approach-angle coupling in :mod:`handrobot.retarget.reach`.
    """

    #: Half-width of the manipulated cube, and the inner half-width of the bin.
    cube_half_extent: float = 0.0125
    bin_inner_half: float = 0.041

    #: Where the objects spawn.
    layout: BoxLayout | PolarLayout = field(
        default_factory=lambda: BoxLayout((0.4, 0.6), (0.1, 0.3), (0.4, 0.6), (-0.3, -0.1))
    )

    #: Heights the scripted plan and the on-screen guidance both use, measured
    #: from the table with the tool site at the fingertips.
    grasp_clearance: float = 0.0
    """Site height above the cube's centre when the jaws close on it."""

    hover_height: float = 0.10
    """Height the cube is carried at."""

    release_clearance: float = 0.05
    """Site height above the bin's rim when the cube is let go."""

    #: How close the cube's centre must be to the bin's, in the plane, to count.
    success_tolerance: float = 0.035

    #: Offset from the gripper to the chase camera's eye, in the base frame.
    #: Behind (towards the operator, +x) and above, far enough back that the
    #: gripper and its target both fit in a 55-degree view.
    chase_offset: tuple[float, float, float] = (0.42, 0.0, 0.30)

    #: How strongly the inverse kinematics resists changing posture.
    #: A seven-joint arm can reach the same tool pose in infinitely many ways,
    #: and with a weak posture term it drifts through that null space during a
    #: long sweep until the solver diverges and drops whatever it was holding.
    #: A five-joint arm has no such freedom and does not want the constraint.
    ik_posture_cost: float = 2e-3

    #: Fastest the arm joints may be commanded, in radians per second.
    max_joint_speed: float = 3.0
    #: Seconds the gripper is allowed to take to travel its whole command range.
    gripper_travel_time: float = 0.35

    #: Jaw gaps the operator's fully-closed and fully-open pinch map onto, in
    #: metres. Robot-specific because it has to bracket the object: a range
    #: tuned for a 25 mm cube barely moves a gripper that opens to 80 mm.
    gripper_gap_range: tuple[float, float] = (0.010, 0.075)

    #: Metres per second the scripted plan moves the gripper. Durations are
    #: derived from distance rather than fixed, because an arm with twice the
    #: reach travels twice as far between the same two waypoints and a fixed
    #: schedule leaves the command trailing its target.
    travel_speed: float = 0.16
    min_segment: float = 0.45

    #: Seconds the scripted plan spends closing and opening the jaws. A soft,
    #: heavily damped gripper needs longer to travel than a stiff one, and a
    #: close that ends before the jaws arrive simply lifts nothing.
    close_duration: float = 0.55

    @property
    def arm_actuators(self) -> tuple[str, ...]:
        return self.actuators[:-1]

    @property
    def gripper_actuator(self) -> str:
        return self.actuators[-1]

    @property
    def gripper_index(self) -> int:
        return len(self.actuators) - 1

    @property
    def n_arm_joints(self) -> int:
        return len(self.actuators) - 1

    def command_speed_limits(self, gripper_command_range: float) -> np.ndarray:
        """Per-actuator command rate limits, in command units per second."""
        limits = np.full(len(self.actuators), self.max_joint_speed, dtype=float)
        limits[self.gripper_index] = gripper_command_range / max(self.gripper_travel_time, 1e-6)
        return limits

    def grasp_rotation(self, approach: np.ndarray, jaw: np.ndarray) -> np.ndarray:
        """Site orientation that approaches along ``approach`` with jaws on ``jaw``."""
        from handrobot.geometry import frame_from_axes

        return frame_from_axes(approach, jaw) @ self.site_from_grasp


def _permutation(x_from: int, y_from: int, z_from: int, signs=(1, 1, 1)) -> np.ndarray:
    """Rotation whose columns pick out (and optionally flip) canonical axes."""
    matrix = np.zeros((3, 3))
    for column, (source, sign) in enumerate(zip((x_from, y_from, z_from), signs)):
        matrix[source, column] = sign
    if not np.isclose(np.linalg.det(matrix), 1.0):
        raise ValueError("permutation is not a rotation")
    return matrix


#: The SO-101's gripper site already uses the canonical convention: its local x
#: points out between the jaws and its local z is the opening direction.
SO101_SITE_FROM_GRASP = np.eye(3)

#: The Panda's tool frame points its local z out of the gripper and opens its
#: fingers along local x, so the canonical axes are permuted.
PANDA_SITE_FROM_GRASP = _permutation(x_from=2, y_from=1, z_from=0, signs=(1, -1, 1))


SO101 = RobotSpec(
    name="so101",
    arm_xml=ASSETS_DIR / "so101" / "so101.xml",
    scene_xml=ASSETS_DIR / "so101" / "scene_task.xml",
    actuators=("shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"),
    tcp_site="gripperframe",
    # The SO-101's jaw bodies sit at their hinges, so the tip geoms are used.
    jaw_bodies=("fixed_jaw_sph_tip1", "moving_jaw_sph_tip1"),
    gripper_joint="gripper",
    site_from_grasp=SO101_SITE_FROM_GRASP,
    # Kept as a box for the shared interface; the true reachable region is the
    # cylindrical sector in WorkspaceConfig, which this arm still needs.
    workspace=CylinderSector(),
    needs_reach_table=True,
    cube_half_extent=0.0125,
    bin_inner_half=0.041,
    layout=PolarLayout(radius=(0.185, 0.245), cube_azimuth=(0.24, 0.62),
                       bin_azimuth=(-0.62, -0.24), min_separation=0.10),
    grasp_clearance=0.0095,
    hover_height=0.095,
    release_clearance=0.036,
    success_tolerance=0.035,
    travel_speed=0.16,
    min_segment=0.45,
    gripper_gap_range=(0.010, 0.075),
    chase_offset=(0.24, 0.0, 0.17),
)

PANDA = RobotSpec(
    name="panda",
    arm_xml=ASSETS_DIR / "panda" / "panda.xml",
    scene_xml=ASSETS_DIR / "panda" / "scene_task.xml",
    actuators=("actuator1", "actuator2", "actuator3", "actuator4",
               "actuator5", "actuator6", "actuator7", "actuator8"),
    tcp_site="tcp",
    jaw_bodies=("left_finger", "right_finger"),
    gripper_joint="finger_joint1",
    site_from_grasp=PANDA_SITE_FROM_GRASP,
    # Measured: a straight-down grasp solves to under a tenth of a millimetre
    # across all of this, with no coupling between height and approach angle.
    # Reach falls off past x = 0.65, which is where this stops.
    workspace=WorkspaceBox(x=(0.32, 0.62), y=(-0.32, 0.32), z=(0.040, 0.40)),
    needs_reach_table=False,
    # Sized to what a hand-tracked operator can actually hit. A published study
    # of vision-based teleoperation with 42 participants reports about 6.7 cm of
    # placement error; asking for the 8 mm a 25 mm cube needs was the single
    # biggest reason teleoperation felt impossible.
    cube_half_extent=0.030,
    bin_inner_half=0.105,
    layout=BoxLayout(cube_x=(0.42, 0.58), cube_y=(0.12, 0.28),
                     bin_x=(0.42, 0.58), bin_y=(-0.28, -0.12),
                     min_separation=0.22),
    # The Panda's tool site sits between the fingertips, so it goes to the
    # cube's centre height to grasp it.
    grasp_clearance=0.0,
    hover_height=0.26,
    release_clearance=0.13,
    # A 21 cm bin and a 6 cm cube: the cube's centre may be 7 cm off and still
    # land inside. That is roughly the placement accuracy a hand-tracked
    # operator actually achieves, which is the whole point of the resizing.
    success_tolerance=0.070,
    close_duration=1.1,
    travel_speed=0.32,
    min_segment=0.5,
    gripper_gap_range=(0.018, 0.078),
    ik_posture_cost=0.1,
    chase_offset=(0.42, 0.0, 0.30),
)

ROBOTS: dict[str, RobotSpec] = {spec.name: spec for spec in (PANDA, SO101)}
DEFAULT_ROBOT = "panda"


def get_robot(name: str | None = None) -> RobotSpec:
    """Look up a robot by name, defaulting to the one teleoperation works best on."""
    key = (name or DEFAULT_ROBOT).lower()
    if key not in ROBOTS:
        raise KeyError(f"unknown robot {name!r}; choose from {sorted(ROBOTS)}")
    return ROBOTS[key]
