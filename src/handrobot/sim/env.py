"""The pick-and-place task: put the cube in the bin.

The action space is six joint position targets in radians -- exactly what a
real SO-101 accepts over its serial bus -- so a policy trained here speaks the
same language as the physical arm.

The observation is what a real SO-101 rig can actually measure: joint positions
read back from the servos, plus camera images. Object poses are available for
scripting and scoring but are deliberately kept out of the policy observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from handrobot.config import Config, SimConfig
from handrobot.robots import RobotSpec

#: Defaults for the SO-101, kept because a few tests import them. The
#: environment itself reads these from the robot it was given.
CUBE_HALF_EXTENT = 0.0125
BIN_INNER_HALF = 0.041
BIN_FLOOR_TOP = 0.008


@dataclass
class Observation:
    """One control-step observation."""

    joint_positions: np.ndarray
    """(6,) measured joint angles in radians, in actuator order."""

    images: dict[str, np.ndarray]
    """Camera name to uint8 RGB array."""

    time: float
    """Simulated seconds since reset."""


@dataclass
class StepResult:
    """What :meth:`PickPlaceEnv.step` returns."""

    observation: Observation
    success: bool
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class PickPlaceEnv:
    """SO-101 pick-and-place, position controlled at a fixed rate."""

    def __init__(
        self,
        config: Config | None = None,
        render_cameras: tuple[str, ...] | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or Config()
        self.spec: RobotSpec = self.config.spec
        self.sim: SimConfig = self.config.sim
        self.layout = self.spec.layout
        #: Old name, kept for callers that still say ``env.randomization``.
        self.randomization = self.layout

        self.model = mujoco.MjModel.from_xml_path(str(self.spec.scene_xml))
        self.model.opt.timestep = self.sim.physics_timestep
        self.data = mujoco.MjData(self.model)

        self._actuator_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.spec.actuators
            ]
        )
        if np.any(self._actuator_ids < 0):
            missing = [
                name for name, i in zip(self.spec.actuators, self._actuator_ids) if i < 0
            ]
            raise RuntimeError(f"{self.spec.scene_xml} is missing actuators {missing}")
        self._joint_ids = self.model.actuator_trnid[self._actuator_ids, 0]
        self._qpos_adr = self.model.jnt_qposadr[self._joint_ids]

        self._cube_id = self.model.body("cube").id
        self._bin_id = self.model.body("bin").id
        self._cube_qpos_adr = self.model.jnt_qposadr[self.model.joint("cube_free").id]
        self._bin_qpos_adr = self.model.jnt_qposadr[self.model.joint("bin_free").id]
        self._gripper_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.spec.tcp_site
        )

        # Read the bin's inner floor height straight off the geometry rather
        # than repeating it as a constant that can drift from the scene.
        floor = self.model.geom("bin_floor")
        self.bin_floor_top = float(self.model.geom_pos[floor.id][2] + self.model.geom_size[floor.id][2])
        self.cube_half_extent = float(self.model.geom("cube").size[0])

        self.ctrl_low = self.model.actuator_ctrlrange[self._actuator_ids, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[self._actuator_ids, 1].copy()

        names = render_cameras
        if names is None:
            names = tuple(c.name for c in self.sim.policy_cameras)
        self.render_cameras = names
        self._camera_sizes = {c.name: (c.height, c.width) for c in self.sim.policy_cameras}
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

        # Geom groups used to detect an object spawning inside the arm.
        # The arm is whatever tree the tool site hangs off, whatever it is called.
        robot_root = self._root_body(int(self.model.site_bodyid[self._gripper_site]))
        self._robot_geoms = {
            g for g in range(self.model.ngeom)
            if self._root_body(self.model.geom_bodyid[g]) == robot_root
        }
        self._object_geoms = {
            g for g in range(self.model.ngeom)
            if self._root_body(self.model.geom_bodyid[g]) in (self._cube_id, self._bin_id)
        }

        self.rng = np.random.default_rng(seed)
        self._step_count = 0
        self._success_streak = 0
        goal = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")
        self._goal_mocap = int(self.model.body_mocapid[goal]) if goal >= 0 else -1
        eye = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chase_eye")
        look = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chase_look")
        self._chase_eye = int(self.model.body_mocapid[eye]) if eye >= 0 else -1
        self._chase_look = int(self.model.body_mocapid[look]) if look >= 0 else -1

        home = self.model.key(self.spec.home_key)
        self._home_qpos = home.qpos.copy()
        self._home_ctrl = home.ctrl.copy()

    # -- operator aids ------------------------------------------------------

    def set_goal_marker(self, position: np.ndarray | None) -> None:
        """Place the on-screen target ring, or hide it.

        A mocap body, so this never touches the physics and can never influence
        an episode: it exists purely so the operator can see where to go.
        """
        if self._goal_mocap < 0:
            return
        if position is None:
            self.data.mocap_pos[self._goal_mocap] = (0.0, 0.0, -1.0)
        else:
            self.data.mocap_pos[self._goal_mocap] = np.asarray(position, dtype=float)

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def __enter__(self) -> "PickPlaceEnv":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- placement ----------------------------------------------------------

    def _root_body(self, body_id: int) -> int:
        """Walk up to the outermost body below the world."""
        while self.model.body_parentid[body_id] != 0:
            body_id = self.model.body_parentid[body_id]
        return int(body_id)

    def objects_intersect_robot(self) -> bool:
        """Whether any spawned object is currently interpenetrating the arm.

        Without this check a bin can be placed inside the arm's folded home
        pose, get flung across the table the moment physics starts, and turn an
        otherwise fine episode into an unexplained failure.
        """
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            a, b = int(contact.geom1), int(contact.geom2)
            if (a in self._robot_geoms and b in self._object_geoms) or (
                b in self._robot_geoms and a in self._object_geoms
            ):
                return True
        return False


    def sample_layout(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Draw a cube pose, a bin pose and a cube yaw for one episode.

        How they are sampled is the robot's business: a cylindrical sector for
        an arm whose reach is dictated by a vertical shoulder axis, a box for
        one with joints to spare.
        """
        return self.layout.sample(self.rng, self.cube_half_extent)

    def reset(
        self,
        seed: int | None = None,
        cube_position: np.ndarray | None = None,
        bin_position: np.ndarray | None = None,
        cube_yaw: float | None = None,
    ) -> Observation:
        """Return the arm home and place the objects, optionally at fixed poses."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._home_qpos
        self.data.ctrl[:] = self._home_ctrl

        fixed = cube_position is not None and bin_position is not None
        for attempt in range(50):
            sampled_cube, sampled_bin, sampled_yaw = self.sample_layout()
            cube = sampled_cube if cube_position is None else np.asarray(cube_position, float)
            bin_pos = sampled_bin if bin_position is None else np.asarray(bin_position, float)
            yaw = sampled_yaw if cube_yaw is None else float(cube_yaw)

            a = self._cube_qpos_adr
            self.data.qpos[a : a + 3] = cube
            self.data.qpos[a + 3 : a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            b = self._bin_qpos_adr
            self.data.qpos[b : b + 3] = bin_pos
            self.data.qpos[b + 3 : b + 7] = [1.0, 0.0, 0.0, 0.0]

            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            if fixed or not self.objects_intersect_robot():
                break
        else:
            raise RuntimeError("could not place the objects clear of the arm")
        # Let the objects settle onto the table before the episode starts.
        for _ in range(int(0.2 / self.sim.physics_timestep)):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        self._success_streak = 0
        return self.observe()

    # -- stepping -----------------------------------------------------------

    @property
    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[self._qpos_adr].copy()

    @property
    def commanded_positions(self) -> np.ndarray:
        return self.data.ctrl[self._actuator_ids].copy()

    @property
    def cube_position(self) -> np.ndarray:
        return self.data.xpos[self._cube_id].copy()

    @property
    def bin_position(self) -> np.ndarray:
        return self.data.xpos[self._bin_id].copy()

    @property
    def gripper_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.data.site_xpos[self._gripper_site].copy(),
            self.data.site_xmat[self._gripper_site].reshape(3, 3).copy(),
        )

    def step(self, action: np.ndarray, observe: bool = True) -> StepResult:
        """Apply joint position targets for one control period.

        Args:
            action: (6,) target angles in radians, clipped to the actuator range.
            observe: render the cameras for the returned observation. Rendering
                costs about 8 ms per camera, so callers that already have the
                images they need -- teleoperation renders its own before the
                step -- should pass ``False`` rather than pay for a result they
                discard.
        """
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape[0] != len(self.spec.actuators):
            raise ValueError(
                f"action must have {len(self.spec.actuators)} elements for "
                f"{self.spec.name}, got {action.shape[0]}"
            )
        self.data.ctrl[self._actuator_ids] = np.clip(action, self.ctrl_low, self.ctrl_high)

        for _ in range(self.sim.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        success_now = self.cube_in_bin()
        self._success_streak = self._success_streak + 1 if success_now else 0
        success = self._success_streak >= self.sim.success_hold_steps
        done = success or self._step_count >= self.sim.max_episode_steps

        return StepResult(
            observation=self.observe() if observe else self.observe_state(),
            success=success,
            done=done,
            info={
                "cube_position": self.cube_position,
                "bin_position": self.bin_position,
                "success_streak": self._success_streak,
                "steps": self._step_count,
            },
        )

    # -- scoring ------------------------------------------------------------

    def cube_in_bin(self) -> bool:
        """Whether the cube currently rests inside the bin."""
        cube = self.cube_position
        bin_pos = self.bin_position
        planar = np.abs(cube[:2] - bin_pos[:2])
        within = bool(np.all(planar <= self.spec.success_tolerance))
        height = cube[2] - bin_pos[2]
        ceiling = self.bin_floor_top + 2.5 * self.cube_half_extent
        return within and self.bin_floor_top <= height <= ceiling

    # -- rendering ----------------------------------------------------------

    def _renderer(self, height: int, width: int) -> mujoco.Renderer:
        key = (height, width)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderers[key]

    def update_chase_camera(self) -> None:
        """Pin the chase rig to the gripper's current position."""
        if self._chase_eye < 0 or self._chase_look < 0:
            return
        tcp, _ = self.gripper_pose
        self.data.mocap_pos[self._chase_look] = tcp
        self.data.mocap_pos[self._chase_eye] = tcp + np.asarray(self.spec.chase_offset)
        # Mocap positions only reach body poses through kinematics, and camera
        # poses only follow from there. Without both, the chase view renders one
        # frame late and any projection through it is simply wrong.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def render(self, camera: str, height: int | None = None, width: int | None = None) -> np.ndarray:
        """Render one camera to a uint8 RGB array."""
        if camera == "chase_cam":
            self.update_chase_camera()
        if height is None or width is None:
            default = self._camera_sizes.get(camera, (self.sim.render_height, self.sim.render_width))
            height, width = default
        renderer = self._renderer(height, width)
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def observe(self) -> Observation:
        """Joint positions plus a freshly rendered image per camera."""
        return Observation(
            joint_positions=self.joint_positions,
            images={name: self.render(name) for name in self.render_cameras},
            time=float(self.data.time),
        )

    def observe_state(self) -> Observation:
        """Joint positions only. Cheap: nothing is rendered."""
        return Observation(
            joint_positions=self.joint_positions,
            images={},
            time=float(self.data.time),
        )
