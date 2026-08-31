"""Differential inverse kinematics for the SO-101 arm.

The solver runs on the *arm-only* MJCF. Loading the full task scene here would
expose the free joints of the cube and the bin as extra degrees of freedom, and
the optimiser would happily "solve" a pose by teleporting the cube.

The SO-101 has five joints before the gripper, so an arbitrary six-DoF target
pose is generally unreachable. The orientation cost is therefore set below the
position cost: when the two conflict, the gripper goes where it was asked and
tilts as close as it can.
"""

from __future__ import annotations

from dataclasses import dataclass

import warnings

import mink
import mujoco
import numpy as np

from handrobot.config import IKConfig
from handrobot.geometry import orthonormalize
from handrobot.robots import RobotSpec, get_robot


@dataclass(frozen=True)
class IKResult:
    """Outcome of one inverse-kinematics solve."""

    q: np.ndarray
    """Full six-element joint vector; element 5 (the gripper) is passed through."""

    position_error: float
    """Euclidean distance between the requested and achieved site position, in metres."""

    orientation_error: float
    """Magnitude of the residual rotation vector, in radians."""

    ok: bool
    """Whether both residuals are within the configured tolerances."""


class ArmIK:
    """Warm-started differential IK for the SO-101 gripper site."""

    def __init__(
        self,
        config: IKConfig | None = None,
        spec: RobotSpec | None = None,
        posture_target: np.ndarray | None = None,
    ) -> None:
        self.config = config or IKConfig()
        self.spec = spec or get_robot()
        self.model = mujoco.MjModel.from_xml_path(str(self.spec.arm_xml))
        self.data = mujoco.MjData(self.model)
        self._site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.spec.tcp_site
        )
        if self._site_id < 0:
            raise RuntimeError(
                f"site {self.spec.tcp_site!r} not found in {self.spec.arm_xml}"
            )
        self.n_arm_joints = self.spec.n_arm_joints

        self._configuration = mink.Configuration(self.model)
        self._frame_task = mink.FrameTask(
            self.spec.tcp_site,
            "site",
            position_cost=self.config.position_cost,
            orientation_cost=self.config.orientation_cost,
            lm_damping=self.config.lm_damping,
        )
        self._posture_task = mink.PostureTask(self.model, cost=self.config.posture_cost)
        self._tasks: list = [self._frame_task, self._posture_task]

        # Freeze anything past the arm joints -- the gripper's finger degrees of
        # freedom. Leaving them in the optimisation lets the solver "reach" a
        # target by opening the hand, and on a tendon-coupled gripper it also
        # makes the QP weights degenerate.
        extra = list(range(self.n_arm_joints, self.model.nv))
        if extra:
            self._tasks.append(mink.DofFreezingTask(self.model, extra, gain=1.0))

        self._limits = [mink.ConfigurationLimit(self.model)]

        # The posture task resists changing shape, rather than pulling towards
        # any particular one. Pulling towards a fixed home pose was tried and is
        # worse: it competes directly with reaching the target, and at any cost
        # high enough to steady a redundant arm it drags the tool centimetres
        # off where it was asked to go.
        self.posture_target = (
            None if posture_target is None else np.asarray(posture_target, dtype=float)
        )

        # Clip to the intersection of the joint limits and the actuator command
        # ranges. The MJCF rounds them differently, and the joint range is a few
        # microradians wider -- enough for a solution to be silently clamped by
        # the simulator afterwards, so that the commanded and executed actions
        # in a recorded dataset would disagree.
        n = self.n_arm_joints
        joint_range = self.model.jnt_range[:n]
        control_range = self.model.actuator_ctrlrange[:n]
        self.joint_low = np.maximum(joint_range[:, 0], control_range[:, 0]).copy()
        self.joint_high = np.minimum(joint_range[:, 1], control_range[:, 1]).copy()

    def _home_reference(self) -> np.ndarray:
        """The arm's home joint angles, read from the task scene."""
        import mujoco as _mujoco

        try:
            scene = _mujoco.MjModel.from_xml_path(str(self.spec.scene_xml))
            return scene.key(self.spec.home_key).qpos[: self.n_arm_joints].copy()
        except Exception:
            return np.zeros(self.n_arm_joints)

    # -- forward kinematics -------------------------------------------------

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the tool site (position, rotation) for a joint vector."""
        q = np.asarray(q, dtype=float)
        self.data.qpos[: min(len(q), self.model.nq)] = q[: self.model.nq]
        mujoco.mj_kinematics(self.model, self.data)
        pos = self.data.site_xpos[self._site_id].copy()
        rot = self.data.site_xmat[self._site_id].reshape(3, 3).copy()
        return pos, rot

    # -- inverse kinematics -------------------------------------------------

    def solve(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        q_init: np.ndarray,
        iterations: int | None = None,
    ) -> IKResult:
        """Solve for joint angles placing the gripper site at the requested pose.

        Args:
            target_position: desired site origin in the robot base frame.
            target_rotation: desired site orientation, columns are the site's
                x (approach), y and z (jaw-opening) axes in the base frame.
            q_init: warm start. Element 5 is carried through untouched so the
                caller keeps control of the gripper.
            iterations: solver iterations; defaults to the configured value.

        Returns:
            An :class:`IKResult`. ``ok`` is False when the residual exceeds the
            configured tolerance, in which case ``q`` still holds the best
            solution found and the caller decides whether to use it.
        """
        cfg = self.config
        iterations = cfg.iterations if iterations is None else iterations

        q_init = np.asarray(q_init, dtype=float).copy()
        n = self.n_arm_joints
        if q_init.shape[0] < n:
            raise ValueError(f"q_init must have at least {n} elements, got {q_init.shape[0]}")
        arm = np.clip(q_init[:n], self.joint_low, self.joint_high)

        full = np.zeros(self.model.nq)
        full[:n] = arm
        # Leave any finger joints where the model's own defaults put them; the
        # gripper is commanded separately and must not be solved for.
        self._configuration.update(full.copy())
        reference = full.copy()
        if self.posture_target is not None:
            reference[: len(self.posture_target)] = self.posture_target
        self._posture_task.set_target(reference)
        target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(orthonormalize(target_rotation)),
            np.asarray(target_position, dtype=float),
        )
        self._frame_task.set_target(target)

        # A target exactly a half turn from the current pose sits on the
        # singularity of the rotation logarithm, and the QP assembly divides by
        # zero there. It is recoverable -- the next iteration steps off it -- so
        # the warning is silenced rather than the solve abandoned.
        for _ in range(iterations):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                velocity = mink.solve_ik(
                    self._configuration,
                    self._tasks,
                    cfg.integration_dt,
                    cfg.solver,
                    1e-10,
                    limits=self._limits,
                )
            if not np.all(np.isfinite(velocity)):
                break
            self._configuration.integrate_inplace(velocity, cfg.integration_dt)

        error = self._frame_task.compute_error(self._configuration)
        position_error = float(np.linalg.norm(error[:3]))
        orientation_error = float(np.linalg.norm(error[3:]))

        q = q_init.copy()
        solved = np.clip(self._configuration.q[:n], self.joint_low, self.joint_high)
        # Never leap away from the warm start: see IKConfig.max_joint_step.
        # The extra clip is belt-and-braces: with an in-range warm start the
        # step already lands between two in-range points, but a caller who
        # passes an out-of-range one must not have it preserved.
        step = np.clip(solved - arm, -cfg.max_joint_step, cfg.max_joint_step)
        q[:n] = np.clip(arm + step, self.joint_low, self.joint_high)

        ok = (
            position_error <= cfg.max_position_error
            and orientation_error <= cfg.max_orientation_error
        )
        return IKResult(q=q, position_error=position_error,
                        orientation_error=orientation_error, ok=ok)
