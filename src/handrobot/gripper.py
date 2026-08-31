"""Measured relationship between the gripper command and the jaw gap.

Every arm expresses its gripper differently. The SO-101 has a hinged jaw driven
by a position servo whose command *is* a joint angle, and the gap is a chord, so
it is markedly non-linear. The Panda has two sliding fingers driven through a
tendon whose command runs 0 to 255. Deriving either mapping on paper is fiddly
and easy to get subtly wrong -- a straight-line fit to the SO-101 is out by
almost a centimetre in the middle of its range, which is the difference between
gripping a cube and closing beside it.

So neither is derived. Both are measured, by commanding the real actuator in the
real simulator and recording where the jaws actually end up. The result is
cached per robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from handrobot.paths import ASSETS_DIR
from handrobot.robots import RobotSpec, get_robot

SAMPLES = 41

#: Physics steps allowed for the jaws to reach a commanded opening.
SETTLE_STEPS = 400


def _point_position(model, data, name: str) -> np.ndarray:
    """Position of a named site, geom or body, whichever exists.

    Arms label the useful point on a jaw differently -- the Panda's finger
    bodies sit at the fingertips, while the SO-101's jaw bodies sit at their
    hinges and only its tip geoms are in the right place.
    """
    import mujoco

    for kind, xpos in (
        (mujoco.mjtObj.mjOBJ_SITE, data.site_xpos),
        (mujoco.mjtObj.mjOBJ_GEOM, data.geom_xpos),
        (mujoco.mjtObj.mjOBJ_BODY, data.xpos),
    ):
        index = mujoco.mj_name2id(model, kind, name)
        if index >= 0:
            return np.asarray(xpos[index]).copy()
    raise KeyError(f"no site, geom or body named {name!r}")


def calibration_path(spec: RobotSpec) -> Path:
    return ASSETS_DIR / "cache" / f"{spec.name}_gripper.npz"


@dataclass
class GripperCalibration:
    """Monotone lookup between the gripper actuator command and the jaw gap."""

    commands: np.ndarray
    """Actuator control values, ascending."""

    gaps: np.ndarray
    """Measured jaw separation in metres, ascending alongside ``commands``."""

    robot: str = "unknown"

    @property
    def command_min(self) -> float:
        return float(self.commands[0])

    @property
    def command_max(self) -> float:
        return float(self.commands[-1])

    @property
    def gap_min(self) -> float:
        return float(self.gaps[0])

    @property
    def gap_max(self) -> float:
        return float(self.gaps[-1])

    def command_to_gap(self, command: float) -> float:
        """Jaw gap the actuator settles at for this command, in metres."""
        return float(np.interp(float(command), self.commands, self.gaps))

    def gap_to_command(self, gap: float) -> float:
        """Actuator command that produces this jaw gap."""
        return float(np.interp(float(gap), self.gaps, self.commands))

    # Older names, kept so nothing has to care whether a robot's command
    # happens to be an angle.
    def q_to_gap(self, command: float) -> float:
        return self.command_to_gap(command)

    def gap_to_q(self, gap: float) -> float:
        return self.gap_to_command(gap)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, commands=self.commands, gaps=self.gaps,
                            robot=np.array(self.robot))
        return path

    @classmethod
    def load(cls, path: Path | str) -> "GripperCalibration":
        with np.load(Path(path)) as raw:
            return cls(commands=raw["commands"], gaps=raw["gaps"],
                       robot=str(raw["robot"]) if "robot" in raw.files else "unknown")

    @classmethod
    def measure(cls, spec: RobotSpec | None = None) -> "GripperCalibration":
        """Command the gripper across its range and record where the jaws settle.

        Uses the full scene rather than the arm alone, so the measurement
        includes whatever the actuator really does under gravity and contact.
        """
        import mujoco

        spec = spec or get_robot()
        model = mujoco.MjModel.from_xml_path(str(spec.scene_xml))
        data = mujoco.MjData(model)

        actuator = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.gripper_actuator
        )
        low, high = model.actuator_ctrlrange[actuator]
        home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, spec.home_key)
        left, right = spec.jaw_bodies

        commands = np.linspace(low, high, SAMPLES)
        gaps = np.empty_like(commands)
        for i, command in enumerate(commands):
            mujoco.mj_resetDataKeyframe(model, data, home)
            # Move the objects far away so they cannot get between the jaws.
            data.qpos[model.nq - 14 :] = 0.0
            data.qpos[model.nq - 14 : model.nq - 14 + 3] = [5.0, 5.0, 5.0]
            data.qpos[model.nq - 7 : model.nq - 7 + 3] = [-5.0, -5.0, 5.0]
            data.ctrl[actuator] = command
            for _ in range(SETTLE_STEPS):
                mujoco.mj_step(model, data)
            gaps[i] = np.linalg.norm(
                _point_position(model, data, left) - _point_position(model, data, right)
            )

        order = np.argsort(gaps)
        commands, gaps = commands[order], gaps[order]
        keep = np.concatenate([[True], np.diff(gaps) > 1e-6])
        return cls(commands=commands[keep], gaps=gaps[keep], robot=spec.name)

    @classmethod
    def cached(cls, spec: RobotSpec | None = None, rebuild: bool = False) -> "GripperCalibration":
        spec = spec or get_robot()
        path = calibration_path(spec)
        if path.exists() and not rebuild:
            calibration = cls.load(path)
            if calibration.robot == spec.name:
                return calibration
        calibration = cls.measure(spec)
        calibration.save(path)
        return calibration
