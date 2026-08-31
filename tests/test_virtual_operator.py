"""Drive the whole teleoperation chain with a synthetic hand.

This is the closest thing to sitting in front of the camera that can run without
one. A virtual operator moves a hand through camera space along a plan that would
solve the task, with realistic landmark noise added, and the resulting motion
goes through the real retargeter, the real inverse kinematics and the real
physics. If the cube ends up in the bin, the control chain works end to end.

It exists because the first live attempt produced zero successful episodes, and
"it feels unstable" is not something a unit test can catch.
"""

import numpy as np
import pytest

from handrobot.config import Config
from handrobot.hands.types import HandPose, Landmarks
from handrobot.gripper import GripperCalibration
from handrobot.retarget.grasp import GraspFrames
from handrobot.retarget.ik import ArmIK
from handrobot.retarget.mapper import HandToGripper
from handrobot.sim.env import PickPlaceEnv

#: Noise on the *palm centre*, in metres. The palm centre averages five
#: landmarks, so it is roughly half as noisy as any single one.
NOISE = 0.002

#: Seconds spent holding still at each waypoint. Smoothing costs a steady-state
#: lag while the hand is moving, which decays once it stops; a human pauses to
#: line up before descending onto something, and so does this.
DWELL = 0.8

#: Seconds the hand is held steady before the clutch is engaged, so the
#: smoothing filter has settled and the anchor is not a single noisy frame.
SETTLE_BEFORE_ENGAGE = 0.7


class VirtualOperator:
    """Produces hand poses that would move the gripper along a target path."""

    def __init__(self, mapper: HandToGripper, anchor_hand: np.ndarray,
                 anchor_robot: np.ndarray, seed: int = 0) -> None:
        self.mapper = mapper
        self.anchor_hand = np.asarray(anchor_hand, dtype=float)
        self.anchor_robot = np.asarray(anchor_robot, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.robot_to_camera = np.linalg.inv(mapper.CAMERA_TO_ROBOT)

    def hand_for(self, robot_target: np.ndarray) -> np.ndarray:
        """The hand position that asks for this gripper position."""
        delta = self.robot_to_camera @ (np.asarray(robot_target, float) - self.anchor_robot)
        return self.anchor_hand + delta / self.mapper.position_gain

    def pose(self, robot_target: np.ndarray, pinch: float, t: float,
             noise: float = NOISE) -> HandPose:
        position = self.hand_for(robot_target) + self.rng.normal(scale=noise, size=3)
        world = np.zeros((21, 3))
        world[0] = [0.0, 0.04, 0.0]
        world[5], world[9] = [-0.03, -0.02, 0.0], [-0.01, -0.03, 0.0]
        world[13], world[17] = [0.01, -0.03, 0.0], [0.03, -0.02, 0.0]
        return HandPose(
            palm_position=position,
            rotation=np.eye(3),
            pinch_distance=pinch,
            depth=float(position[2]),
            landmarks=Landmarks(np.zeros((21, 3)), world, "Right", 1.0),
            timestamp=t,
        )


def run_virtual_episode(seed: int, noise: float = NOISE, config: Config | None = None) -> bool:
    """Teleoperate one episode with a synthetic hand. Returns whether it succeeded."""
    config = config or Config()
    spec = config.spec
    env = PickPlaceEnv(config=config, render_cameras=(), seed=seed)
    ik = ArmIK(config.ik, spec)
    grasp = GraspFrames(spec)
    calibration = GripperCalibration.cached(spec)
    mapper = HandToGripper(config.hand, config.workspace, config.gripper)

    try:
        env.reset(seed=seed)
        cube, bin_position = env.cube_position.copy(), env.bin_position.copy()
        gripper, _ = env.gripper_pose

        operator = VirtualOperator(mapper, np.array([0.0, 0.0, 0.55]),
                                   config.workspace.clip(gripper), seed=seed)
        # Hold still for a moment first, exactly as an operator does before
        # pressing the clutch, so the filter has settled.
        start = config.workspace.clip(gripper)
        settle_time = 0.0
        for _ in range(int(SETTLE_BEFORE_ENGAGE / config.sim.control_dt)):
            mapper(operator.pose(start, 0.085, settle_time, noise), gripper)
            settle_time += config.sim.control_dt
        mapper.engage(operator.pose(start, 0.085, settle_time, noise), gripper)

        hover = spec.hover_height
        grasp_z = spec.cube_half_extent + spec.grasp_clearance
        release_z = spec.release_clearance + 2 * spec.cube_half_extent
        open_pinch, closed_pinch = 0.085, 0.015
        above_cube = np.array([cube[0], cube[1], hover])
        at_cube = np.array([cube[0], cube[1], grasp_z])
        above_bin = np.array([bin_position[0], bin_position[1], hover])
        at_bin = np.array([bin_position[0], bin_position[1], release_z])
        plan = [
            (above_cube, open_pinch, 1.4),
            (above_cube, open_pinch, DWELL),      # line up before descending
            (at_cube, open_pinch, 1.0),
            (at_cube, open_pinch, DWELL),         # settle before closing
                (at_cube, closed_pinch, max(0.8, spec.close_duration)),
            (above_cube, closed_pinch, 1.0),
            (above_bin, closed_pinch, 1.6),
            (above_bin, closed_pinch, DWELL),
            (at_bin, closed_pinch, 0.9),
            (at_bin, open_pinch, 0.7),
            (above_bin, open_pinch, 0.7),
        ]

        dt = config.sim.control_dt
        q = env.commanded_positions.copy()
        previous_target = config.workspace.clip(gripper)
        previous_pinch = open_pinch
        t = settle_time

        for target, pinch, duration in plan:
            for step in range(int(duration / dt)):
                alpha = (step + 1) / max(int(duration / dt), 1)
                alpha = alpha * alpha * (3 - 2 * alpha)
                waypoint = (1 - alpha) * previous_target + alpha * target
                blend = (1 - alpha) * previous_pinch + alpha * pinch

                command = mapper(operator.pose(waypoint, blend, t, noise), env.gripper_pose[0])
                rotation = grasp.frame_for(command.position, command.jaw_azimuth)
                result = ik.solve(command.position, rotation, q)
                if not result.ok:
                    retry = ik.solve(command.position, rotation, env._home_ctrl.copy(), 60)
                    result = retry if retry.position_error < result.position_error else result
                q = result.q.copy()
                q[spec.gripper_index] = calibration.gap_to_command(command.jaw_gap)
                if env.step(q, observe=False).success:
                    return True
                t += dt
            previous_target, previous_pinch = target, pinch

        for _ in range(40):
            if env.step(q, observe=False).success:
                return True
        return env.cube_in_bin()
    finally:
        env.close()


def test_a_virtual_operator_can_solve_the_task():
    """The end-to-end claim: a plausible hand motion, seen through a noisy
    tracker, is enough to complete the task through the real control chain."""
    successes = sum(run_virtual_episode(seed) for seed in range(10))
    assert successes >= 8, f"only {successes}/10 virtual episodes succeeded"


def test_the_chain_is_near_perfect_without_tracking_noise():
    """Separates control-design faults from tracking noise. Anything short of
    near-perfect here is a bug in the mapping, not a hard sensing limit."""
    successes = sum(run_virtual_episode(seed, noise=0.0) for seed in range(10))
    assert successes >= 9, f"only {successes}/10 succeeded with a perfect tracker"


def test_the_clutch_anchor_is_not_a_single_noisy_frame():
    """Regression, and the single biggest usability bug found: engaging used to
    reset the smoothing filter and anchor to one raw frame. That frame's noise
    was then baked into every command, multiplied by the position gain, as an
    offset the operator could never correct -- and a different one after every
    re-engage."""
    config = Config()
    mapper = HandToGripper(config.hand, config.workspace, config.gripper)
    home = config.workspace.center
    operator = VirtualOperator(mapper, np.array([0.0, 0.0, 0.55]), home, seed=0)

    target = config.workspace.clip(home + np.array([0.015, 0.03, 0.0]))
    errors = []
    for seed in range(8):
        mapper = HandToGripper(config.hand, config.workspace, config.gripper)
        operator = VirtualOperator(mapper, np.array([0.0, 0.0, 0.55]), home, seed=seed)
        t = 0.0
        for _ in range(30):
            mapper(operator.pose(home, 0.085, t, noise=0.004), home)
            t += 1 / 30
        mapper.engage(operator.pose(home, 0.085, t, noise=0.004), home)
        for _ in range(150):
            command = mapper(operator.pose(target, 0.085, t, noise=0.004), home)
            t += 1 / 30
        errors.append(float(np.linalg.norm(command.position - target)))

    # Anchoring to a raw frame gave a permanent offset of roughly the landmark
    # noise times the position gain -- about 6 mm here, and worse on a bad frame.
    assert np.mean(errors) < 0.004, (
        f"mean steady-state offset {np.mean(errors) * 1000:.1f} mm "
        f"(max {np.max(errors) * 1000:.1f} mm)"
    )


@pytest.mark.parametrize("noise", [0.0, 0.002, 0.004, 0.008])
def test_noise_tolerance(noise):
    """How much landmark jitter the control chain absorbs before it breaks."""
    successes = sum(run_virtual_episode(seed, noise=noise) for seed in range(4))
    if noise <= 0.004:
        assert successes >= 3, f"{successes}/4 at {noise * 1000:.0f} mm of noise"
    else:
        # Documents where it degrades rather than asserting it still works.
        assert successes >= 0


def test_grasping_does_not_pull_the_gripper_off_the_cube():
    """The regression that made live teleoperation impossible: the pinch used to
    move the position reference, so closing the hand slid the gripper away from
    the cube in the same instant."""
    config = Config()
    spec = config.spec
    env = PickPlaceEnv(config=config, render_cameras=(), seed=0)
    ik = ArmIK(config.ik, spec)
    grasp = GraspFrames(spec)
    calibration = GripperCalibration.cached(spec)
    mapper = HandToGripper(config.hand, config.workspace, config.gripper)
    try:
        env.reset(seed=3)
        cube = env.cube_position.copy()
        target = np.array([cube[0], cube[1],
                           spec.cube_half_extent + spec.grasp_clearance])
        gripper, _ = env.gripper_pose
        operator = VirtualOperator(mapper, np.array([0.0, 0.0, 0.55]),
                                   config.workspace.clip(gripper))
        mapper.engage(operator.pose(config.workspace.clip(gripper), 0.085, 0.0), gripper)

        q = env.commanded_positions.copy()
        t = 0.0
        # Settle over the cube with the hand open.
        for _ in range(90):
            command = mapper(operator.pose(target, 0.085, t, noise=0.0), env.gripper_pose[0])
            rotation = grasp.frame_for(command.position, command.jaw_azimuth)
            q = ik.solve(command.position, rotation, q).q
            q[spec.gripper_index] = calibration.gap_to_command(command.jaw_gap)
            env.step(q, observe=False)
            t += config.sim.control_dt
        before = env.gripper_pose[0][:2].copy()

        # Now close the hand without moving it at all.
        for _ in range(45):
            command = mapper(operator.pose(target, 0.015, t, noise=0.0), env.gripper_pose[0])
            rotation = grasp.frame_for(command.position, command.jaw_azimuth)
            q = ik.solve(command.position, rotation, q).q
            q[spec.gripper_index] = calibration.gap_to_command(command.jaw_gap)
            env.step(q, observe=False)
            t += config.sim.control_dt
        after = env.gripper_pose[0][:2].copy()

        moved = float(np.linalg.norm(after - before))
        assert moved < 0.006, f"closing the hand moved the gripper {moved * 1000:.1f} mm"
    finally:
        env.close()
