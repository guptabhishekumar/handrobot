import numpy as np
import pytest

from handrobot.config import Config
from handrobot.rollout import ScriptedController, evaluate_controller, run_episode
from handrobot.scripted import ScriptedExpert


@pytest.fixture
def expert(config):
    return ScriptedExpert(config)


def test_plan_covers_the_full_pick_and_place_cycle(env, expert):
    env.reset(seed=0)
    labels = [w.label for w in expert.plan(env)]
    assert labels == ["approach", "descend", "close", "lift", "transfer",
                      "lower", "release", "retreat"]


def test_plan_positions_track_the_objects(env, expert):
    env.reset(seed=11)
    waypoints = {w.label: w for w in expert.plan(env)}
    assert np.allclose(waypoints["descend"].position[:2], env.cube_position[:2], atol=2e-3)
    assert np.allclose(waypoints["lower"].position[:2], env.bin_position[:2], atol=2e-3)
    assert waypoints["descend"].position[2] < waypoints["approach"].position[2]


def test_every_waypoint_is_inside_the_workspace(env, expert):
    workspace = env.config.workspace
    for seed in range(20):
        env.reset(seed=seed)
        for waypoint in expert.plan(env):
            assert workspace.contains(waypoint.position, tolerance=1e-6), (
                f"{waypoint.label} at seed {seed} is outside the workspace"
            )


def test_grasp_azimuth_picks_a_face_of_the_cube(expert):
    """A cube has four-fold symmetry, so any quarter turn is an equally good grasp."""
    position = expert.config.workspace.center
    for yaw in np.linspace(-np.pi, np.pi, 17):
        azimuth = expert.grasp_azimuth(yaw, position)
        remainder = (azimuth - yaw) % (np.pi / 2)
        assert min(remainder, np.pi / 2 - remainder) < 1e-9


def test_carry_rotation_stays_within_the_wrist_range(env, expert):
    """Holding a fixed world jaw angle across the base exhausts the wrist roll."""
    for seed in range(20):
        env.reset(seed=seed)
        waypoints = {w.label: w for w in expert.plan(env)}
        from handrobot.retarget.mapper import wrap_to_pi

        change = abs(
            wrap_to_pi(waypoints["lower"].jaw_azimuth - waypoints["close"].jaw_azimuth)
        )
        assert change <= np.pi / 2 + 1e-9


def test_jaw_azimuths_are_wrapped(env, expert):
    for seed in range(20):
        env.reset(seed=seed)
        for waypoint in expert.plan(env):
            assert -np.pi < waypoint.jaw_azimuth <= np.pi + 1e-12


def test_grip_closes_and_release_opens(env, expert):
    env.reset(seed=0)
    waypoints = {w.label: w for w in expert.plan(env)}
    assert waypoints["close"].jaw_gap < waypoints["approach"].jaw_gap
    assert waypoints["release"].jaw_gap > waypoints["lower"].jaw_gap


def test_interpolation_is_continuous_across_waypoints(env, expert):
    env.reset(seed=0)
    expert.reset(env)
    times = np.linspace(0, expert.duration, 400)
    positions = np.array([expert.target_at(t)[0] for t in times])
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    assert steps.max() < 0.02, "the scripted path jumps between waypoints"


def test_target_is_clamped_past_the_end_of_the_plan(env, expert):
    env.reset(seed=0)
    expert.reset(env)
    last = expert.target_at(expert.duration)
    beyond = expert.target_at(expert.duration + 5.0)
    assert np.allclose(last[0], beyond[0])
    assert beyond[3] == "retreat"


def test_expert_solves_the_task(env, expert):
    """The headline claim: the scripted plan actually completes the task."""
    controller = ScriptedController(expert)
    results = evaluate_controller(env, controller, episodes=20, seed=4000, verbose=False)
    assert results["success_rate"] >= 0.95, results


def test_expert_actions_stay_within_the_actuator_range(env, expert):
    controller = ScriptedController(expert)
    seen = []
    run_episode(env, controller, seed=0, on_step=lambda i, o, a: seen.append(a))
    actions = np.array(seen)
    assert np.all(actions >= env.ctrl_low - 1e-6)
    assert np.all(actions <= env.ctrl_high + 1e-6)


def test_expert_actions_are_smooth(env, expert):
    """Jerky demonstrations teach a jerky policy, so this is worth pinning down."""
    controller = ScriptedController(expert)
    seen = []
    run_episode(env, controller, seed=1, on_step=lambda i, o, a: seen.append(a))
    deltas = np.abs(np.diff(np.array(seen), axis=0))
    arm = deltas[:, : env.spec.n_arm_joints]
    assert arm.max() < 0.35, f"largest single-step joint jump {arm.max():.3f} rad"
