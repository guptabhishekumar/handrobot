import numpy as np
import pytest


def converge(ik, target, rotation, start, steps: int = 60):
    """Solve repeatedly, the way the control loop does.

    A single solve is deliberately step-limited so a redundant arm cannot flip
    through its null space in one control period, so reaching a distant pose
    takes several calls.
    """
    q = np.asarray(start, dtype=float).copy()
    result = None
    for _ in range(steps):
        result = ik.solve(target, rotation, q)
        q = result.q
    return result

from handrobot.geometry import rotation_geodesic, top_down_frame




def test_forward_kinematics_is_deterministic(ik, rest):
    a = ik.forward(rest)
    b = ik.forward(rest)
    assert np.allclose(a[0], b[0])
    assert np.allclose(a[1], b[1])


def test_solve_reaches_a_pose_it_previously_produced(ik, rest):
    """Round trip: take a real reachable pose from FK, then ask IK to find it again."""
    q = rest.copy()
    q[: ik.n_arm_joints] += 0.12
    position, rotation = ik.forward(q)
    result = converge(ik, position, rotation, rest)
    assert result.ok
    assert result.position_error < 2e-3
    assert result.orientation_error < 5e-2


def test_gripper_joint_is_passed_through_untouched(ik, rest, spec):
    q_init = rest.copy()
    index = spec.gripper_index
    q_init[index] = 0.5 * (q_init[index] + 1.0)
    target = spec.workspace.center
    result = converge(ik, target, spec.grasp_rotation(np.array([0.0, 0.0, -1.0]),
                                                      np.array([0.0, 1.0, 0.0])), q_init)
    assert result.q[index] == pytest.approx(q_init[index])


def test_solution_respects_joint_limits(ik, rest):
    # A target far outside the arm's reach: the solver must saturate, not escape.
    result = ik.solve(np.array([2.5, 0.0, 1.6]), top_down_frame(0.0), rest, iterations=60)
    assert not result.ok
    n = ik.n_arm_joints
    assert np.all(result.q[:n] >= ik.joint_low - 1e-9)
    assert np.all(result.q[:n] <= ik.joint_high + 1e-9)


def test_solve_rejects_a_wrongly_sized_warm_start(ik):
    with pytest.raises(ValueError):
        ik.solve(np.array([0.2, 0.0, 0.1]), np.eye(3), np.zeros(3))


def test_achieved_orientation_matches_the_request_inside_the_workspace(ik, rest, config):
    target = config.workspace.center
    rotation = config.spec.grasp_rotation(np.array([0.0, 0.0, -1.0]),
                                          np.array([np.cos(1.0), np.sin(1.0), 0.0]))
    result = converge(ik, target, rotation, rest)
    _, achieved = ik.forward(result.q)
    assert rotation_geodesic(rotation, achieved) < 0.08
