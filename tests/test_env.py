import numpy as np
import pytest

from handrobot.sim.env import PickPlaceEnv


def test_reset_is_reproducible_for_a_seed(env):
    env.reset(seed=42)
    cube_a, bin_a = env.cube_position.copy(), env.bin_position.copy()
    env.reset(seed=42)
    assert np.allclose(env.cube_position, cube_a, atol=1e-6)
    assert np.allclose(env.bin_position, bin_a, atol=1e-6)


def test_reset_varies_across_seeds(env):
    env.reset(seed=1)
    first = env.cube_position.copy()
    env.reset(seed=2)
    assert not np.allclose(env.cube_position, first, atol=1e-3)


def test_sampled_layouts_keep_cube_and_bin_apart(env):
    minimum = env.config.randomization.min_separation
    for seed in range(30):
        env.reset(seed=seed)
        distance = np.linalg.norm(env.cube_position[:2] - env.bin_position[:2])
        assert distance >= minimum - 0.02


def test_sampled_objects_stay_inside_the_workspace(env):
    """Every spawned object must be somewhere the arm can actually grasp it."""
    workspace = env.config.workspace
    for seed in range(40):
        env.reset(seed=seed)
        cube = env.cube_position.copy()
        cube[2] = workspace.low[2]
        assert workspace.contains(cube, tolerance=1e-3), f"cube out of reach at seed {seed}"


def test_objects_never_spawn_inside_the_arm(env):
    """A bin placed inside the folded arm gets flung across the table on step one."""
    for seed in range(40):
        env.reset(seed=seed)
        assert not env.objects_intersect_robot(), f"object intersects the arm at seed {seed}"
        assert env.bin_position[2] < 0.01, f"bin was displaced at seed {seed}"


def test_objects_start_resting_on_the_table(env):
    env.reset(seed=3)
    assert env.cube_position[2] == pytest.approx(env.cube_half_extent, abs=2e-3)
    assert env.bin_position[2] == pytest.approx(0.0, abs=2e-3)


def test_action_is_clipped_to_the_actuator_range(env):
    env.reset(seed=0)
    n = len(env.spec.actuators)
    env.step(np.full(n, 1e3))
    assert np.all(env.commanded_positions <= env.ctrl_high + 1e-9)
    env.step(np.full(n, -1e3))
    assert np.all(env.commanded_positions >= env.ctrl_low - 1e-9)


def test_action_shape_is_validated(env):
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.zeros(3))


def test_episode_terminates_at_the_step_limit(env):
    env.reset(seed=0)
    hold = env.joint_positions
    limit = env.config.sim.max_episode_steps
    for _ in range(limit - 1):
        assert not env.step(hold).done
    assert env.step(hold).done


def _layout(env):
    return env.spec.layout.sample(np.random.default_rng(0), env.cube_half_extent)


def test_success_requires_the_cube_to_be_in_the_bin(env):
    cube, bin_position, _ = _layout(env)
    env.reset(seed=0, cube_position=cube, bin_position=bin_position)
    assert not env.cube_in_bin()

    # Teleport the cube into the bin and let it settle.
    address = env._cube_qpos_adr
    env.data.qpos[address : address + 3] = [
        bin_position[0], bin_position[1], env.bin_floor_top + env.cube_half_extent + 0.005
    ]
    env.data.qvel[:] = 0.0
    hold = env.commanded_positions
    for _ in range(30):
        env.step(hold)
    assert env.cube_in_bin()


def test_success_needs_to_be_held_before_the_episode_ends(env):
    cube, bin_position, _ = _layout(env)
    env.reset(seed=0, cube_position=cube, bin_position=bin_position)
    address = env._cube_qpos_adr
    env.data.qpos[address : address + 3] = [
        bin_position[0], bin_position[1], env.bin_floor_top + env.cube_half_extent
    ]
    env.data.qvel[:] = 0.0
    hold = env.commanded_positions
    result = env.step(hold)
    assert not result.success  # one step is not enough
    for _ in range(env.config.sim.success_hold_steps):
        result = env.step(hold)
    assert result.success


def test_gripper_gap_calibration_matches_the_simulated_jaws(config):
    """The commanded gap must be the gap the jaws actually reach."""
    from handrobot.gripper import GripperCalibration, _point_position

    calibration = GripperCalibration.cached(config.spec)
    left, right = config.spec.jaw_bodies
    env = PickPlaceEnv(config=config, render_cameras=())
    try:
        low, high = calibration.gap_min, calibration.gap_max
        for gap in np.linspace(low + 0.005, high - 0.005, 5):
            env.reset(seed=0, cube_position=np.array([2.0, 2.0, 0.5]),
                      bin_position=np.array([-2.0, -2.0, 0.0]))
            q = env.commanded_positions.copy()
            q[config.spec.gripper_index] = calibration.gap_to_command(gap)
            for _ in range(120):
                env.step(q)
            measured = float(np.linalg.norm(
                _point_position(env.model, env.data, left)
                - _point_position(env.model, env.data, right)
            ))
            assert measured == pytest.approx(gap, abs=0.003), (
                f"commanded {gap * 1000:.0f} mm, measured {measured * 1000:.1f} mm"
            )
    finally:
        env.close()


def test_observation_images_have_the_configured_shape(config):
    env = PickPlaceEnv(config=config)
    try:
        observation = env.reset(seed=0)
        for camera in config.sim.policy_cameras:
            image = observation.images[camera.name]
            assert image.shape == (camera.height, camera.width, 3)
            assert image.dtype == np.uint8
            assert image.std() > 1.0, f"{camera.name} rendered a flat image"
    finally:
        env.close()


def test_success_tolerance_is_inside_the_bin_walls(config):
    """A tolerance wider than the bin would count an object balanced on the rim."""
    assert config.spec.success_tolerance < config.spec.bin_inner_half


def test_a_cube_on_the_rim_does_not_count_as_success(env):
    cube, bin_position, _ = _layout(env)
    env.reset(seed=0, cube_position=cube, bin_position=bin_position)
    address = env._cube_qpos_adr
    # Perched on the wall: just outside the opening, at rim height.
    outside = env.spec.bin_inner_half + env.cube_half_extent + 0.005
    env.data.qpos[address : address + 3] = [
        bin_position[0], bin_position[1] + outside, env.bin_floor_top + env.cube_half_extent
    ]
    env.data.qvel[:] = 0.0
    hold = env.commanded_positions
    for _ in range(5):
        env.step(hold)
    assert not env.cube_in_bin()


def test_recorded_action_equals_the_executed_command(env):
    """If the simulator clipped a commanded action, the dataset would be a lie."""
    from handrobot.scripted import ScriptedExpert

    expert = ScriptedExpert(env.config)
    env.reset(seed=5)
    expert.reset(env)
    for _ in range(120):
        action = expert.act(env)
        env.step(action)
        assert np.array_equal(env.commanded_positions, action)


def test_step_can_skip_rendering(env):
    """Rendering costs about 8 ms per camera; callers that discard the images
    must be able to opt out."""
    cheap = env.step(env.commanded_positions, observe=False)
    assert cheap.observation.images == {}
    assert cheap.observation.joint_positions.shape == (len(env.spec.actuators),)

    full = env.step(env.commanded_positions, observe=True)
    assert set(full.observation.images) == set(env.render_cameras)


def test_skipping_rendering_does_not_change_the_physics(env):
    env.reset(seed=9)
    hold = env.commanded_positions.copy()
    for _ in range(30):
        env.step(hold, observe=False)
    without = env.cube_position.copy(), env.joint_positions.copy()

    env.reset(seed=9)
    for _ in range(30):
        env.step(hold, observe=True)
    assert np.allclose(env.cube_position, without[0], atol=1e-9)
    assert np.allclose(env.joint_positions, without[1], atol=1e-9)


# -- the follow camera -------------------------------------------------------


def test_the_follow_camera_holds_still_within_one_frame(env):
    """The panel is rendered through this camera and then drawn on with points
    projected through it. If the second call moved the rig, every overlay would
    be drawn for a camera pose that no longer matched the picture underneath."""
    env.reset(seed=0)
    env.update_chase_camera()
    first = env.chase_camera_pose
    env.update_chase_camera()
    second = env.chase_camera_pose
    assert np.allclose(first[0], second[0])
    assert np.allclose(first[1], second[1])


def test_the_follow_camera_lags_the_gripper_instead_of_copying_it(env):
    env.reset(seed=0)
    env.update_chase_camera()
    start = env.chase_camera_pose[1].copy()

    action = env.commanded_positions.copy()
    action[1] += 0.12
    for _ in range(3):
        env.step(action, observe=False)
    tcp, _ = env.gripper_pose
    travelled = float(np.linalg.norm(tcp - start))
    assert 1e-3 < travelled < env.CHASE_SNAP, "the arm did not move a usable amount"

    env.update_chase_camera()
    look = env.chase_camera_pose[1]
    followed = float(np.linalg.norm(look - start))
    assert 0 < followed < travelled, "the camera either froze or copied the gripper exactly"


def test_the_follow_camera_removes_tremor_rather_than_amplifying_it(env, monkeypatch):
    """A camera pinned rigidly to the tool inherits every tremor of the arm, and
    operators correct against what they see -- which drives the oscillation they
    are trying to remove."""
    from handrobot.sim.env import PickPlaceEnv

    env.reset(seed=0)
    base, _ = env.gripper_pose
    holder = {"position": base.copy()}
    monkeypatch.setattr(
        PickPlaceEnv, "gripper_pose",
        property(lambda self: (holder["position"].copy(), np.eye(3))),
    )
    env.update_chase_camera()

    amplitude = 0.005
    excursions = []
    for i in range(40):
        holder["position"] = base + np.array([0.0, 0.0, amplitude * (1 if i % 2 else -1)])
        env.data.time += 1.0 / 30.0
        env.update_chase_camera()
        if i > 10:
            excursions.append(abs(env.chase_camera_pose[1][2] - base[2]))

    assert max(excursions) < 0.4 * amplitude, "the camera followed the tremor"


def test_the_follow_camera_snaps_when_the_scene_is_reset(env):
    env.reset(seed=0)
    env.update_chase_camera()
    before = env.chase_camera_pose[1].copy()
    env.reset(seed=7)
    action = env.commanded_positions.copy()
    action[1] += 0.6
    for _ in range(10):
        env.step(action, observe=False)
    tcp, _ = env.gripper_pose
    env.update_chase_camera()
    assert np.linalg.norm(env.chase_camera_pose[1] - tcp) < 1e-9, (
        "the camera glided in from the previous episode"
    )
    assert not np.allclose(before, env.chase_camera_pose[1])
