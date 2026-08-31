import numpy as np

from handrobot.data.dataset import EpisodeWriter, load_episode
from handrobot.rollout import ScriptedController, evaluate_controller, run_episode
from handrobot.scripted import ScriptedExpert


class HoldStill:
    """A controller that never moves. Useful as a negative control."""

    def reset(self, env):
        self.q = env.joint_positions.copy()

    def act(self, env, observation):
        return self.q


def test_a_motionless_controller_never_succeeds(env):
    result = run_episode(env, HoldStill(), max_steps=40, seed=0)
    assert not result.success


def test_recorded_episode_matches_the_rollout_length(tmp_path, config):
    from handrobot.sim import PickPlaceEnv

    environment = PickPlaceEnv(config=config, seed=0)
    try:
        writer = EpisodeWriter(
            tmp_path, [c.name for c in config.sim.policy_cameras], source="test"
        )
        controller = ScriptedController(ScriptedExpert(config))
        result = run_episode(environment, controller, writer=writer, seed=4321)
        assert len(writer) == result.steps
        path = writer.finish(success=result.success)
    finally:
        environment.close()

    episode = load_episode(path)
    assert len(episode) == result.steps
    assert episode.states.shape[1] == len(config.spec.actuators)
    assert episode.actions.shape[1] == len(config.spec.actuators)
    assert episode.success == result.success


def test_render_frames_are_produced_on_request(env):
    controller = ScriptedController(ScriptedExpert(env.config))
    result = run_episode(env, controller, max_steps=20, seed=0,
                         render_camera="front_cam", render_size=(96, 96))
    assert len(result.frames) >= 20
    assert result.frames[0].shape == (96, 96, 3)


def test_evaluation_is_reproducible(env):
    controller = ScriptedController(ScriptedExpert(env.config))
    a = evaluate_controller(env, controller, episodes=4, seed=77, verbose=False)
    b = evaluate_controller(env, controller, episodes=4, seed=77, verbose=False)
    assert a["success_rate"] == b["success_rate"]
    assert [e["success"] for e in a["per_episode"]] == [e["success"] for e in b["per_episode"]]


def test_evaluation_uses_a_distinct_layout_per_episode(env):
    controller = ScriptedController(ScriptedExpert(env.config))
    layouts = []
    for seed in range(5):
        run_episode(env, controller, max_steps=1, seed=seed)
        layouts.append(tuple(np.round(env.cube_position, 4)))
    assert len(set(layouts)) == len(layouts)
