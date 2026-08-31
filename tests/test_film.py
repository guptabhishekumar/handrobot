"""Tests for the film composer.

The load-bearing claim here is that a recorded episode can be re-rendered at any
resolution because the simulator is deterministic. If that were only
approximately true, the middle panel would drift out of sync with the operator
panel beside it, and the film would be quietly wrong.
"""

import numpy as np
import pytest

from handrobot.config import Config
from handrobot.data.dataset import EpisodeWriter, list_episodes, load_episode
from handrobot.rollout import ScriptedController, run_episode
from handrobot.scripted import ScriptedExpert
from handrobot.sim.env import PickPlaceEnv
from handrobot.viz.film import (
    CAPTION_HEIGHT,
    PANEL_HEIGHT,
    PANEL_WIDTH,
    Panel,
    build_film,
    compose,
    replay_episode,
)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """One scripted episode written to a temporary dataset."""
    root = tmp_path_factory.mktemp("film_data")
    config = Config()
    env = PickPlaceEnv(config=config, seed=0)
    try:
        writer = EpisodeWriter(
            root, [c.name for c in config.sim.policy_cameras], source="scripted"
        )
        controller = ScriptedController(ScriptedExpert(config))
        result = run_episode(env, controller, writer=writer, seed=777)
        writer.finish(success=result.success, metadata={"seed": 777})
    finally:
        env.close()
    return root


def test_replay_reproduces_the_recorded_states_exactly(recorded):
    """Replay is exact, not merely similar: same seed, same actions, same physics."""
    episode = load_episode(list_episodes(recorded)[0])
    env = PickPlaceEnv(config=Config(), render_cameras=())
    try:
        env.reset(seed=int(episode.metadata["seed"]))
        for t, action in enumerate(episode.actions):
            assert np.allclose(env.joint_positions, episode.states[t], atol=1e-6), (
                f"replay diverged from the recording at step {t}"
            )
            env.step(action)
    finally:
        env.close()


def test_replay_renders_one_frame_per_step(recorded):
    episode = load_episode(list_episodes(recorded)[0])
    frames = replay_episode(episode, size=(120, 160))
    assert len(frames) == len(episode) + 1
    assert frames[0].shape == (120, 160, 3)
    assert np.stack(frames).std() > 1.0, "the replay rendered a flat image"


def test_replay_refuses_an_episode_with_no_seed(tmp_path):
    from tests.test_dataset import write_dataset

    write_dataset(tmp_path, episodes=1, length=3)
    episode = load_episode(list_episodes(tmp_path)[0])
    episode.metadata.pop("seed", None)
    with pytest.raises(ValueError, match="seed"):
        replay_episode(episode)


def test_compose_lays_panels_side_by_side():
    a = [np.full((100, 100, 3), 40, np.uint8) for _ in range(5)]
    b = [np.full((80, 160, 3), 200, np.uint8) for _ in range(3)]
    frames = compose([Panel(a, "One", "first"), Panel(b, "Two", "second")])
    assert len(frames) == 5, "the shorter panel should hold, not truncate the film"
    assert frames[0].shape == (PANEL_HEIGHT, 2 * PANEL_WIDTH, 3)


def test_compose_holds_a_short_panel_on_its_last_frame():
    long_panel = [np.full((60, 60, 3), i, np.uint8) for i in range(4)]
    short_panel = [np.full((60, 60, 3), 9, np.uint8)]
    frames = compose([Panel(long_panel, "a", ""), Panel(short_panel, "b", "")])
    right = [f[:, PANEL_WIDTH:] for f in frames]
    assert all(np.array_equal(right[0], r) for r in right[1:])


def test_compose_preserves_aspect_ratio_by_padding():
    wide = [np.full((40, 400, 3), 255, np.uint8)]
    frame = compose([Panel(wide, "a", "")])[0]
    body = frame[CAPTION_HEIGHT:]
    # A very wide image in a tall panel must be letterboxed, not stretched.
    assert body[0, PANEL_WIDTH // 2].max() < 60
    assert body[body.shape[0] // 2, PANEL_WIDTH // 2].max() > 200


def test_compose_rejects_empty_input():
    with pytest.raises(ValueError):
        compose([])


def test_build_film_without_a_checkpoint_makes_the_replay_panel(recorded, tmp_path):
    result = build_film(recorded, 0, None, tmp_path / "film.mp4")
    assert result["panels"] == ["The robot copying"]
    assert result["policy_success"] is None
    assert (tmp_path / "film.mp4").exists()
    assert (tmp_path / "film.mp4").stat().st_size > 10_000


def test_build_film_rejects_an_out_of_range_episode(recorded, tmp_path):
    with pytest.raises(IndexError):
        build_film(recorded, 99, None, tmp_path / "film.mp4")


def test_build_film_rejects_an_empty_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_film(tmp_path, 0, None, tmp_path / "film.mp4")


@pytest.fixture(scope="module")
def recorded_with_operator(tmp_path_factory):
    """A short episode carrying an operator-camera stream, as teleop writes."""
    root = tmp_path_factory.mktemp("film_operator")
    config = Config()
    policy_cameras = [c.name for c in config.sim.policy_cameras]
    env = PickPlaceEnv(config=config, seed=0)
    try:
        writer = EpisodeWriter(
            root, policy_cameras + ["operator_cam"], source="human",
            policy_cameras=policy_cameras,
        )
        expert = ScriptedExpert(config)
        observation = env.reset(seed=555)
        expert.reset(env)
        for t in range(40):
            action = expert.act(env)
            images = dict(observation.images)
            # Stand-in for the webcam frame; a real one comes from run_teleop.
            images["operator_cam"] = np.full((360, 480, 3), (t * 5) % 255, np.uint8)
            writer.add(observation.joint_positions, action, images)
            observation = env.step(action).observation
        writer.finish(success=False, metadata={"seed": 555, "teleop": True})
    finally:
        env.close()
    return root


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """An untrained ACT checkpoint, enough to exercise the policy panel."""
    import torch

    from handrobot.data.dataset import NormalizationStats
    from handrobot.policy.act import ACTConfig, ACTPolicy
    from handrobot.policy.inference import save_checkpoint

    config = Config()
    cameras = tuple(c.name for c in config.sim.policy_cameras)
    torch.manual_seed(0)
    policy = ACTPolicy(
        ACTConfig(state_dim=len(config.spec.actuators),
                  action_dim=len(config.spec.actuators),
                  chunk_size=8, cameras=cameras, hidden_dim=64, dim_feedforward=128,
                  n_heads=4, n_encoder_layers=1, n_decoder_layers=1, n_vae_layers=1,
                  latent_dim=8, dropout=0.0, pretrained_backbone=False)
    )
    n = len(config.spec.actuators)
    stats = NormalizationStats(
        state_mean=np.zeros(n, np.float32), state_std=np.ones(n, np.float32),
        action_mean=np.zeros(n, np.float32), action_std=np.ones(n, np.float32) * 0.1,
    )
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    save_checkpoint(path, policy, stats, extra={"step": 0})
    return path


def test_three_panel_film_composes_operator_replay_and_policy(
    recorded_with_operator, tiny_checkpoint, tmp_path
):
    result = build_film(
        recorded_with_operator, 0, tiny_checkpoint, tmp_path / "three.mp4",
        device_name="cpu",
    )
    assert result["panels"] == ["The operator", "The robot copying", "The robot alone"]
    assert result["policy_success"] in (True, False)
    assert (tmp_path / "three.mp4").exists()

    import imageio.v2 as imageio

    reader = imageio.get_reader(tmp_path / "three.mp4")
    frame = reader.get_data(5)
    reader.close()
    assert frame.shape == (PANEL_HEIGHT, 3 * PANEL_WIDTH, 3)
    # Each panel must carry different content, not a repeated image.
    columns = [frame[:, i * PANEL_WIDTH : (i + 1) * PANEL_WIDTH] for i in range(3)]
    assert not np.array_equal(columns[0], columns[1])
    assert not np.array_equal(columns[1], columns[2])


def test_operator_panel_is_absent_from_a_scripted_recording(recorded, tmp_path):
    result = build_film(recorded, 0, None, tmp_path / "scripted.mp4")
    assert "The operator" not in result["panels"]
