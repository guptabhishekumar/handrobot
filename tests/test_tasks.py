"""The multi-task layer: registry, per-task success, experts, conditioning."""

import numpy as np
import pytest
import torch

from handrobot.config import Config
from handrobot.tasks import TASK_NAMES, get_task, task_id


def test_every_alias_resolves_to_its_task():
    for name in TASK_NAMES:
        task = get_task(name)
        assert get_task(task.instruction).name == name
        for alias in task.aliases:
            assert get_task(alias).name == name


def test_unknown_phrasings_are_refused_with_the_menu():
    with pytest.raises(KeyError, match="bin"):
        get_task("do a backflip")


def test_task_ids_are_stable_and_dense():
    assert [task_id(n) for n in TASK_NAMES] == list(range(len(TASK_NAMES)))


@pytest.fixture(scope="module")
def panda_config():
    return Config(robot="panda")


@pytest.mark.parametrize("task", TASK_NAMES)
def test_the_expert_solves_each_task(panda_config, task):
    """One seed per task here; the wide sweep lives in the collection runs."""
    from handrobot.rollout import ScriptedController, run_episode
    from handrobot.scripted.expert import ScriptedExpert
    from handrobot.sim import PickPlaceEnv

    env = PickPlaceEnv(config=panda_config, render_cameras=(), seed=3, task=task)
    result = run_episode(env, ScriptedController(ScriptedExpert(panda_config)))
    env.close()
    assert result.success, f"the scripted expert failed task {task!r}"


def test_success_predicates_disagree_between_tasks(panda_config):
    """Solving 'lift' must not count as solving 'push': the objectives are
    genuinely different, or multi-task conditioning would be learning nothing."""
    from handrobot.rollout import ScriptedController, run_episode
    from handrobot.scripted.expert import ScriptedExpert
    from handrobot.sim import PickPlaceEnv
    from handrobot.tasks import TASKS

    env = PickPlaceEnv(config=panda_config, render_cameras=(), seed=5, task="lift")
    result = run_episode(env, ScriptedController(ScriptedExpert(panda_config)))
    assert result.success
    assert not TASKS["push"].success(env), "a lifted puck must not satisfy push"
    assert not TASKS["bin"].success(env), "a lifted puck must not satisfy bin"
    env.close()


def test_push_zone_is_always_visible_and_reachable(panda_config):
    from handrobot.sim import PickPlaceEnv

    for seed in range(5):
        env = PickPlaceEnv(config=panda_config, render_cameras=(), seed=seed, task="push")
        env.reset()
        assert env.zone_position is not None
        clipped = panda_config.workspace.clip(
            np.array([*env.zone_position[:2], 0.05])
        )
        assert np.allclose(clipped[:2], env.zone_position[:2], atol=1e-9), (
            "the sampled zone must lie inside the reachable workspace"
        )
        env.close()


def test_task_conditioning_changes_the_policy_output():
    """The same observation with a different task id must produce a different
    action chunk; otherwise the conditioning is dead weight."""
    from handrobot.policy.act import ACTConfig, ACTPolicy

    torch.manual_seed(0)
    policy = ACTPolicy(ACTConfig(n_tasks=len(TASK_NAMES), pretrained_backbone=False)).eval()
    images = [torch.rand(1, 3, 128, 128), torch.rand(1, 3, 128, 128)]
    state = torch.rand(1, 6)
    with torch.no_grad():
        a = policy(images, state, task=torch.tensor([0]))["actions"]
        b = policy(images, state, task=torch.tensor([2]))["actions"]
    assert not torch.allclose(a, b), "task id had no effect on the output"


def test_single_task_config_has_no_task_parameters():
    """n_tasks=1 must build the exact pre-multi-task architecture, so every
    existing checkpoint keeps loading."""
    from handrobot.policy.act import ACTConfig, ACTPolicy

    policy = ACTPolicy(ACTConfig(n_tasks=1, pretrained_backbone=False))
    assert not any("task_embed" in k for k in policy.state_dict())
    old_payload = {k: v for k, v in ACTConfig().to_dict().items() if k != "n_tasks"}
    assert ACTConfig.from_dict(old_payload).n_tasks == 1


def test_dataset_carries_task_ids(tmp_path):
    from handrobot.data.dataset import DemoDataset, EpisodeWriter

    for task in ("bin", "lift"):
        writer = EpisodeWriter(tmp_path, cameras=["front_cam"], source="scripted")
        for _ in range(40):
            writer.add(np.zeros(6), np.zeros(6),
                       {"front_cam": np.zeros((8, 8, 3), dtype=np.uint8)})
        writer.finish(success=True, metadata={"task": task, "instruction": "x"})

    dataset = DemoDataset(tmp_path, chunk_size=4)
    assert dataset.n_tasks == len(TASK_NAMES)
    assert sorted(set(dataset.episode_tasks)) == [task_id("bin"), task_id("lift")]
    assert dataset[0]["task"] == dataset.episode_tasks[0]


def test_diffusion_policy_can_overfit_a_single_batch():
    """Same bar the ACT model must clear: a denoiser that cannot memorise one
    batch has a wiring bug. Trained and sampled on CPU at a small size."""
    from handrobot.policy.diffusion import DiffusionChunkPolicy, DiffusionConfig

    torch.manual_seed(0)
    config = DiffusionConfig(
        chunk_size=8, cameras=("front_cam",), hidden_dim=64, dim_feedforward=128,
        n_heads=4, n_encoder_layers=1, n_denoiser_layers=2, dropout=0.0,
        pretrained_backbone=False, train_timesteps=50, sample_steps=25,
    )
    policy = DiffusionChunkPolicy(config)
    images = [torch.rand(4, 3, 64, 64)]
    state = torch.rand(4, 6)
    actions = torch.rand(4, 8, 6) * 2 - 1
    is_pad = torch.zeros(4, 8, dtype=torch.bool)

    optimiser = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    for _ in range(2000):
        loss = policy.compute_loss(images, state, actions, is_pad)["loss"]
        optimiser.zero_grad(); loss.backward(); optimiser.step()
    # The per-step loss is noisy (a fresh noise draw and timestep every step),
    # so judge the average of several evaluations, not one draw.
    with torch.no_grad():
        final = float(np.mean([
            float(policy.compute_loss(images, state, actions, is_pad)["loss"])
            for _ in range(20)
        ]))
    assert final < 0.15, f"noise loss stuck at {final:.3f}"

    policy.eval()
    sampled = policy(images, state)["actions"]
    error = (sampled - actions).abs().mean()
    assert float(error) < 0.35, (
        f"sampled chunk is {float(error):.3f} L1 from the memorised one"
    )


def test_diffusion_checkpoint_round_trips(tmp_path):
    from handrobot.data.dataset import NormalizationStats
    from handrobot.policy.diffusion import DiffusionChunkPolicy, DiffusionConfig
    from handrobot.policy.inference import load_checkpoint, save_checkpoint

    policy = DiffusionChunkPolicy(DiffusionConfig(
        chunk_size=4, cameras=("front_cam",), hidden_dim=64, dim_feedforward=128,
        n_heads=4, n_encoder_layers=1, n_denoiser_layers=1, pretrained_backbone=False,
    ))
    stats = NormalizationStats(*(np.ones(6, dtype=np.float32),) * 4)
    save_checkpoint(tmp_path / "d.pt", policy, stats)
    loaded, _, _ = load_checkpoint(tmp_path / "d.pt")
    assert isinstance(loaded, DiffusionChunkPolicy)
    assert loaded.config.chunk_size == 4
