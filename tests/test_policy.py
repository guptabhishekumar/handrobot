import numpy as np
import pytest
import torch

from handrobot.data.dataset import NormalizationStats
from handrobot.policy.act import (
    ACTConfig,
    ACTPolicy,
    sinusoidal_position_embedding_1d,
    sinusoidal_position_embedding_2d,
)
from handrobot.policy.inference import ChunkedActor, load_checkpoint, save_checkpoint
from handrobot.policy.train import learning_rate_at, split_episodes


@pytest.fixture(scope="module")
def tiny_config():
    return ACTConfig(
        chunk_size=6, cameras=("front_cam",), hidden_dim=64, dim_feedforward=128,
        n_heads=4, n_encoder_layers=1, n_decoder_layers=1, n_vae_layers=1,
        latent_dim=8, dropout=0.0, pretrained_backbone=False,
    )


@pytest.fixture(scope="module")
def tiny_policy(tiny_config):
    torch.manual_seed(0)
    return ACTPolicy(tiny_config)


def batch(config, n=2, size=64):
    images = [torch.rand(n, 3, size, size) for _ in config.cameras]
    state = torch.randn(n, config.state_dim)
    actions = torch.randn(n, config.chunk_size, config.action_dim)
    is_pad = torch.zeros(n, config.chunk_size, dtype=torch.bool)
    return images, state, actions, is_pad


def test_position_embeddings_have_the_right_shape():
    assert sinusoidal_position_embedding_2d(64, 4, 5).shape == (20, 64)
    assert sinusoidal_position_embedding_1d(32, 7).shape == (7, 32)


def test_2d_position_embedding_is_distinct_per_cell():
    embedding = sinusoidal_position_embedding_2d(64, 4, 4)
    similarity = embedding @ embedding.T
    diagonal = similarity.diag()
    off_diagonal = similarity - torch.diag(diagonal)
    assert diagonal.min() > off_diagonal.max()


def test_2d_position_embedding_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        sinusoidal_position_embedding_2d(66, 2, 2)


def test_forward_shapes(tiny_policy, tiny_config):
    images, state, actions, is_pad = batch(tiny_config)
    out = tiny_policy(images, state, actions, is_pad)
    assert out["actions"].shape == (2, tiny_config.chunk_size, tiny_config.action_dim)
    assert out["mu"].shape == (2, tiny_config.latent_dim)


def test_inference_uses_the_prior_and_is_deterministic(tiny_policy, tiny_config):
    tiny_policy.eval()
    images, state, _, _ = batch(tiny_config)
    with torch.no_grad():
        a = tiny_policy(images, state)
        b = tiny_policy(images, state)
    assert a["mu"] is None
    assert torch.allclose(a["actions"], b["actions"])


def test_wrong_camera_count_is_rejected(tiny_policy, tiny_config):
    images, state, _, _ = batch(tiny_config)
    with pytest.raises(ValueError):
        tiny_policy.encode_observation(images + images, state,
                                       torch.zeros(2, tiny_config.latent_dim))


def test_padded_actions_do_not_affect_the_loss(tiny_policy, tiny_config):
    images, state, actions, is_pad = batch(tiny_config)
    is_pad[:, 3:] = True
    torch.manual_seed(1)
    a = tiny_policy.compute_loss(images, state, actions, is_pad, kl_weight=0.0)["l1"]

    perturbed = actions.clone()
    perturbed[:, 3:] += 100.0
    torch.manual_seed(1)
    b = tiny_policy.compute_loss(images, state, perturbed, is_pad, kl_weight=0.0)["l1"]
    assert torch.allclose(a, b, atol=1e-5)


def test_kl_is_non_negative(tiny_policy, tiny_config):
    images, state, actions, is_pad = batch(tiny_config)
    losses = tiny_policy.compute_loss(images, state, actions, is_pad, kl_weight=1.0)
    assert float(losses["kl"].detach()) >= -1e-6


def test_gradients_reach_every_trainable_parameter(tiny_config):
    torch.manual_seed(0)
    policy = ACTPolicy(tiny_config)
    images, state, actions, is_pad = batch(tiny_config)
    policy.compute_loss(images, state, actions, is_pad, kl_weight=1.0)["loss"].backward()
    missing = [n for n, p in policy.named_parameters()
               if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert not missing, f"parameters received no gradient: {missing}"


def test_policy_can_overfit_a_single_batch(tiny_config):
    """A model that cannot memorise one batch has a wiring bug, not a data problem."""
    torch.manual_seed(0)
    policy = ACTPolicy(tiny_config)
    images, state, actions, is_pad = batch(tiny_config)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-3)
    first = last = None
    for step in range(120):
        losses = policy.compute_loss(images, state, actions, is_pad, kl_weight=0.0)
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()
        if step == 0:
            first = float(losses["l1"].detach())
        last = float(losses["l1"].detach())
    assert last < 0.3 * first, f"L1 only fell from {first:.3f} to {last:.3f}"


def test_checkpoint_round_trip(tmp_path, tiny_config):
    torch.manual_seed(0)
    policy = ACTPolicy(tiny_config).eval()
    stats = NormalizationStats(
        state_mean=np.zeros(6, np.float32), state_std=np.ones(6, np.float32),
        action_mean=np.zeros(6, np.float32), action_std=np.ones(6, np.float32),
    )
    path = save_checkpoint(tmp_path / "ckpt.pt", policy, stats, extra={"step": 7})
    restored, restored_stats, extra = load_checkpoint(path)
    restored.eval()

    images, state, _, _ = batch(tiny_config)
    with torch.no_grad():
        assert torch.allclose(policy(images, state)["actions"],
                              restored(images, state)["actions"], atol=1e-6)
    assert extra["step"] == 7
    assert np.allclose(restored_stats.state_std, stats.state_std)


def test_actor_denormalises_its_output(tiny_config):
    torch.manual_seed(0)
    policy = ACTPolicy(tiny_config).eval()
    mean = np.full(6, 5.0, np.float32)
    scale = np.full(6, 3.0, np.float32)
    stats = NormalizationStats(np.zeros(6, np.float32), np.ones(6, np.float32), mean, scale)
    actor = ChunkedActor(policy, stats, torch.device("cpu"), image_size=64)

    images = {"front_cam": np.zeros((64, 64, 3), np.uint8)}
    raw = actor._predict_chunk(images, np.zeros(6))
    # Undoing the affine map must land back on the network's normalised output.
    normalised = (raw - mean) / scale
    assert np.all(np.abs(normalised) < 25)
    assert not np.allclose(raw, normalised)


def test_temporal_ensembling_blends_overlapping_chunks(tiny_config):
    torch.manual_seed(0)
    policy = ACTPolicy(tiny_config).eval()
    stats = NormalizationStats(np.zeros(6, np.float32), np.ones(6, np.float32),
                               np.zeros(6, np.float32), np.ones(6, np.float32))
    actor = ChunkedActor(policy, stats, torch.device("cpu"), image_size=64,
                         temporal_ensemble_coeff=0.01)
    images = {"front_cam": np.zeros((64, 64, 3), np.uint8)}
    for step in range(4):
        actor.act(images, np.zeros(6))
    # After four steps four chunks overlap the current timestep, all still in horizon.
    assert len(actor._pending) == 4
    actor.reset()
    assert actor._pending == []


def test_open_loop_actor_replans_on_schedule(tiny_config):
    torch.manual_seed(0)
    policy = ACTPolicy(tiny_config).eval()
    stats = NormalizationStats(np.zeros(6, np.float32), np.ones(6, np.float32),
                               np.zeros(6, np.float32), np.ones(6, np.float32))
    actor = ChunkedActor(policy, stats, torch.device("cpu"), image_size=64,
                         temporal_ensemble_coeff=None, query_every=3)
    images = {"front_cam": np.zeros((64, 64, 3), np.uint8)}
    actions = [actor.act(images, np.zeros(6)) for _ in range(6)]
    assert not np.allclose(actions[0], actions[1])
    assert len(actor._pending) == 1


def test_learning_rate_warms_up_then_decays():
    from handrobot.config import TrainConfig

    config = TrainConfig(steps=1000, warmup_steps=100)
    assert learning_rate_at(0, config) < learning_rate_at(50, config)
    assert learning_rate_at(99, config) == pytest.approx(1.0)
    assert learning_rate_at(100, config) == pytest.approx(1.0, abs=1e-6)
    assert learning_rate_at(999, config) < 0.2
    assert learning_rate_at(2000, config) == pytest.approx(0.1, abs=1e-6)


def test_validation_split_never_cuts_an_episode(tmp_path):
    from tests.test_dataset import write_dataset
    from handrobot.data.dataset import DemoDataset

    write_dataset(tmp_path, episodes=5, length=8)
    dataset = DemoDataset(tmp_path, chunk_size=3)
    train_idx, val_idx = split_episodes(dataset, val_fraction=0.2, seed=0)

    assert set(train_idx).isdisjoint(val_idx)
    assert len(train_idx) + len(val_idx) == len(dataset)
    for start, end in dataset.episode_boundaries():
        frames = set(range(start, end))
        assert frames <= set(train_idx.tolist()) or frames <= set(val_idx.tolist())


def test_augmentation_changes_images_without_leaving_the_valid_range():
    from handrobot.policy.train import augment

    torch.manual_seed(0)
    images = {"front_cam": torch.rand(4, 3, 32, 32)}
    out = augment(images)["front_cam"]
    assert out.shape == images["front_cam"].shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert not torch.allclose(out, images["front_cam"])


def test_augmentation_is_reproducible_from_the_global_seed():
    from handrobot.policy.train import augment

    images = {"front_cam": torch.rand(2, 3, 16, 16)}
    torch.manual_seed(7)
    first = augment(images)["front_cam"].clone()
    torch.manual_seed(7)
    assert torch.allclose(first, augment(images)["front_cam"])


def test_single_episode_dataset_still_writes_a_best_checkpoint(tmp_path):
    """With one episode there is nothing to validate on; best.pt must still exist."""
    from tests.test_dataset import write_dataset
    from handrobot.config import TrainConfig
    from handrobot.policy.train import train

    data = tmp_path / "data"
    write_dataset(data, episodes=1, length=12)
    config = TrainConfig()
    config.steps, config.save_every, config.log_every = 3, 3, 3
    config.batch_size, config.chunk_size = 2, 4
    config.hidden_dim, config.dim_feedforward = 64, 128
    config.n_heads, config.n_encoder_layers, config.n_decoder_layers = 4, 1, 1
    config.latent_dim, config.num_workers = 8, 0

    train(data, tmp_path / "out", config, device_name="cpu", log=False)
    assert (tmp_path / "out" / "best.pt").exists()
    assert (tmp_path / "out" / "last.pt").exists()


@pytest.mark.parametrize("augment", [True, False])
def test_training_runs_with_augmentation_on_and_off(tmp_path, augment):
    """Regression: without augmentation the batch stayed a permuted view, and the
    backward pass raised "view size is not compatible with input tensor's size"."""
    from tests.test_dataset import write_dataset
    from handrobot.config import TrainConfig
    from handrobot.policy.train import train

    data = tmp_path / "data"
    write_dataset(data, episodes=2, length=10)
    config = TrainConfig()
    config.steps, config.save_every, config.log_every = 2, 2, 99
    config.batch_size, config.chunk_size = 2, 4
    config.hidden_dim, config.dim_feedforward = 64, 128
    config.n_heads, config.n_encoder_layers, config.n_decoder_layers = 4, 1, 1
    config.latent_dim, config.augment = 8, augment

    summary = train(data, tmp_path / f"out{augment}", config, device_name="cpu", log=False)
    assert summary["steps"] == 2
