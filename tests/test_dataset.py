import numpy as np
import pytest

from handrobot.data.dataset import (
    DemoDataset,
    EpisodeWriter,
    list_episodes,
    load_episode,
    read_meta,
)

CAMERAS = ["front_cam", "wrist_cam"]


def make_images(rng, size=64):
    """Smooth gradients, not noise: JPEG round-trips these the way real renders behave."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    images = {}
    for i, camera in enumerate(CAMERAS):
        base = np.stack([x, y, np.full_like(x, 0.2 * (i + 1))], axis=-1)
        base = base + rng.normal(scale=0.01, size=base.shape)
        images[camera] = np.clip(base * 255, 0, 255).astype(np.uint8)
    return images


def write_dataset(root, episodes=3, length=12, seed=0, success=True):
    rng = np.random.default_rng(seed)
    writer = EpisodeWriter(root, CAMERAS, source="test")
    for e in range(episodes):
        for t in range(length):
            writer.add(
                np.full(6, t + e, dtype=np.float32),
                np.full(6, t + e + 0.5, dtype=np.float32),
                make_images(rng),
            )
        writer.finish(success=success, metadata={"episode": e})
    return writer


def test_episode_round_trips_through_disk(tmp_path):
    write_dataset(tmp_path, episodes=1, length=5)
    episode = load_episode(list_episodes(tmp_path)[0])
    assert len(episode) == 5
    assert episode.success
    assert episode.source == "test"
    assert episode.metadata["episode"] == 0
    assert set(episode.images) == set(CAMERAS)
    assert episode.images["front_cam"][0].shape == (64, 64, 3)
    assert np.allclose(episode.states[3], 3.0)
    assert np.allclose(episode.actions[3], 3.5)


def test_images_survive_the_jpeg_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    original = make_images(rng)
    writer = EpisodeWriter(tmp_path, CAMERAS, source="test")
    writer.add(np.zeros(6), np.zeros(6), original)
    path = writer.finish(success=True)
    decoded = load_episode(path).images["front_cam"][0]
    error = np.abs(decoded.astype(float) - original["front_cam"].astype(float)).mean()
    assert error < 4.0, f"mean absolute JPEG error {error:.2f} is too high"


def test_episodes_are_numbered_consecutively(tmp_path):
    write_dataset(tmp_path, episodes=3, length=4)
    names = [p.name for p in list_episodes(tmp_path)]
    assert names == ["episode_000000.npz", "episode_000001.npz", "episode_000002.npz"]


def test_writer_rejects_missing_cameras(tmp_path):
    writer = EpisodeWriter(tmp_path, CAMERAS, source="test")
    with pytest.raises(ValueError):
        writer.add(np.zeros(6), np.zeros(6), {"front_cam": np.zeros((8, 8, 3), np.uint8)})


def test_writer_refuses_to_write_an_empty_episode(tmp_path):
    writer = EpisodeWriter(tmp_path, CAMERAS, source="test")
    with pytest.raises(RuntimeError):
        writer.finish(success=True)


def test_discard_drops_buffered_steps(tmp_path):
    rng = np.random.default_rng(0)
    writer = EpisodeWriter(tmp_path, CAMERAS, source="test")
    writer.add(np.zeros(6), np.zeros(6), make_images(rng))
    assert len(writer) == 1
    writer.discard()
    assert len(writer) == 0
    assert list_episodes(tmp_path) == []


def test_meta_tracks_what_is_on_disk(tmp_path):
    write_dataset(tmp_path, episodes=2, length=7)
    meta = read_meta(tmp_path)
    assert meta["num_episodes"] == 2
    assert meta["num_frames"] == 14
    assert meta["num_successful"] == 2
    assert meta["cameras"] == CAMERAS


def test_chunk_is_padded_at_the_end_of_an_episode(tmp_path):
    write_dataset(tmp_path, episodes=1, length=10)
    dataset = DemoDataset(tmp_path, chunk_size=4)
    middle = dataset[2]
    assert not middle["is_pad"].any()
    assert np.allclose(middle["action"][:, 0], [2.5, 3.5, 4.5, 5.5])

    last = dataset[9]
    assert last["is_pad"].tolist() == [False, True, True, True]
    # Padding repeats the final action, so a policy that follows it simply holds still.
    assert np.allclose(last["action"][1:], last["action"][0])


def test_dataset_length_is_the_total_frame_count(tmp_path):
    write_dataset(tmp_path, episodes=3, length=8)
    assert len(DemoDataset(tmp_path, chunk_size=4)) == 24


def test_failures_are_excluded_by_default(tmp_path):
    write_dataset(tmp_path, episodes=2, length=5, success=True)
    write_dataset(tmp_path, episodes=1, length=5, seed=1, success=False)
    assert len(DemoDataset(tmp_path, chunk_size=2).episodes) == 2
    assert len(DemoDataset(tmp_path, chunk_size=2, successful_only=False).episodes) == 3


def test_dataset_raises_when_nothing_is_usable(tmp_path):
    write_dataset(tmp_path, episodes=1, length=5, success=False)
    with pytest.raises(ValueError):
        DemoDataset(tmp_path, chunk_size=2)


def test_statistics_match_a_direct_computation(tmp_path):
    write_dataset(tmp_path, episodes=3, length=6)
    dataset = DemoDataset(tmp_path, chunk_size=3)
    states = np.concatenate([e.states for e in dataset.episodes])
    assert np.allclose(dataset.stats.state_mean, states.mean(0))
    assert np.allclose(dataset.stats.state_std, np.maximum(states.std(0), 1e-3))


def test_standard_deviation_is_floored_for_constant_dimensions(tmp_path):
    rng = np.random.default_rng(0)
    writer = EpisodeWriter(tmp_path, CAMERAS, source="test")
    for _ in range(6):
        writer.add(np.zeros(6), np.zeros(6), make_images(rng))
    writer.finish(success=True)
    dataset = DemoDataset(tmp_path, chunk_size=2)
    assert np.all(dataset.stats.state_std >= 1e-3)


def test_episode_boundaries_partition_the_index(tmp_path):
    write_dataset(tmp_path, episodes=3, length=5)
    dataset = DemoDataset(tmp_path, chunk_size=2)
    bounds = dataset.episode_boundaries()
    assert bounds == [(0, 5), (5, 10), (10, 15)]
    assert bounds[-1][1] == len(dataset)


def test_auxiliary_cameras_are_recorded_but_excluded_from_training(tmp_path):
    """The operator's webcam must never become a policy input."""
    rng = np.random.default_rng(0)
    cameras = CAMERAS + ["operator_cam"]
    writer = EpisodeWriter(tmp_path, cameras, source="human", policy_cameras=CAMERAS)
    for _ in range(6):
        images = make_images(rng)
        images["operator_cam"] = make_images(rng)["front_cam"]
        writer.add(np.zeros(6), np.zeros(6), images)
    writer.finish(success=True)

    episode = load_episode(list_episodes(tmp_path)[0])
    assert set(episode.images) == set(cameras)
    assert episode.policy_cameras == CAMERAS

    dataset = DemoDataset(tmp_path, chunk_size=2)
    assert dataset.cameras == sorted(CAMERAS)
    assert "image.operator_cam" not in dataset[0]


def test_policy_cameras_must_be_recorded(tmp_path):
    with pytest.raises(ValueError):
        EpisodeWriter(tmp_path, CAMERAS, source="test", policy_cameras=["nope"])


def test_dataset_rejects_episodes_missing_a_requested_camera(tmp_path):
    write_dataset(tmp_path, episodes=1, length=4)
    with pytest.raises(ValueError):
        DemoDataset(tmp_path, chunk_size=2, cameras=["front_cam", "absent_cam"])
