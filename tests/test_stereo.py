"""Stereo triangulation: pure geometry, verified against a synthetic scene."""

import numpy as np
import pytest

from handrobot.hands.stereo import focal_length_px, triangulate_depth


def project(point, camera_x, fx, width):
    """Pinhole x-pixel of a 3D point seen from a camera at (camera_x, 0, 0)."""
    return width / 2 + fx * (point[0] - camera_x) / point[2]


@pytest.mark.parametrize("depth", [0.25, 0.4, 0.6, 0.9, 1.3])
@pytest.mark.parametrize("lateral", [-0.15, 0.0, 0.2])
def test_triangulation_recovers_synthetic_depth_exactly(depth, lateral):
    """Points projected through two ideal parallel pinholes must come back at
    exactly their true depth -- the relation is algebra, not approximation."""
    width, hfov, baseline = 1280, 62.0, 0.12
    fx = focal_length_px(width, hfov)
    point = np.array([lateral, 0.05, depth])
    x_left = project(point, 0.0, fx, width)
    x_right = project(point, baseline, fx, width)
    recovered = triangulate_depth(x_left, x_right, fx, baseline)
    assert recovered == pytest.approx(depth, rel=1e-9)


def test_tiny_disparity_is_refused_rather_than_guessed():
    fx = focal_length_px(1280, 62.0)
    assert triangulate_depth(640.0, 640.4, fx, 0.12) is None


def test_depth_error_scales_with_pixel_noise():
    """One pixel of landmark noise at half a metre must stay under a
    centimetre of depth error with a hand-width baseline -- the whole reason
    stereo beats the monocular estimate's centimetres."""
    width, hfov, baseline, depth = 1280, 62.0, 0.12, 0.5
    fx = focal_length_px(width, hfov)
    point = np.array([0.0, 0.0, depth])
    x_left = project(point, 0.0, fx, width)
    x_right = project(point, baseline, fx, width)
    noisy = triangulate_depth(x_left + 1.0, x_right, fx, baseline)
    assert abs(noisy - depth) < 0.010


def test_refine_preserves_the_bearing(monkeypatch):
    """The stereo fix rescales along the primary camera's ray: direction to the
    hand must be identical before and after, only the length changes."""
    from handrobot.hands import stereo as stereo_module
    from handrobot.hands.types import HandPose, Landmarks

    class FakeCam:
        width, height = 640, 480
        def read(self):
            return np.zeros((480, 640, 3), dtype=np.uint8)
        def close(self):
            pass

    landmarks = Landmarks(
        image=np.full((21, 3), 0.30), world=np.zeros((21, 3)),
        handedness="Right", score=1.0,
    )
    secondary = Landmarks(
        image=np.full((21, 3), 0.42), world=np.zeros((21, 3)),
        handedness="Right", score=1.0,
    )

    class FakeTracker:
        def __init__(self, *a, **k): pass
        def detect(self, frame):
            pose = HandPose(
                palm_position=np.zeros(3), rotation=np.eye(3), pinch_distance=0.05,
                depth=0.5, landmarks=secondary, timestamp=0.0,
            )
            return pose, secondary
        def close(self): pass

    monkeypatch.setattr(stereo_module, "HandTracker", FakeTracker, raising=False)
    import handrobot.hands.tracker as tracker_module
    monkeypatch.setattr(tracker_module, "Webcam", lambda d: FakeCam())
    monkeypatch.setattr(tracker_module, "HandTracker", FakeTracker)

    rig = stereo_module.StereoRig(device=1, baseline_m=0.12)
    original = HandPose(
        palm_position=np.array([0.10, -0.05, 0.60]), rotation=np.eye(3),
        pinch_distance=0.05, depth=0.60, landmarks=landmarks, timestamp=0.0,
    )
    refined = rig.refine(original, primary_width=640)
    rig.close()

    assert refined is not None and refined.depth != original.depth
    direction_before = original.palm_position / np.linalg.norm(original.palm_position)
    direction_after = refined.palm_position / np.linalg.norm(refined.palm_position)
    assert np.allclose(direction_before, direction_after, atol=1e-12)
    assert refined.palm_position[2] == pytest.approx(refined.depth)


def test_lerobot_export_writes_a_valid_dataset(tmp_path):
    """Structural check on the exporter, via a tiny synthetic episode. Skipped
    when the optional lerobot package is not installed."""
    pytest.importorskip("lerobot")
    from handrobot.data.dataset import EpisodeWriter
    from handrobot.data.lerobot_export import export_lerobot

    data = tmp_path / "demo"
    writer = EpisodeWriter(data, cameras=["front_cam"], source="scripted")
    rng = np.random.default_rng(0)
    for _ in range(24):
        writer.add(rng.normal(size=6), rng.normal(size=6),
                   {"front_cam": rng.integers(0, 255, (32, 32, 3)).astype(np.uint8)})
    writer.finish(success=True, metadata={"task": "lift", "instruction": "lift it"})

    out = tmp_path / "lerobot"
    result = export_lerobot(data, out, "test/handrobot-test", log=False)
    assert result["episodes"] == 1

    import json
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["total_frames"] == 24
    assert "observation.images.front_cam" in info["features"]
    parquets = list(out.rglob("*.parquet"))
    videos = list(out.rglob("*.mp4"))
    assert parquets and videos
