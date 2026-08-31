"""Neural retargeting: the differentiable model, the network, the live chain.

The order of trust matters. The FK is checked against MuJoCo to a micron first,
because a network trained through wrong geometry learns wrong hands perfectly.
Then the trained network is held to the behaviours an operator would notice in
the first five seconds: an open hand opens it, a fist closes it, and the real
photograph of a real hand produces a plausible pose inside the joint limits.
"""

import numpy as np
import pytest
import torch

from handrobot.dexhand.fk import (
    KEYPOINT_BODIES,
    KEYPOINT_NAMES,
    LEAP_XML,
    TIP_OFFSETS,
    LeapFK,
)
from handrobot.dexhand.retarget_net import (
    DIGITS,
    build_targets,
    digit_lengths,
    load_retargeter,
    normalise,
)
from handrobot.dexhand.synth import landmarks_to_keypoints, sample_hands


@pytest.fixture(scope="module")
def fk():
    return LeapFK()


@pytest.fixture(scope="module")
def retargeter():
    return load_retargeter(train_if_missing=False)


def test_differentiable_fk_matches_mujoco_exactly(fk):
    """A micron of disagreement here poisons everything trained on top."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(LEAP_XML))
    data = mujoco.MjData(model)
    palm = model.body("palm").id
    rng = np.random.default_rng(0)

    worst = 0.0
    for _ in range(10):
        q = rng.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1])
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        palm_rotation = data.xmat[palm].reshape(3, 3)
        palm_position = data.xpos[palm]
        ours = fk(torch.tensor(q, dtype=torch.float64).unsqueeze(0))[0].numpy()
        for i, (body, name) in enumerate(zip(KEYPOINT_BODIES, KEYPOINT_NAMES)):
            world = data.xpos[model.body(body).id].copy()
            if name.endswith("_tip"):
                world = world + data.xmat[model.body(body).id].reshape(3, 3) @ np.array(
                    TIP_OFFSETS[body]
                )
            reference = palm_rotation.T @ (world - palm_position)
            worst = max(worst, float(np.abs(ours[i] - reference).max()))
    # float32 constants in the chain leave a few nanometres of round-off.
    assert worst < 1e-7, f"FK disagrees with MuJoCo by {worst * 1e6:.3f} microns"


def test_fk_is_differentiable(fk):
    q = torch.zeros(2, 16, dtype=torch.float64, requires_grad=True)
    fk(q).sum().backward()
    assert torch.isfinite(q.grad).all()
    assert not torch.all(q.grad == 0)


def test_synthetic_hands_are_hand_shaped():
    hands = sample_hands(200, np.random.default_rng(0)).numpy()
    assert hands.shape == (200, 16, 3)
    assert np.isfinite(hands).all()
    lengths = digit_lengths(torch.tensor(hands)).numpy()
    assert (lengths > 0.04).all() and (lengths < 0.16).all()
    # Index knuckle on the thumb side, ring on the far side, every time.
    assert (hands[:, 0, 1] > hands[:, 8, 1]).all()


def test_targets_are_anchored_to_the_robot_and_scaled_per_digit(fk):
    straight = fk(torch.zeros(1, 16))
    anchors = torch.stack([straight[0, d[0]] for d in DIGITS])
    robot_lengths = digit_lengths(straight)[0]
    human = sample_hands(32, np.random.default_rng(1))
    targets = build_targets(human, anchors, robot_lengths)

    for d, digit in enumerate(DIGITS):
        assert torch.allclose(targets[:, digit[0]], anchors[d].expand(32, 3), atol=1e-6)
    assert torch.allclose(digit_lengths(targets), robot_lengths.expand(32, 4), atol=1e-4)


def test_normalise_is_size_invariant():
    hands = sample_hands(8, np.random.default_rng(2))
    assert torch.allclose(normalise(hands * 1.7), normalise(hands), atol=1e-5)


def test_the_network_is_near_the_kinematic_ceiling(fk, retargeter):
    """The honest quality bar: within a few millimetres of the best any
    controller could do on this hand, measured through the same FK."""
    human = sample_hands(64, np.random.default_rng(7))
    straight = fk(torch.zeros(1, 16))
    anchors = torch.stack([straight[0, d[0]] for d in DIGITS])
    robot_lengths = digit_lengths(straight)[0]
    targets = build_targets(human, anchors, robot_lengths)

    q = torch.tensor(np.stack([retargeter(h.numpy()) for h in human]), dtype=torch.float32)
    error = torch.sqrt(((fk(q) - targets) ** 2).sum(-1)).mean()
    assert float(error) < 0.020, f"mean keypoint error {float(error) * 1000:.1f} mm"


def test_an_open_hand_opens_and_a_fist_closes(retargeter):
    hands = sample_hands(400, np.random.default_rng(3))
    curls = [float(-(h[3, 2] + h[7, 2] + h[11, 2])) for h in hands]
    open_hand = hands[int(np.argmin(curls))].numpy()
    fist = hands[int(np.argmax(curls))].numpy()

    flexes = [0, 2, 3, 4, 6, 7, 8, 10, 11]  # finger flexion joints
    q_open = retargeter(open_hand)[flexes]
    q_fist = retargeter(fist)[flexes]
    assert q_fist.mean() > q_open.mean() + 0.4, (
        f"fist {q_fist.mean():.2f} vs open {q_open.mean():.2f}"
    )


def test_output_respects_joint_limits(fk, retargeter):
    for hand in sample_hands(32, np.random.default_rng(4)):
        q = torch.tensor(retargeter(hand.numpy()))
        assert torch.all(q >= fk.joint_low - 1e-5)
        assert torch.all(q <= fk.joint_high + 1e-5)


def test_the_real_photograph_produces_a_plausible_pose(retargeter, hand_landmarks):
    pose, _ = hand_landmarks
    keypoints = landmarks_to_keypoints(pose.landmarks.world)
    q = retargeter(keypoints)
    assert np.isfinite(q).all()
    # The photographed hand is open with spread fingers: no finger near a fist.
    assert q[[0, 4, 8]].max() < 1.2


def test_the_anatomical_frame_round_trips_synthetic_hands():
    """Training data and live data must pass through one frame definition; this
    pins that the definition reproduces the generator's own convention."""
    from handrobot.dexhand.synth import HUMAN_KEYPOINTS

    hands = sample_hands(30, np.random.default_rng(0)).numpy()
    for hand in hands:
        world = np.zeros((21, 3))
        world[list(HUMAN_KEYPOINTS)] = hand
        rebuilt = landmarks_to_keypoints(world)
        assert np.abs(rebuilt - hand).max() < 1e-9


@pytest.mark.parametrize("axis", [0, 1])
def test_a_mirrored_camera_changes_nothing_at_all(axis):
    """A mirrored feed labels the operator's right hand "Left"; with that label
    honoured, every keypoint must come out identical to the unmirrored case --
    curls, spreads, thumb, everything. This is the property whose absence made
    the first live session read curls as their opposite."""
    from handrobot.dexhand.synth import HUMAN_KEYPOINTS

    for seed in range(10):
        hand = sample_hands(1, np.random.default_rng(seed))[0].numpy()
        world = np.zeros((21, 3))
        world[list(HUMAN_KEYPOINTS)] = hand
        mirrored = world.copy()
        mirrored[:, axis] *= -1.0

        a = landmarks_to_keypoints(world, "Right")
        b = landmarks_to_keypoints(mirrored, "Left")
        assert np.abs(a - b).max() < 1e-9


def test_personal_training_learns_a_specific_hand(tmp_path, monkeypatch):
    """The full personalisation path: record file in, fine-tuned checkpoint out,
    and the loader preferring it. Redirected into tmp_path -- a test must never
    plant a 200-step toy checkpoint where the operator's real one lives."""
    from handrobot.dexhand import retarget_net as rn

    fake_personal = tmp_path / "personal.pt"
    monkeypatch.setattr(rn, "PERSONAL_CHECKPOINT", fake_personal)

    base = sample_hands(300, np.random.default_rng(8)).numpy()
    recording = tmp_path / "hand_poses.npz"
    np.savez_compressed(recording, keypoints=base)

    result = rn.train_personal(recording_path=recording, steps=200, log=False,
                               require_improvement=False)
    assert result["recorded_poses"] == 300
    assert fake_personal.exists()

    personal = rn.load_retargeter()          # must prefer the personal net
    q = personal(base[0])
    assert np.isfinite(q).all()


def test_too_short_a_recording_is_refused(tmp_path):
    from handrobot.dexhand import retarget_net as rn

    recording = tmp_path / "short.npz"
    np.savez_compressed(recording, keypoints=np.zeros((50, 16, 3)))
    with pytest.raises(ValueError, match="record"):
        rn.train_personal(recording_path=recording, steps=10, log=False)


def test_a_frozen_hand_recording_is_refused(tmp_path):
    """One pose repeated 300 times must not produce a personal checkpoint."""
    from handrobot.dexhand import retarget_net as rn

    single = sample_hands(1, np.random.default_rng(3)).numpy()
    recording = tmp_path / "frozen.npz"
    np.savez_compressed(recording, keypoints=np.repeat(single, 300, axis=0))
    with pytest.raises(ValueError, match="varies"):
        rn.train_personal(recording_path=recording, steps=10, log=False)


def test_training_that_does_not_beat_the_base_installs_nothing(tmp_path, monkeypatch):
    """Zero steps of training equals the base net; the gate must refuse the
    tie and leave no personal checkpoint for the loader to silently prefer."""
    from handrobot.dexhand import retarget_net as rn

    fake_personal = tmp_path / "personal.pt"
    monkeypatch.setattr(rn, "PERSONAL_CHECKPOINT", fake_personal)
    recording = tmp_path / "hand_poses.npz"
    np.savez_compressed(
        recording, keypoints=sample_hands(300, np.random.default_rng(5)).numpy()
    )
    with pytest.raises(ValueError, match="did not beat"):
        rn.train_personal(recording_path=recording, steps=0, log=False)
    assert not fake_personal.exists()


class _FakeWebcam:
    """Feeds the real photograph as live frames."""

    def __init__(self, frame, frames=120):
        self._frame = frame
        self._remaining = frames
        self.height, self.width = frame.shape[:2]

    def read(self):
        if self._remaining <= 0:
            return None
        self._remaining -= 1
        return self._frame.copy()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def close(self):
        pass


@pytest.fixture
def stub_camera(monkeypatch):
    import cv2

    import handrobot.hands.tracker as tracker_module
    from handrobot.paths import PROJECT_ROOT

    path = PROJECT_ROOT / "tests" / "fixtures" / "hand_sample.jpg"
    if not path.exists():
        pytest.skip("hand fixture image is not present")
    frame = cv2.cvtColor(cv2.resize(cv2.imread(str(path)), (480, 640)), cv2.COLOR_BGR2RGB)

    import handrobot.dexhand.live as live_module
    import handrobot.dexhand.record as record_module

    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "waitKey", lambda t: 255)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    # Webcam is imported at module level wherever it is used, so the stub must
    # land on every importer -- patching only the tracker module leaves the
    # real camera in play.
    for module in (tracker_module, record_module, live_module):
        monkeypatch.setattr(
            module, "Webcam", lambda device=0, **kw: _FakeWebcam(frame)
        )
    return frame


def test_recording_runs_end_to_end_without_a_camera(stub_camera, tmp_path):
    """The exact `dexhand --record` capture path, headless."""
    from handrobot.dexhand.record import load_recording, record_hand

    result = record_hand(seconds_per_prompt=0.2, device=0, hand="right",
                         out=tmp_path / "rec.npz")
    assert result["poses"] > 30
    data = load_recording(tmp_path / "rec.npz")
    assert data.shape[1:] == (16, 3)
    assert np.isfinite(data).all()


def test_recording_then_personal_training_then_loading(stub_camera, tmp_path, monkeypatch):
    """The full --record promise: capture, train, verify, and the live loader
    picking the personal network up."""
    from handrobot.dexhand import retarget_net as rn
    from handrobot.dexhand.record import record_hand

    monkeypatch.setattr(rn, "PERSONAL_CHECKPOINT", tmp_path / "personal.pt")
    import time as time_module

    import handrobot.dexhand.record as record_module

    fake = {"t": 0.0}

    def ticking():
        fake["t"] += 0.05
        return fake["t"]

    monkeypatch.setattr(time_module, "perf_counter", ticking)
    monkeypatch.setattr(
        record_module, "Webcam", lambda device=0, **kw: _FakeWebcam(stub_camera, frames=260)
    )
    recording = tmp_path / "rec.npz"
    result = record_hand(seconds_per_prompt=2.4, device=0, hand="right", out=recording)
    assert result["poses"] >= 200, "the paused-clock capture should gather plenty"

    trained = rn.train_personal(recording_path=recording, steps=150, log=False,
                                min_curl_span=0.0, require_improvement=False)
    assert (tmp_path / "personal.pt").exists()
    assert np.isfinite(trained["holdout_error"])
    # One static photo repeated: the net should fit this hand's pose closely.
    assert trained["holdout_error"] < 0.03

    personal = rn.load_retargeter()
    q = personal(rn.load_recording(recording)[0] if hasattr(rn, "load_recording")
                 else __import__("handrobot.dexhand.record", fromlist=["load_recording"]).load_recording(recording)[0])
    assert np.isfinite(q).all()


def test_the_paused_clock_does_not_advance_when_untracked(tmp_path, monkeypatch):
    """Seconds with no hand in frame must not eat the capture."""
    import cv2

    import handrobot.hands.tracker as tracker_module
    from handrobot.dexhand.record import record_hand

    blank = np.zeros((640, 480, 3), dtype=np.uint8)
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    import handrobot.dexhand.record as record_module

    for module in (tracker_module, record_module):
        monkeypatch.setattr(
            module, "Webcam", lambda device=0, **kw: _FakeWebcam(blank, frames=40)
        )
    monkeypatch.setattr(cv2, "waitKey", lambda t: 255)

    result = record_hand(seconds_per_prompt=0.05, device=0, out=tmp_path / "rec.npz")
    assert result["poses"] == 0
    assert result["tracking"] == 0.0
