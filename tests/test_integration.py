"""End-to-end tests of the interactive loops, with the camera and windows stubbed.

The webcam and the OpenCV windows are the only parts of ``run_teleop`` and
``run_handcheck`` that cannot exist in a test. Everything else -- MediaPipe
running on a real photograph, the retargeter, inverse kinematics, the simulator,
the dataset writer, the keyboard handling -- is real here. Without this, the two
commands a user actually types would never have been executed before shipping.
"""

from pathlib import Path

import numpy as np
import pytest

from handrobot.data.dataset import list_episodes, load_episode
from handrobot.teleop import OPERATOR_CAMERA


@pytest.fixture(scope="module")
def hand_frame():
    """A real photograph of a hand, as an RGB frame a webcam would produce."""
    import cv2

    from handrobot.paths import PROJECT_ROOT

    path = PROJECT_ROOT / "tests" / "fixtures" / "hand_sample.jpg"
    if not path.exists():
        pytest.skip("hand fixture image is not present")
    bgr = cv2.imread(str(path))
    return cv2.cvtColor(cv2.resize(bgr, (480, 640)), cv2.COLOR_BGR2RGB)


class FakeWebcam:
    """Stands in for :class:`handrobot.hands.tracker.Webcam`."""

    def __init__(self, frame: np.ndarray, frames: int) -> None:
        self._frame = frame
        self._remaining = frames
        self.height, self.width = frame.shape[:2]
        self.closed = False

    def read(self):
        if self._remaining <= 0:
            return None
        self._remaining -= 1
        return self._frame.copy()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.closed = True


@pytest.fixture
def stub_display(monkeypatch):
    """Silence the OpenCV windows and script the keys the operator presses."""
    import cv2

    shown = []
    keys: list[int] = []

    monkeypatch.setattr(cv2, "imshow", lambda name, image: shown.append(image))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: keys.pop(0) if keys else 255)
    return shown, keys


def test_teleop_loop_runs_and_records(tmp_path, hand_frame, stub_display, monkeypatch):
    import handrobot.teleop as teleop

    shown, keys = stub_display
    camera = FakeWebcam(hand_frame, frames=60)
    monkeypatch.setattr(teleop, "Webcam", lambda device=0, **kwargs: camera)

    # Engage the clutch immediately, drive for a while, then save and quit.
    keys.extend([ord(" ")] + [255] * 40 + [ord("s")] + [255] * 5 + [ord("q")])

    stats = teleop.run_teleop(
        output=tmp_path, device=0, seed=11
    )

    assert stats.frames_seen > 0
    assert stats.frames_tracked > 0, "MediaPipe found no hand in a photograph of a hand"
    assert shown, "nothing was ever drawn to the preview window"
    assert camera.closed

    episodes = list_episodes(tmp_path)
    assert episodes, "the operator pressed save but nothing was written"
    episode = load_episode(episodes[0])
    assert len(episode) > 0
    assert OPERATOR_CAMERA in episode.images
    assert OPERATOR_CAMERA not in episode.policy_cameras
    assert episode.metadata["seed"] is not None
    assert stats.episodes_saved == 1


def test_teleop_keys_reach_the_session(tmp_path, hand_frame, stub_display, monkeypatch):
    import handrobot.teleop as teleop

    _, keys = stub_display
    camera = FakeWebcam(hand_frame, frames=80)
    monkeypatch.setattr(teleop, "Webcam", lambda device=0, **kwargs: camera)

    # engage, drive, home, new episode, discard, quit
    keys.extend(
        [ord(" ")] + [255] * 10 + [ord("h")] + [255] * 10
        + [ord("n")] + [255] * 10 + [ord("d")] + [255] * 5 + [ord("q")]
    )
    stats = teleop.run_teleop(output=tmp_path, device=0, seed=3)
    assert stats.episodes_discarded >= 1
    assert list_episodes(tmp_path) == []


def test_teleop_survives_frames_with_no_hand(tmp_path, stub_display, monkeypatch):
    import handrobot.teleop as teleop

    _, keys = stub_display
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    camera = FakeWebcam(blank, frames=25)
    monkeypatch.setattr(teleop, "Webcam", lambda device=0, **kwargs: camera)
    keys.extend([ord(" ")] + [255] * 20 + [ord("q")])

    stats = teleop.run_teleop(output=tmp_path, device=0, seed=1)
    assert stats.frames_seen > 0
    assert stats.frames_tracked == 0
    assert list_episodes(tmp_path) == []


def test_teleop_without_recording_writes_nothing(tmp_path, hand_frame, stub_display, monkeypatch):
    import handrobot.teleop as teleop

    _, keys = stub_display
    camera = FakeWebcam(hand_frame, frames=20)
    monkeypatch.setattr(teleop, "Webcam", lambda device=0, **kwargs: camera)
    keys.extend([ord(" ")] + [255] * 15 + [ord("q")])

    teleop.run_teleop(output=None, device=0, seed=1)
    assert not (tmp_path / "episodes").exists()


def test_handcheck_reports_a_usable_camera(hand_frame, stub_display, monkeypatch):
    import handrobot.diagnostics as diagnostics
    import handrobot.hands.tracker as tracker_module

    _, keys = stub_display
    camera = FakeWebcam(hand_frame, frames=30)
    monkeypatch.setattr(tracker_module, "Webcam", lambda device=0, **kwargs: camera)
    keys.extend([255] * 29 + [ord("q")])

    assert diagnostics.run_handcheck(device=0) == 0


def test_handcheck_fails_when_no_hand_is_visible(stub_display, monkeypatch):
    import handrobot.diagnostics as diagnostics
    import handrobot.hands.tracker as tracker_module

    _, keys = stub_display
    camera = FakeWebcam(np.zeros((480, 640, 3), np.uint8), frames=20)
    monkeypatch.setattr(tracker_module, "Webcam", lambda device=0, **kwargs: camera)
    keys.extend([255] * 19 + [ord("q")])

    assert diagnostics.run_handcheck(device=0) == 1


def test_the_sim_view_can_be_cycled(tmp_path, hand_frame, stub_display, monkeypatch):
    """The MuJoCo 3D window cannot coexist with an OpenCV window on macOS, so the
    simulator view lives inside the preview and 'v' cycles it."""
    import handrobot.teleop as teleop

    shown, keys = stub_display
    camera = FakeWebcam(hand_frame, frames=12)
    monkeypatch.setattr(teleop, "Webcam", lambda device=0, **kwargs: camera)
    keys.extend([ord("v")] + [255] * 3 + [ord("v")] + [255] * 3 + [ord("q")])

    teleop.run_teleop(output=None, device=0, seed=1)
    # Every preview frame is the camera and a simulator view stacked horizontally.
    assert shown
    for frame in shown:
        assert frame.shape[1] > frame.shape[0], "preview should be side by side"


def test_an_invalid_starting_view_falls_back(tmp_path, hand_frame, stub_display, monkeypatch):
    import handrobot.teleop as teleop

    _, keys = stub_display
    camera = FakeWebcam(hand_frame, frames=6)
    monkeypatch.setattr(teleop, "Webcam", lambda device=0, **kwargs: camera)
    keys.extend([255] * 5 + [ord("q")])
    teleop.run_teleop(output=None, device=0, seed=1, sim_view="nonsense_cam")


def test_the_default_panel_stacks_a_top_and_a_front_view(tmp_path, hand_frame,
                                                         stub_display, monkeypatch):
    """Neither view alone is enough: the top one shows horizontal alignment, the
    front one shows height."""
    import handrobot.teleop as teleop

    shown, keys = stub_display
    camera = FakeWebcam(hand_frame, frames=8)
    monkeypatch.setattr(teleop, "Webcam", lambda device=0, **kwargs: camera)
    keys.extend([255] * 7 + [ord("q")])

    teleop.run_teleop(output=None, device=0, seed=1, sim_view="top+front")
    frame = shown[-1]
    panel = frame[:, frame.shape[1] // 2 :]
    top_half = panel[: panel.shape[0] // 2]
    bottom_half = panel[panel.shape[0] // 2 :]
    assert not np.array_equal(top_half, bottom_half), "the two views are identical"
