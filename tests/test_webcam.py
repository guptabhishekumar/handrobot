"""Opening the camera.

The first run on macOS is the one that triggers the permission prompt, and it
fails while the dialog is still on screen. That is not an error condition worth
crashing on, so it is retried -- and the message it eventually gives has to tell
the user the one thing that actually fixes it.
"""

import numpy as np
import pytest

from handrobot.hands.tracker import Webcam


class FakeCapture:
    """Stands in for cv2.VideoCapture."""

    def __init__(self, opens_after: int = 0, frame_shape=(480, 640, 3)) -> None:
        self.opens_after = opens_after
        self.frame_shape = frame_shape
        self.released = False

    def isOpened(self):
        return self.opens_after <= 0

    def read(self):
        if self.opens_after > 0:
            return False, None
        return True, np.full(self.frame_shape, 128, np.uint8)

    def set(self, prop, value):
        return True

    def get(self, prop):
        return 0.0

    def release(self):
        self.released = True


@pytest.fixture
def patch_capture(monkeypatch):
    import cv2

    state = {"calls": 0, "fail_first": 0, "frame_shape": (480, 640, 3)}

    def factory(device, backend=None):
        state["calls"] += 1
        remaining = state["fail_first"] - (state["calls"] - 1)
        return FakeCapture(opens_after=max(remaining, 0), frame_shape=state["frame_shape"])

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    monkeypatch.setattr("time.sleep", lambda s: None)
    return state


def test_opens_a_working_camera(patch_capture):
    camera = Webcam(0)
    assert (camera.width, camera.height) == (640, 480)
    frame = camera.read()
    assert frame.shape == (480, 640, 3)
    camera.close()


def test_size_comes_from_the_frame_not_the_driver(patch_capture):
    """Drivers routinely report a resolution they do not deliver."""
    patch_capture["frame_shape"] = (720, 1280, 3)
    camera = Webcam(0, width=640, height=480)
    assert (camera.width, camera.height) == (1280, 720)
    camera.close()


def test_a_camera_that_appears_late_is_retried(patch_capture, capsys):
    """This is the macOS permission dialog: refused now, allowed a moment later."""
    patch_capture["fail_first"] = 3
    camera = Webcam(0, attempts=4, retry_delay=0.0)
    assert camera.read() is not None
    assert "waiting for camera access" in capsys.readouterr().out
    camera.close()


def test_a_camera_that_never_opens_raises_an_actionable_message(patch_capture):
    patch_capture["fail_first"] = 999
    with pytest.raises(RuntimeError) as excinfo:
        Webcam(0, attempts=2, retry_delay=0.0)
    message = str(excinfo.value)
    assert "Privacy & Security" in message
    assert "Cmd+Q" in message, "the fix is relaunching, and the message must say so"
    assert "tccutil" in message


def test_iteration_stops_when_the_camera_stops(patch_capture):
    camera = Webcam(0)
    frames = []
    for i, frame in enumerate(camera):
        frames.append(frame)
        if i == 2:
            camera.capture.opens_after = 1  # simulate the stream ending
    assert len(frames) == 3
    camera.close()


def test_close_releases_the_device(patch_capture):
    camera = Webcam(0)
    capture = camera.capture
    camera.close()
    assert capture.released


def test_context_manager_closes(patch_capture):
    with Webcam(0) as camera:
        capture = camera.capture
    assert capture.released
