"""Shared fixtures.

Anything that is not specifically about one arm runs against both, because the
code is written against a robot description and a suite that only ever exercised
one of them would not prove that.
"""

import numpy as np
import pytest

from handrobot.config import Config
from handrobot.retarget.ik import ArmIK
from handrobot.robots import ROBOTS, get_robot


@pytest.fixture(params=sorted(ROBOTS))
def robot(request) -> str:
    """Each supported arm in turn."""
    return request.param


@pytest.fixture
def config(robot) -> Config:
    return Config(robot=robot)


@pytest.fixture
def spec(robot):
    return get_robot(robot)


@pytest.fixture
def ik(config) -> ArmIK:
    return ArmIK(config.ik, config.spec)


@pytest.fixture
def rest(config) -> np.ndarray:
    """A sensible warm start: the arm's own home pose."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(config.spec.scene_xml))
    return model.key(config.spec.home_key).ctrl.copy()


@pytest.fixture
def env(config):
    from handrobot.sim import PickPlaceEnv

    environment = PickPlaceEnv(config=config, render_cameras=(), seed=0)
    yield environment
    environment.close()


@pytest.fixture(scope="session")
def hand_landmarks():
    """Landmarks detected from the bundled photograph of a real hand."""
    import cv2

    from handrobot.hands.tracker import HandTracker
    from handrobot.paths import PROJECT_ROOT

    path = PROJECT_ROOT / "tests" / "fixtures" / "hand_sample.jpg"
    if not path.exists():
        pytest.skip("hand fixture image is not present")
    bgr = cv2.imread(str(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tracker = HandTracker(rgb.shape[1], rgb.shape[0])
    pose, landmarks = tracker.detect(rgb)
    tracker.close()
    if landmarks is None:
        pytest.skip("hand detector found nothing in the fixture image")
    return pose, landmarks
