import numpy as np
import pytest

from handrobot.gripper import GripperCalibration


@pytest.fixture
def calibration(spec):
    return GripperCalibration.cached(spec)


def test_gap_is_monotone_in_the_command(calibration):
    assert np.all(np.diff(calibration.gaps) > 0)
    assert np.all(np.diff(calibration.commands) > 0)


def test_conversions_round_trip(calibration):
    for gap in np.linspace(calibration.gap_min, calibration.gap_max, 25):
        command = calibration.gap_to_command(gap)
        assert calibration.command_to_gap(command) == pytest.approx(gap, abs=1e-4)


def test_conversions_clamp_outside_the_measured_range(calibration):
    assert calibration.gap_to_command(-1.0) == pytest.approx(calibration.command_min)
    assert calibration.gap_to_command(10.0) == pytest.approx(calibration.command_max)
    assert calibration.command_to_gap(-1e6) == pytest.approx(calibration.gap_min)
    assert calibration.command_to_gap(1e6) == pytest.approx(calibration.gap_max)


def test_the_so101_relationship_is_genuinely_nonlinear():
    """Its jaw is a hinge, so the gap is a chord. A straight-line fit is out by
    almost a centimetre in the middle, which is why this is measured."""
    from handrobot.robots import get_robot

    calibration = GripperCalibration.cached(get_robot("so101"))
    straight_line = np.interp(
        calibration.commands,
        [calibration.command_min, calibration.command_max],
        [calibration.gap_min, calibration.gap_max],
    )
    worst = float(np.max(np.abs(straight_line - calibration.gaps)))
    assert worst > 0.005, f"largest deviation from a straight line is only {worst * 1000:.1f} mm"


def test_calibration_round_trips_through_disk(tmp_path, calibration):
    restored = GripperCalibration.load(calibration.save(tmp_path / "gripper.npz"))
    assert np.allclose(restored.commands, calibration.commands)
    assert np.allclose(restored.gaps, calibration.gaps)
    assert restored.robot == calibration.robot


def test_measurement_is_reproducible(calibration, spec):
    fresh = GripperCalibration.measure(spec)
    assert np.allclose(fresh.gaps, calibration.gaps, atol=1e-9)
