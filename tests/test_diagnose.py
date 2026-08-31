"""Tests for the failure-mode diagnostics.

Each probe is checked against a policy deliberately built to exhibit the failure
it is meant to detect, because a diagnostic that only ever says "looks fine" is
worse than none.
"""

import numpy as np
import pytest
import torch

from handrobot.config import Config
from handrobot.data.dataset import NormalizationStats
from handrobot.policy.act import ACTConfig, ACTPolicy
from handrobot.policy.diagnose import (
    GRASP_TOLERANCE,
    VISION_THRESHOLD,
    Diagnosis,
    measure_grasp_accuracy,
    measure_vision_sensitivity,
)
from handrobot.policy.inference import save_checkpoint


def make_checkpoint(path, action_std=0.4, seed=0, config=None):
    config = config or Config()
    torch.manual_seed(seed)
    policy = ACTPolicy(
        ACTConfig(
            state_dim=len(config.spec.actuators),
            action_dim=len(config.spec.actuators),
            chunk_size=8, cameras=tuple(c.name for c in config.sim.policy_cameras),
            hidden_dim=64, dim_feedforward=128, n_heads=4, n_encoder_layers=1,
            n_decoder_layers=1, n_vae_layers=1, latent_dim=8, dropout=0.0,
            pretrained_backbone=False,
        )
    )
    n = len(config.spec.actuators)
    stats = NormalizationStats(
        state_mean=np.zeros(n, np.float32), state_std=np.ones(n, np.float32),
        action_mean=np.zeros(n, np.float32),
        action_std=np.full(n, action_std, np.float32),
    )
    return save_checkpoint(path, policy, stats, extra={"step": 0})


# -- the verdict logic, which is pure and cheap to test exhaustively -----------


def healthy(**overrides) -> Diagnosis:
    fields = dict(
        grasp_errors=[0.004] * 5, vision_sensitivity=0.5, action_range=0.9,
        success_rate=0.9, episodes=10,
    )
    fields.update(overrides)
    return Diagnosis(**fields)


def test_a_healthy_policy_is_reported_as_healthy():
    report = healthy()
    assert report.moves and report.uses_vision and report.grasp_is_accurate
    text = " ".join(report.verdict()).lower()
    assert "using its cameras" in text
    assert "accuracy is good" in text


def test_a_frozen_policy_is_reported_first_and_alone():
    report = healthy(action_range=0.01, vision_sensitivity=0.0)
    assert not report.moves
    verdict = report.verdict()
    assert len(verdict) == 1, "a frozen arm makes the other readings meaningless"
    assert "barely moves" in verdict[0]


def test_a_policy_ignoring_its_cameras_is_told_to_vary_the_data():
    report = healthy(vision_sensitivity=VISION_THRESHOLD / 2)
    assert not report.uses_vision
    text = " ".join(report.verdict()).lower()
    assert "ignores its cameras" in text
    assert "more training will not help" in text


def test_an_inaccurate_grasp_is_told_to_train_longer():
    report = healthy(grasp_errors=[0.03] * 5)
    assert not report.grasp_is_accurate
    text = " ".join(report.verdict()).lower()
    assert "misses by 30.0 mm" in text
    assert "training longer" in text


def test_failure_despite_healthy_probes_points_at_the_later_motion():
    report = healthy(success_rate=0.1)
    text = " ".join(report.verdict()).lower()
    assert "later in the motion" in text


def test_no_such_hint_when_the_policy_already_succeeds():
    assert "later in the motion" not in " ".join(healthy(success_rate=0.9).verdict()).lower()


def test_mean_grasp_error_is_nan_when_nothing_was_measured():
    assert np.isnan(healthy(grasp_errors=[]).mean_grasp_error)


# -- the probes themselves, against a real untrained policy --------------------


def test_grasp_probe_returns_one_error_per_target(tmp_path, config):
    path = make_checkpoint(tmp_path / "p.pt", config=config)
    errors = measure_grasp_accuracy(path, config, device_name="cpu", grasp_step=8)
    assert len(errors) == 5
    assert all(np.isfinite(e) and e >= 0 for e in errors)


def test_an_untrained_policy_fails_the_grasp_probe(tmp_path, config):
    """An untrained network must not be reported as accurate; that would be worse
    than no diagnostic at all."""
    path = make_checkpoint(tmp_path / "p.pt", config=config)
    errors = measure_grasp_accuracy(path, config, device_name="cpu", grasp_step=12)
    assert np.mean(errors) > GRASP_TOLERANCE


def test_vision_probe_returns_a_difference_and_a_motion(tmp_path, config):
    path = make_checkpoint(tmp_path / "p.pt", config=config)
    difference, motion = measure_vision_sensitivity(
        path, config, device_name="cpu", steps=10
    )
    assert difference >= 0.0 and motion >= 0.0


def test_a_constant_policy_is_detected_as_frozen(tmp_path, config):
    """Scaling the action range to nothing produces the frozen failure mode."""
    path = make_checkpoint(tmp_path / "frozen.pt", action_std=1e-6, config=config)
    difference, motion = measure_vision_sensitivity(
        path, config, device_name="cpu", steps=12
    )
    assert motion < 0.1
    report = Diagnosis(grasp_errors=[0.05] * 5, vision_sensitivity=difference,
                       action_range=motion, success_rate=0.0, episodes=1)
    assert not report.moves
