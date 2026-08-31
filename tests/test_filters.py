"""Tests for the adaptive smoothing.

The whole point of the One Euro filter is that it behaves differently depending
on speed, so both regimes are checked: a still signal must come out much
cleaner, and a moving one must not fall behind.
"""

import math

import numpy as np
import pytest

from handrobot.filters import AngleFilter, OneEuroFilter, _alpha

DT = 1 / 30


def test_alpha_is_between_zero_and_one():
    for cutoff in (0.1, 1.0, 50.0):
        assert 0.0 < _alpha(cutoff, DT) <= 1.0


def test_a_higher_cutoff_smooths_less():
    assert _alpha(10.0, DT) > _alpha(0.5, DT)


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff=0.0)
    with pytest.raises(ValueError):
        OneEuroFilter(beta=-1.0)


def test_first_sample_passes_through_unchanged():
    f = OneEuroFilter(min_cutoff=0.5, beta=0.5)
    assert np.allclose(f(np.array([1.0, 2.0, 3.0]), DT), [1.0, 2.0, 3.0])
    assert f.initialised


def test_shape_changes_are_rejected():
    f = OneEuroFilter()
    f(np.zeros(3), DT)
    with pytest.raises(ValueError):
        f(np.zeros(2), DT)


def test_reset_forgets_everything():
    f = OneEuroFilter()
    f(np.ones(2), DT)
    f.reset()
    assert not f.initialised and f.value is None


def test_a_still_noisy_signal_is_strongly_smoothed():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=0.6, beta=0.5, d_cutoff=1.0)
    raw, filtered = [], []
    for _ in range(400):
        sample = rng.normal(scale=0.01, size=1)
        raw.append(sample[0])
        filtered.append(f(sample, DT)[0])
    assert np.std(filtered[50:]) < 0.25 * np.std(raw[50:])


def test_a_fast_move_is_followed_closely():
    """Adaptivity is the point: the same filter must not lag a deliberate move."""
    f = OneEuroFilter(min_cutoff=0.6, beta=0.5, d_cutoff=1.0)
    fixed_lag, adaptive_lag = None, None

    position = 0.0
    for step in range(60):
        position += 0.02  # 0.6 m/s, a brisk hand movement
        adaptive = f(np.array([position]), DT)[0]
    adaptive_lag = position - adaptive

    # The same filter with beta = 0 is a plain low-pass, and lags much more.
    g = OneEuroFilter(min_cutoff=0.6, beta=0.0, d_cutoff=1.0)
    position = 0.0
    for step in range(60):
        position += 0.02
        fixed = g(np.array([position]), DT)[0]
    fixed_lag = position - fixed

    assert adaptive_lag < fixed_lag * 0.6, (
        f"adaptive lag {adaptive_lag:.4f} vs fixed {fixed_lag:.4f}"
    )


def test_each_component_is_filtered_independently():
    f = OneEuroFilter(min_cutoff=0.5, beta=0.5)
    f(np.array([0.0, 0.0]), DT)
    for _ in range(40):
        out = f(np.array([1.0, 0.0]), DT)
    assert out[0] > 0.5
    assert abs(out[1]) < 1e-6


def test_angle_filter_does_not_take_the_long_way_round():
    f = AngleFilter(min_cutoff=2.0, beta=0.5)
    f(3.0, DT)
    # Crossing pi: the short way is +0.28 rad, the long way is -6.0.
    for _ in range(30):
        out = f(-3.0, DT)
    assert abs(((out - -3.0) + math.pi) % (2 * math.pi) - math.pi) < 0.05


def test_angle_filter_output_is_always_wrapped():
    f = AngleFilter(min_cutoff=5.0, beta=1.0)
    for angle in np.linspace(-10, 10, 200):
        assert -math.pi - 1e-9 <= f(float(angle), DT) <= math.pi + 1e-9


def test_half_turn_symmetric_filter_ignores_a_pi_flip():
    """A parallel jaw is symmetric: pointing the fingers the opposite way is the
    same grasp, and the wrist must not spin to chase it."""
    f = AngleFilter(min_cutoff=5.0, beta=1.0, half_turn_symmetric=True)
    f(0.2, DT)
    for _ in range(40):
        out = f(0.2 - math.pi, DT)
    assert abs(((out - 0.2) + math.pi) % (2 * math.pi) - math.pi) < 0.05


def test_per_axis_cutoffs_smooth_axes_differently():
    """A noisy axis must be smoothable harder than a clean one, in one filter."""
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=[6.0, 0.3], beta=[0.0, 0.0])
    raw, filtered = [], []
    for _ in range(400):
        sample = rng.normal(scale=0.01, size=2)
        raw.append(sample)
        filtered.append(f(sample, DT))
    raw, filtered = np.array(raw), np.array(filtered)
    loose = np.std(filtered[50:, 0]) / np.std(raw[50:, 0])
    tight = np.std(filtered[50:, 1]) / np.std(raw[50:, 1])
    assert tight < loose / 2, f"loose axis kept {loose:.3f}, tight axis {tight:.3f}"


def test_mismatched_cutoff_length_is_rejected():
    f = OneEuroFilter(min_cutoff=[1.0, 2.0])
    f(np.zeros(2), DT)
    g = OneEuroFilter(min_cutoff=[1.0, 2.0])
    g(np.zeros(3), DT)
    with pytest.raises(ValueError):
        g(np.zeros(3), DT)


def test_rate_limiter_passes_a_reasonable_command_through():
    from handrobot.filters import RateLimiter

    limiter = RateLimiter(max_speed=3.0, dt=DT)
    limiter(np.zeros(6))
    gentle = np.full(6, 0.05)  # 1.5 rad/s, well inside the limit
    assert np.allclose(limiter(gentle), gentle)
    assert limiter.clipped == 0


def test_rate_limiter_caps_a_lurch():
    from handrobot.filters import RateLimiter

    limiter = RateLimiter(max_speed=3.0, dt=DT)
    limiter(np.zeros(6))
    out = limiter(np.full(6, 2.0))
    assert np.allclose(out, 3.0 * DT)
    assert limiter.clipped == 1


def test_rate_limiter_still_reaches_the_target_over_time():
    from handrobot.filters import RateLimiter

    limiter = RateLimiter(max_speed=3.0, dt=DT)
    limiter(np.zeros(2))
    target = np.array([1.0, -1.0])
    for _ in range(40):
        out = limiter(target)
    assert np.allclose(out, target)


def test_rate_limiter_rejects_bad_parameters():
    from handrobot.filters import RateLimiter

    with pytest.raises(ValueError):
        RateLimiter(max_speed=0.0, dt=DT)
    with pytest.raises(ValueError):
        RateLimiter(max_speed=1.0, dt=0.0)


def test_rate_limiter_reset_reseeds():
    from handrobot.filters import RateLimiter

    limiter = RateLimiter(max_speed=3.0, dt=DT)
    limiter(np.zeros(3))
    limiter.reset(np.full(3, 0.4))
    assert np.allclose(limiter.value, 0.4)
    assert limiter.clipped == 0
