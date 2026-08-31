"""Speed-adaptive smoothing for noisy interactive signals.

A fixed exponential filter cannot win here. Set it loose and the arm shivers
while your hand is still; set it tight and the arm lags behind every deliberate
move. The One Euro filter (Casiez, Roussel and Vogel, 2012) resolves that by
making the cutoff frequency depend on how fast the signal is currently moving:
heavy smoothing when you hold still, almost none when you move with intent.

Hand landmarks jitter measurably even with a motionless hand, so this is the
difference between a usable teleoperator and an unusable one.
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    """Exponential-smoothing factor for a given cutoff frequency and timestep."""
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroFilter:
    """One Euro filter over a scalar or a fixed-length vector.

    Args:
        min_cutoff: cutoff in Hz while the signal is still. Lower is smoother
            and adds more lag to slow movements. May be per-component, which is
            what lets a noisy axis be smoothed harder than a clean one -- a
            monocular depth estimate is far noisier than the two axes that are
            read straight off the image plane.
        beta: how much the cutoff rises with speed. Higher follows fast motion
            more closely at the cost of letting more jitter through. May also
            be per-component.
        d_cutoff: cutoff in Hz for the internal speed estimate, which is itself
            noisy and needs its own smoothing.
    """

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff: float = 1.0) -> None:
        min_cutoff = np.atleast_1d(np.asarray(min_cutoff, dtype=float))
        beta = np.atleast_1d(np.asarray(beta, dtype=float))
        if np.any(min_cutoff <= 0) or d_cutoff <= 0:
            raise ValueError("cutoff frequencies must be positive")
        if np.any(beta < 0):
            raise ValueError("beta must not be negative")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = float(d_cutoff)
        self._value: np.ndarray | None = None
        self._derivative: np.ndarray | None = None

    @property
    def initialised(self) -> bool:
        return self._value is not None

    @property
    def value(self) -> np.ndarray | None:
        return None if self._value is None else self._value.copy()

    def reset(self) -> None:
        self._value = None
        self._derivative = None

    def __call__(self, x, dt: float) -> np.ndarray:
        """Filter one sample taken ``dt`` seconds after the previous one."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if self._value is None:
            self._value = x.copy()
            self._derivative = np.zeros_like(x)
            return self._value.copy()
        if self._value.shape != x.shape:
            raise ValueError(
                f"sample shape {x.shape} does not match the filter's {self._value.shape}"
            )
        if self.min_cutoff.size not in (1, x.size):
            raise ValueError(
                f"min_cutoff has {self.min_cutoff.size} entries for a {x.size}-vector"
            )

        dt = max(float(dt), 1e-6)
        derivative = (x - self._value) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._derivative = a_d * derivative + (1 - a_d) * self._derivative

        cutoff = self.min_cutoff + self.beta * np.abs(self._derivative)
        alpha = np.array([_alpha(float(c), dt) for c in np.atleast_1d(cutoff)])
        self._value = alpha * x + (1 - alpha) * self._value
        return self._value.copy()


class AngleFilter:
    """One Euro filter for an angle, tracking it across the branch cut.

    Filtering an angle directly makes it swing the long way round whenever it
    crosses pi. This accumulates the unwrapped angle instead, filters that, and
    wraps only on the way out.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0,
                 d_cutoff: float = 1.0, half_turn_symmetric: bool = False) -> None:
        self._filter = OneEuroFilter(min_cutoff, beta, d_cutoff)
        #: A parallel jaw is unchanged by a half turn, so treat angles that
        #: differ by pi as the same and never rotate the wrist to reach one.
        self.half_turn_symmetric = half_turn_symmetric
        self._unwrapped: float | None = None

    @property
    def initialised(self) -> bool:
        return self._unwrapped is not None

    def reset(self) -> None:
        self._filter.reset()
        self._unwrapped = None

    def __call__(self, angle: float, dt: float) -> float:
        angle = float(angle)
        if self._unwrapped is None:
            self._unwrapped = angle
        else:
            period = math.pi if self.half_turn_symmetric else 2 * math.pi
            difference = (angle - self._unwrapped + period / 2) % period - period / 2
            self._unwrapped += difference
        filtered = float(self._filter(self._unwrapped, dt)[0])
        if self.half_turn_symmetric:
            # Continuous rotation accumulates the unwrapped angle past the
            # half-turn branch; wrapping only to a full turn would then hand
            # back an angle on the far branch, which for the jaw is the same
            # grasp but for the wrist can be an unreachable command.
            return float((filtered + math.pi / 2) % math.pi - math.pi / 2)
        return float((filtered + math.pi) % (2 * math.pi) - math.pi)


class RateLimiter:
    """Caps how fast a vector of joint commands is allowed to change.

    A hard ceiling on how violently the arm can react to anything upstream: a
    bad detection, a solver hiccup, an operator's flinch. Real servos have a
    speed limit; without one the simulated arm can be commanded to jump an
    arbitrary distance in a single control period, which looks and feels like
    instability even when every component is behaving correctly.

    Unlike a filter, this changes nothing while the command is moving at a
    reasonable speed. It only ever removes the extremes.
    """

    def __init__(self, max_speed, dt: float) -> None:
        max_speed = np.atleast_1d(np.asarray(max_speed, dtype=float))
        if np.any(max_speed <= 0) or dt <= 0:
            raise ValueError("max_speed and dt must be positive")
        #: May be per-element: a robot's actuators are not all in the same
        #: units. Clamping a gripper whose command runs 0 to 255 at the same
        #: rate as a joint measured in radians would take it minutes to close.
        self.max_speed = max_speed
        self.dt = float(dt)
        self._value: np.ndarray | None = None
        #: How many commands have been clipped. A rising count means something
        #: upstream is asking for motion the arm should not make.
        self.clipped = 0

    @property
    def value(self) -> np.ndarray | None:
        return None if self._value is None else self._value.copy()

    def reset(self, value=None) -> None:
        self._value = None if value is None else np.asarray(value, dtype=float).copy()
        self.clipped = 0

    def __call__(self, command) -> np.ndarray:
        command = np.asarray(command, dtype=float)
        if self._value is None:
            self._value = command.copy()
            return self._value.copy()

        limit = self.max_speed * self.dt
        if limit.size not in (1, command.size):
            raise ValueError(
                f"max_speed has {limit.size} entries for a {command.size}-vector"
            )
        step = command - self._value
        excess = np.abs(step) > limit
        if np.any(excess):
            self.clipped += 1
            step = np.clip(step, -limit, limit)
        self._value = self._value + step
        return self._value.copy()
