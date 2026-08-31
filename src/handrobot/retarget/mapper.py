"""Map a tracked human hand onto a gripper command.

The arm has five joints before the gripper, so it has five controllable degrees
of freedom: three of position, one of jaw rotation about the vertical, and the
jaw opening. The approach *pitch* is not free -- it is dictated by how far out
and how high the target is (see :mod:`handrobot.retarget.reach`).

This module therefore maps the hand onto exactly those five, and no more.
Commanding a full six-DoF pose from the hand would look reasonable and behave
badly: the solver would quietly trade position away to chase an orientation the
arm cannot hold, and the gripper would drift from where the operator put it.

The three mappings:

* **Position is relative.** Monocular depth is the weakest signal available, so
  hand translation is measured from an anchor captured when the operator engages
  the clutch, scaled by a gain, and added to the gripper's position at that
  moment. Releasing and re-engaging re-anchors, exactly like lifting a mouse.
  It comes from the rigid palm, never the fingertips, so closing your hand to
  grasp something does not also drag the arm sideways.
* **Jaw rotation is relative, like position.** The line through the operator's
  knuckles, flattened into the horizontal plane, is measured against its
  direction at the moment of engaging, and that *change* is applied to the jaw
  angle the arm already holds. An absolute mapping reads naturally right up
  until the operator releases the clutch, repositions, and re-engages with the
  wrist turned -- at which point the wrist of the arm snaps to the new absolute
  angle, dragging the whole arm with it. Anchoring the angle exactly like the
  position makes re-engaging seamless by construction, and still turns the jaws
  clockwise when the operator turns clockwise.
* **Jaw opening is absolute.** Pinch distance maps onto jaw gap.

Every channel is smoothed by a One Euro filter rather than a fixed exponential
one, and the depth axis gets a much lower cutoff than the other two because it
is far noisier. See :mod:`handrobot.filters`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from handrobot.config import GripperConfig, HandConfig, WorkspaceConfig
from handrobot.filters import AngleFilter, OneEuroFilter
from handrobot.hands.types import HandPose


@dataclass(frozen=True)
class GripperCommand:
    """The five degrees of freedom the operator controls."""

    position: np.ndarray
    """Gripper site position in the robot base frame."""

    jaw_azimuth: float
    """Direction the jaws open, in radians within the horizontal plane."""

    jaw_gap: float
    """Commanded distance between the jaw tips, in metres."""

    engaged: bool
    """Whether the clutch was engaged when this command was produced."""


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to the half-open interval (-pi, pi]."""
    wrapped = float((angle + np.pi) % (2 * np.pi) - np.pi)
    # The modulo lands exact half turns on -pi; the documented interval is open there.
    return np.pi if wrapped == -np.pi else wrapped


def wrap_to_half_turn(angle: float) -> float:
    """Wrap an angle to (-pi/2, pi/2].

    A parallel jaw is unchanged by a half turn -- opening across a line is the
    same grasp whichever end you call the front -- so the two branches are
    interchangeable and only one of them need ever be commanded. Choosing the
    near-zero branch matters because the far one is not always reachable: the
    SO-101's wrist runs out of travel at exactly half a turn, and asking for it
    sends the arm somewhere else entirely.
    """
    wrapped = float((angle + np.pi / 2) % np.pi - np.pi / 2)
    return np.pi / 2 if wrapped == -np.pi / 2 else wrapped


class HandToGripper:
    """Stateful hand-to-gripper retargeter with a clutch and smoothing."""

    #: Camera axes are (right, down, forward). Which robot axis each maps to is
    #: decided by what the operator sees and does, and the correct map is a
    #: MIRROR (determinant -1), not a rotation. That is not a bug; it is the
    #: geometry of a selfie view:
    #:
    #: the camera faces the operator, which reverses rotation sense once, and
    #: the preview is mirrored, which reverses it again. Two reversals cancel,
    #: so a mirror map preserves the operator's own sense of clockwise -- and it
    #: is the only map under which all three translations read naturally at the
    #: same time:
    #:
    #: * hand right            -> robot +y  (rightwards in every operator view)
    #: * hand down             -> robot -z  (downwards)
    #: * hand pushed forwards, towards the screen -> robot -x (away, on screen)
    #:
    #: The previous version used a proper rotation on principle, which can fix
    #: left/right or push/pull but never both, and inverts perceived clockwise.
    #: ``tests/test_screen_directions.py`` pins all of this to rendered pixels.
    CAMERA_TO_ROBOT = np.array(
        [
            [0.0, 0.0, 1.0],   # camera forward (towards operator) -> robot +x
            [1.0, 0.0, 0.0],   # camera right                      -> robot +y
            [0.0, -1.0, 0.0],  # camera down                       -> robot -z
        ]
    )

    def __init__(
        self,
        hand_config: HandConfig | None = None,
        workspace: WorkspaceConfig | None = None,
        gripper: GripperConfig | None = None,
    ) -> None:
        self.hand = hand_config or HandConfig()
        self.workspace = workspace or WorkspaceConfig()
        self.gripper = gripper or GripperConfig()

        # Metres of gripper travel per metre of hand travel. Adjustable at
        # runtime: the comfortable value depends on how far the operator sits
        # from the camera and how much they like to move, which no default can
        # know. The teleop UI binds this to two keys.
        self.position_gain = float(self.hand.position_gain)

        h = self.hand
        # Camera axes are (right, down, forward). The third is depth, and gets a
        # far lower cutoff than the first two.
        self._hand_filter = OneEuroFilter(
            min_cutoff=[h.position_min_cutoff, h.position_min_cutoff, h.depth_min_cutoff],
            beta=[h.position_beta, h.position_beta, h.depth_beta],
            d_cutoff=h.derivative_cutoff,
        )
        self._azimuth_filter = AngleFilter(
            min_cutoff=h.orientation_min_cutoff, beta=h.orientation_beta,
            d_cutoff=h.derivative_cutoff, half_turn_symmetric=True,
        )
        self._gap_filter = OneEuroFilter(
            min_cutoff=h.gripper_min_cutoff, beta=h.gripper_beta,
            d_cutoff=h.derivative_cutoff,
        )

        self._hand_anchor: np.ndarray | None = None
        self._robot_anchor: np.ndarray | None = None
        self._azimuth_hand_anchor: float | None = None
        self._azimuth_robot_anchor: float | None = None
        self._filtered_position: np.ndarray | None = None
        self._filtered_azimuth: float | None = None
        self._filtered_gap: float | None = None
        self._last_hand: np.ndarray | None = None
        self._last_timestamp: float | None = None
        self._last_dt: float = self.hand.nominal_dt
        self._hand_speed: float = 0.0
        #: The last few raw palm positions, for noise-averaged re-anchoring.
        self._raw_history: deque[np.ndarray] = deque(maxlen=30)
        #: The deadband's dragged point: the hand position actually commanded.
        self._park_point: np.ndarray | None = None
        self._engaged = False

        #: Whether the last command was pushed against the edge of the workspace.
        #: Worth telling the operator: it means their hand is asking for
        #: somewhere the arm cannot go, and no amount of further movement in
        #: that direction will help.
        self.saturated = False

        #: How many times tracking was lost long enough to force a re-anchor.
        #: Surfaced on screen: a high count means the camera is losing the hand,
        #: which no amount of smoothing can fix.
        self.tracking_gaps = 0

    # -- clutch -------------------------------------------------------------

    @property
    def engaged(self) -> bool:
        return self._engaged

    @property
    def anchored(self) -> bool:
        return self._hand_anchor is not None

    @property
    def has_target(self) -> bool:
        """Whether any command has been produced since the last reset."""
        return self._filtered_position is not None

    def engage(self, pose: HandPose, current_position: np.ndarray) -> None:
        """Anchor the mapping to the operator's current hand and gripper pose.

        The anchor is taken from the *smoothed* hand position, which is why the
        filter keeps running even while the clutch is released. Anchoring to a
        single raw frame instead bakes that one frame's landmark noise into
        every subsequent command, multiplied by the position gain -- a
        permanent offset the operator cannot correct for, and a different one
        after every re-engage. That is not a small effect: a few millimetres of
        noise becomes most of a centimetre at the gripper, which is the
        difference between grasping a 25 mm cube and knocking it away.
        """
        # The filter is deliberately sluggish, so at the moment of engaging it
        # can still be centimetres behind a hand that recently moved. Anchoring
        # to a value that is still converging turns the leftover convergence
        # into phantom hand movement -- the arm slides on its own after every
        # re-engage. So the filter is restarted here from the average of the
        # last few *raw* frames: noise-averaged like the smoothed value, but
        # with zero convergence pending. A hand far from even that average was
        # lost and has reappeared elsewhere; then the current frame is all
        # there is.
        seed = (
            np.mean(self._raw_history, axis=0)
            if len(self._raw_history) >= 3
            else np.asarray(pose.palm_position, dtype=float)
        )
        if np.linalg.norm(pose.palm_position - seed) > self.hand.stale_distance:
            seed = np.asarray(pose.palm_position, dtype=float)
            # The history predates wherever the hand went; the recentering
            # crawl must not be pulled towards it.
            self._raw_history.clear()
        self._hand_filter.reset()
        self._hand_filter(seed, self.hand.nominal_dt)
        self._last_hand = None
        self._hand_speed = 0.0
        smoothed = self.track(pose)
        self._hand_anchor = smoothed.copy()
        self._last_hand = smoothed.copy()
        # Anchor the robot end to the command the arm is already being held at,
        # not to its measured pose: the physical arm always trails the command a
        # little (rate limits, actuator dynamics, gravity), and anchoring to the
        # trailing value makes every re-engage start with a small correction
        # jump in exactly the direction the arm was last moving.
        anchor = (
            self._filtered_position
            if self._filtered_position is not None
            else np.asarray(current_position, dtype=float)
        )
        self._robot_anchor = self.workspace.clip(np.asarray(anchor, dtype=float))
        self._filtered_position = self._robot_anchor.copy()
        # The jaw angle re-anchors exactly like the position does.
        self._azimuth_hand_anchor = self.jaw_azimuth_from_hand(pose)
        if self._filtered_azimuth is None:
            self._filtered_azimuth = self._azimuth_hand_anchor
        self._azimuth_robot_anchor = float(self._filtered_azimuth)
        self._park_point = smoothed.copy()
        self._engaged = True

    def disengage(self) -> None:
        """Stop tracking. The gripper holds its last command."""
        self._hand_anchor = None
        self._robot_anchor = None
        self._engaged = False

    def adjust_gain(self, delta: float) -> float:
        """Nudge the position gain, re-anchoring so the arm does not jump.

        Changing the gain rescales the offset from the anchor, which would
        teleport the gripper. Re-anchoring *both* ends -- hand and gripper -- to
        where they are right now makes the change take effect only from the
        operator's next movement.
        """
        self.position_gain = float(np.clip(self.position_gain + delta, 0.3, 4.0))
        if self._last_hand is not None and self._filtered_position is not None:
            # While parked, the held command derives from the lock point, so
            # that -- not the still-wandering smoothed value -- is the hand
            # position the command actually corresponds to.
            anchor = self._park_point if self._park_point is not None else self._last_hand
            self._hand_anchor = anchor.copy()
            self._robot_anchor = self._filtered_position.copy()
        return self.position_gain

    def reset(self) -> None:
        """Clear the clutch and every filter. The gain is deliberately kept."""
        self.disengage()
        self._hand_filter.reset()
        self._azimuth_filter.reset()
        self._gap_filter.reset()
        self._filtered_position = None
        self._filtered_azimuth = None
        self._filtered_gap = None
        self._azimuth_hand_anchor = None
        self._azimuth_robot_anchor = None
        self._last_hand = None
        self._last_timestamp = None
        self._hand_speed = 0.0
        self._raw_history.clear()
        self._park_point = None
        self.saturated = False

    # -- individual mappings ------------------------------------------------

    def jaw_azimuth_from_hand(self, pose: HandPose) -> float:
        """Horizontal direction the jaws should open, from the knuckle line.

        Taken from the rigid palm frame, so it rolls with the operator's wrist
        and does not twitch when they pinch.
        """
        jaw_camera = pose.rotation[:, 2]
        jaw_robot = self.CAMERA_TO_ROBOT @ jaw_camera
        planar = jaw_robot[:2]
        if np.linalg.norm(planar) < 1e-6:
            # The knuckles are edge-on; keep the angle we already had.
            return self._filtered_azimuth if self._filtered_azimuth is not None else 0.0
        return wrap_to_half_turn(float(np.arctan2(planar[1], planar[0])))

    def jaw_gap_from_pinch(self, pinch_distance: float) -> float:
        """Map the thumb-to-index distance onto a commanded jaw gap."""
        span = self.hand.pinch_open_m - self.hand.pinch_closed_m
        t = float(np.clip((float(pinch_distance) - self.hand.pinch_closed_m) / span, 0.0, 1.0))
        return self.gripper.min_command_gap + t * (
            self.gripper.max_command_gap - self.gripper.min_command_gap
        )

    def _plausible(self, position: np.ndarray, dt: float) -> np.ndarray:
        """Clamp a sample to a physically possible hand speed.

        The detector occasionally emits a frame that places the hand somewhere
        it could not have reached. Following it would fling the arm across the
        workspace; clamping turns a glitch into a brief, harmless drag.
        """
        if self._last_hand is None:
            return np.asarray(position, dtype=float)
        step = np.asarray(position, dtype=float) - self._last_hand
        distance = float(np.linalg.norm(step))
        limit = self.hand.max_hand_speed * dt
        if distance > limit > 0:
            return self._last_hand + step * (limit / distance)
        return np.asarray(position, dtype=float)

    def position_from_hand(self, smoothed_hand: np.ndarray) -> np.ndarray:
        """Clutch-anchored gripper position, clipped into the workspace.

        Clipping alone is not enough. Moving your hand past the edge of the
        reachable region keeps growing the offset from the anchor even though
        the arm has stopped, so you then have to retrace every centimetre of
        that overshoot before the arm responds at all -- the control feels dead,
        and you have no way of knowing why.

        The fix is the same one a well-behaved integrator uses: when the command
        saturates, move the anchor by exactly the amount that was clipped away.
        The hand then maps to the boundary while it is outside, and the arm
        starts moving the instant it comes back.
        """
        delta = (smoothed_hand - self._hand_anchor) * self.position_gain
        raw = self._robot_anchor + self.CAMERA_TO_ROBOT @ delta
        clipped = self.workspace.clip(raw)

        excess = clipped - raw
        overshoot = float(np.linalg.norm(excess))
        self.saturated = bool(overshoot > 1e-9)
        # Shift the anchor by all but the last millimetre of the overshoot.
        # Shifting it fully would re-map the boundary exactly onto the hand, at
        # which point a hand held past the edge stops reading as saturated and
        # the operator loses the one signal that explains why the arm stopped.
        # The millimetre left over keeps the flag honest, costs a millimetre of
        # retrace, and caps the windup all the same.
        margin = 1e-3
        if overshoot > margin:
            self._robot_anchor = self._robot_anchor + excess * (1 - margin / overshoot)
        return clipped

    # -- the loop -----------------------------------------------------------

    def hand_move_for(self, robot_delta: np.ndarray) -> np.ndarray:
        """How the operator must move their hand to move the gripper by this much.

        The inverse of the position mapping, in camera axes (right, down,
        forward). Knowing that the gripper is 159 mm from the cube is not
        actionable; knowing to move your hand 40 mm forward and 60 mm left is.
        """
        return (
            np.linalg.inv(self.CAMERA_TO_ROBOT) @ np.asarray(robot_delta, dtype=float)
        ) / max(self.position_gain, 1e-6)

    def hold(self, current_position: np.ndarray) -> GripperCommand:
        """The command to issue when there is nothing to follow."""
        return GripperCommand(
            position=(
                self._filtered_position.copy()
                if self._filtered_position is not None
                else self.workspace.clip(np.asarray(current_position, dtype=float))
            ),
            jaw_azimuth=self._filtered_azimuth if self._filtered_azimuth is not None else 0.0,
            jaw_gap=(
                self._filtered_gap
                if self._filtered_gap is not None
                else self.gripper.max_command_gap
            ),
            engaged=False,
        )

    def timestep(self, pose: HandPose) -> float:
        """Seconds since the previous pose, guarded against a stalled camera."""
        dt = self.hand.nominal_dt
        if self._last_timestamp is not None:
            elapsed = pose.timestamp - self._last_timestamp
            # A stall would otherwise hand the filters a huge timestep and let a
            # single frame snap the arm across the workspace.
            if 1e-4 < elapsed < 0.5:
                dt = elapsed
        self._last_timestamp = pose.timestamp
        return dt

    def track(self, pose: HandPose) -> np.ndarray:
        """Feed one hand pose to the smoothing filter and return the result.

        Called on every frame whether or not the clutch is engaged, so that the
        moment the operator does engage there is already a well-averaged hand
        position to anchor to.
        """
        dt = self.timestep(pose)
        self._last_dt = dt
        self._raw_history.append(np.asarray(pose.palm_position, dtype=float))
        smoothed = self._hand_filter(self._plausible(pose.palm_position, dt), dt)
        if self._last_hand is not None and dt > 0:
            self._hand_speed = float(np.linalg.norm(smoothed - self._last_hand)) / dt
        self._last_hand = smoothed.copy()
        return smoothed

    def __call__(self, pose: HandPose | None, current_position: np.ndarray) -> GripperCommand:
        """Produce the next gripper command.

        With the clutch released, or tracking lost, this holds the last command
        so the arm freezes rather than lurching -- but the hand filter keeps
        running, so re-engaging is instant and noise-free.
        """
        if pose is None:
            return self.hold(current_position)

        # A long gap means the hand was lost and has come back, possibly
        # somewhere else entirely. Re-anchor rather than chase it.
        gap = (
            pose.timestamp - self._last_timestamp
            if self._last_timestamp is not None
            else 0.0
        )
        resumed = gap > self.hand.regrab_after

        smoothed_hand = self.track(pose)
        if not self._engaged or self._hand_anchor is None:
            return self.hold(current_position)

        if resumed:
            self._hand_filter.reset()
            self._raw_history.clear()
            smoothed_hand = self._hand_filter(pose.palm_position, self.hand.nominal_dt)
            self._last_hand = smoothed_hand.copy()
            self._hand_anchor = smoothed_hand.copy()
            self._robot_anchor = (
                self._filtered_position.copy()
                if self._filtered_position is not None
                else self.workspace.clip(np.asarray(current_position, dtype=float))
            )
            self._azimuth_hand_anchor = self.jaw_azimuth_from_hand(pose)
            if self._filtered_azimuth is not None:
                self._azimuth_robot_anchor = float(self._filtered_azimuth)
            self._park_point = smoothed_hand.copy()
            self.tracking_gaps += 1

        # The deadband. The commanded hand position is not the smoothed value
        # but a point dragged behind it on a short rope: inside the radius the
        # point does not move at all, so a still hand's residual jitter --
        # which measures the same *speed* as a deliberate slow drag, and so
        # cannot be told apart by any velocity test -- moves nothing. A real
        # movement takes up the slack and from then on the point follows at
        # exactly the hand's own pace: no hops, no thresholds to mis-trip, and
        # the only cost is one radius of dead travel on each reversal. While
        # inside the radius the point also crawls, well below perception,
        # towards the average of the recent raw frames -- so parked error
        # drains away instead of accumulating.
        if self._park_point is None:
            self._park_point = smoothed_hand.copy()
        offset = smoothed_hand - self._park_point
        distance = float(np.linalg.norm(offset))
        radius = self.hand.deadband_radius
        if distance > radius:
            self._park_point = (
                smoothed_hand - offset * (radius / distance)
                if radius > 0 else smoothed_hand.copy()
            )
        elif len(self._raw_history) >= 15:
            # Crawl only while the parked error is clearly larger than the
            # rolling average's own fluctuation -- otherwise the crawl would
            # chase that fluctuation forever, a slow perpetual wander in
            # exactly the situation that must be perfectly still.
            pull = np.mean(self._raw_history, axis=0) - self._park_point
            length = float(np.linalg.norm(pull))
            crawl = self.hand.deadband_recenter * self._last_dt
            if length > 2 * radius / 3:
                self._park_point = self._park_point + pull * (min(length, crawl) / length)
        effective_hand = self._park_point

        previous_command = (
            self._filtered_position.copy()
            if self._filtered_position is not None
            else None
        )
        target = self.position_from_hand(effective_hand)
        # Whatever happened upstream, the command itself may only glide.
        if previous_command is not None:
            step = target - previous_command
            distance = float(np.linalg.norm(step))
            limit = self.hand.max_command_speed * self._last_dt
            if distance > limit > 0:
                target = previous_command + step * (limit / distance)
        self._filtered_position = target

        raw_azimuth = self.jaw_azimuth_from_hand(pose)
        if self._azimuth_hand_anchor is not None and self._azimuth_robot_anchor is not None:
            turned = wrap_to_half_turn(raw_azimuth - self._azimuth_hand_anchor)
            raw_azimuth = wrap_to_half_turn(self._azimuth_robot_anchor + turned)
        self._filtered_azimuth = self._azimuth_filter(raw_azimuth, self._last_dt)
        self._filtered_gap = float(
            self._gap_filter(self.jaw_gap_from_pinch(pose.pinch_distance), self._last_dt)[0]
        )

        return GripperCommand(
            position=self._filtered_position.copy(),
            jaw_azimuth=float(self._filtered_azimuth),
            jaw_gap=float(self._filtered_gap),
            engaged=True,
        )
