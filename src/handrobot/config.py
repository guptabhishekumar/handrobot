"""Typed configuration for every stage of the pipeline.

Values here are the single source of truth: the simulator, the retargeter, the
dataset writer and the policy all read the same dataclasses so that a change in
one place cannot silently desynchronise the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from handrobot.robots import (
    BoxLayout,
    CylinderSector,
    PolarLayout,
    RobotSpec,
    WorkspaceBox,
    get_robot,
)

#: Kept under its old name: the SO-101's reachable region.
WorkspaceConfig = CylinderSector

#: Kept for the SO-101, whose actuator names several tests still refer to.
#: Anything robot-agnostic should read these from the :class:`RobotSpec`.
ARM_ACTUATORS: tuple[str, ...] = get_robot("so101").actuators
GRIPPER_INDEX = get_robot("so101").gripper_index
N_ARM_JOINTS = get_robot("so101").n_arm_joints


@dataclass(frozen=True)
class GripperConfig:
    """How far the jaws are allowed to open and close under teleoperation.

    The angle-to-gap conversion itself lives in
    :class:`~handrobot.gripper.GripperCalibration`, measured from the model,
    because the jaw is a hinge and the relationship is not linear.
    """

    #: Jaw gap treated as "fully closed" by the teleop mapping, in metres.
    min_command_gap: float = 0.010
    #: Jaw gap treated as "fully open" by the teleop mapping, in metres.
    max_command_gap: float = 0.075


@dataclass(frozen=True)
class CameraConfig:
    """One rendered viewpoint."""

    name: str
    width: int
    height: int


@dataclass(frozen=True)
class SimConfig:
    """Simulator timing, cameras and episode termination."""

    control_hz: float = 30.0
    #: 600 Hz physics divides the 30 Hz control period exactly (frame_skip = 20).
    physics_timestep: float = 1.0 / 600.0
    #: Episode limit for scripted rollouts and for scoring a policy.
    max_episode_steps: int = 400
    #: Episode limit while a human is teleoperating. Far longer: a person
    #: lining a gripper up by hand is not on the same clock as a controller
    #: that already knows where everything is, and silently cutting the
    #: recording off mid-demonstration is worse than a long episode.
    teleop_max_steps: int = 2000

    #: Images stored in the dataset and fed to the policy.
    policy_cameras: tuple[CameraConfig, ...] = (
        CameraConfig("front_cam", 128, 128),
        CameraConfig("wrist_cam", 128, 128),
    )
    #: High-resolution viewpoint used only for rendering demo videos.
    render_camera: str = "hero_cam"
    render_width: int = 1280
    render_height: int = 720

    #: Cube must be inside the bin and below this height above the bin floor.
    success_xy_tolerance: float = 0.035
    success_z_max: float = 0.045
    #: Consecutive control steps the success condition must hold.
    success_hold_steps: int = 10

    @property
    def frame_skip(self) -> int:
        n = round((1.0 / self.control_hz) / self.physics_timestep)
        if n < 1:
            raise ValueError("control_hz is faster than the physics timestep")
        return int(n)

    @property
    def control_dt(self) -> float:
        return self.frame_skip * self.physics_timestep


@dataclass(frozen=True)
class IKConfig:
    """Differential inverse-kinematics solver settings."""

    position_cost: float = 1.0
    #: Deliberately below the position cost: the SO-101 arm has five joints and
    #: cannot in general realise a full six-DoF pose, so position wins ties.
    orientation_cost: float = 0.35
    posture_cost: float = 2e-3
    lm_damping: float = 1e-2
    iterations: int = 12
    integration_dt: float = 0.05
    solver: str = "daqp"
    #: Reject a solve whose residual exceeds these bounds.
    max_position_error: float = 0.012
    max_orientation_error: float = 0.45

    #: Largest change from the warm start any single solve may return, in
    #: radians. An arm with joints to spare can reach the same tool pose in
    #: many ways, and while following a moving target it can flip between them
    #: -- the tool pose stays correct at both ends, but the arm swings violently
    #: through the middle and throws whatever it was holding. Clamping the step
    #: turns a flip into a gradual change over several control periods.
    max_joint_step: float = 0.25


@dataclass(frozen=True)
class HandConfig:
    """Hand tracking and hand-to-gripper mapping.

    The smoothing values are the difference between a teleoperator that is
    pleasant and one that is unusable. They configure a One Euro filter, whose
    cutoff rises with hand speed: still hands get heavy smoothing, deliberate
    movements get almost none. See :mod:`handrobot.filters`.
    """

    #: Two, always. Not because two hands drive the arm -- it has one gripper --
    #: but because the operator's other hand is on the keyboard, and a detector
    #: told to find exactly one hand will sometimes find that one. Seeing both
    #: and choosing deliberately is the only way to stay on the right one.
    num_hands: int = 2

    #: Which hand to follow: "left", "right", or None to follow whichever hand
    #: is nearer the camera when tracking starts and then stay with it.
    #: Expressed as the operator sees themselves, so "right" means the hand a
    #: right-handed person would naturally use.
    prefer_hand: str | None = None

    #: How far a hand may be from the one being followed, in metres, before it
    #: is treated as a different hand rather than the same one having moved.
    hand_match_distance: float = 0.25

    #: Deliberately below MediaPipe's 0.5 default. A rejected frame is not free:
    #: it is a gap in the control loop, and gaps are what make teleoperation
    #: feel unstable. A marginal detection is far more useful than none, because
    #: the speed clamp and the smoothing filter absorb a bad one anyway.
    min_detection_confidence: float = 0.35
    min_presence_confidence: float = 0.3
    min_tracking_confidence: float = 0.3

    #: Horizontal field of view assumed for an uncalibrated webcam, in degrees.
    assumed_hfov_deg: float = 62.0

    #: Metres of gripper travel per metre of hand travel.
    position_gain: float = 1.6

    #: Cutoff in Hz for the two image-plane axes of hand position. Chosen by
    #: sweeping the filter against measured landmark noise and against tracking
    #: lag on a deliberate reach: this keeps about a sixth of the jitter, at the
    #: cost of some lag while the hand is in transit. That is the right trade
    #: for this task -- lag during transit is easy to correct for, whereas
    #: jitter while hovering over a 25 mm cube makes it ungraspable.
    position_min_cutoff: float = 0.6
    #: Cutoff in Hz for the depth axis. Much lower, because monocular depth is
    #: the noisiest signal in the system by a wide margin, and depth jitter
    #: shows up as the gripper lunging towards and away from the operator.
    depth_min_cutoff: float = 0.25
    #: How sharply the position cutoff rises with speed. Kept modest: a high
    #: value lets the noise itself look like speed and reopens the filter.
    position_beta: float = 0.25
    depth_beta: float = 0.15

    orientation_min_cutoff: float = 1.0
    orientation_beta: float = 0.3
    gripper_min_cutoff: float = 1.5
    gripper_beta: float = 0.3
    #: Cutoff for the filters' own internal speed estimates.
    derivative_cutoff: float = 1.0

    #: How far the hand may be from the filter's tracked position before the
    #: filter is treated as stale and restarted, in metres. This happens when
    #: tracking is lost while the clutch is released and the operator moves
    #: their hand somewhere else entirely.
    stale_distance: float = 0.06

    #: Fastest hand movement treated as real, in metres per second. Anything
    #: quicker is a tracking glitch -- MediaPipe occasionally emits a wild frame
    #: -- and is clamped rather than followed, which is what stops the arm
    #: teleporting across the workspace on a single bad detection.
    max_hand_speed: float = 2.0

    #: Hand pinch distance mapped to a fully closed jaw, in metres.
    pinch_closed_m: float = 0.025
    #: Hand pinch distance mapped to a fully open jaw, in metres.
    pinch_open_m: float = 0.085

    #: Reject depth estimates outside this range, in metres. Generous, because
    #: rejecting a frame costs more than accepting a slightly odd one.
    depth_min: float = 0.15
    depth_max: float = 1.60

    #: A gap in tracking longer than this re-anchors the mapping instead of
    #: following the hand to wherever it reappeared. Losing the hand is exactly
    #: like lifting a mouse: when it comes back, the arm should stay where it
    #: is, not leap across the workspace to catch up.
    regrab_after: float = 0.25

    #: Fastest the commanded joints may move, in radians per second. A hard
    #: ceiling on how violently the arm can react to anything upstream -- a bad
    #: detection, a solver hiccup, an operator's flinch. Real servos have one;
    #: this makes the simulated arm behave like hardware rather than like maths.
    max_joint_speed: float = 3.0

    #: Radius of the position deadband, in metres. The commanded hand position
    #: is dragged behind the smoothed one on a rope this long: jitter smaller
    #: than the radius moves nothing at all, so a still hand commands a still
    #: arm, while any real movement takes up the slack and is then followed at
    #: the hand's own pace. Sized just above the smoothed noise excursions
    #: (measured: excursions never reach 3.5 mm); the price is twice this in
    #: dead hand travel on each reversal of direction. Chosen over a velocity
    #: test deliberately: measured smoothed
    #: noise *speed* overlaps the slowest deliberate drags, so no speed
    #: threshold can separate the two, but their displacements separate cleanly.
    deadband_radius: float = 0.003
    #: While inside the deadband, the commanded point crawls towards the
    #: average of the last second of raw frames at this rate, in metres per
    #: second -- far below anything visible, but enough to drain parked error
    #: away within a couple of seconds instead of freezing it in. The long
    #: window matters: a short average is itself noisy, and chasing it was
    #: measured to wander the arm 3 mm.
    deadband_recenter: float = 0.0015

    #: Fastest the commanded gripper position may travel, in metres per second.
    #: The joint-space rate limiter already protects the arm; this protects the
    #: *feel*: any discontinuity upstream -- a re-engage somewhere new, a filter
    #: restart, an anchor shift -- becomes a bounded glide instead of a lunge.
    max_command_speed: float = 0.8

    #: Control period assumed when a pose carries no usable timestamp.
    nominal_dt: float = 1.0 / 30.0


@dataclass
class TrainConfig:
    """ACT training hyper-parameters."""

    chunk_size: int = 32
    hidden_dim: int = 512
    dim_feedforward: int = 2048
    n_encoder_layers: int = 4
    n_decoder_layers: int = 6
    n_heads: int = 8
    latent_dim: int = 32
    dropout: float = 0.1
    kl_weight: float = 10.0

    batch_size: int = 32
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    steps: int = 12000
    warmup_steps: int = 500
    log_every: int = 50
    save_every: int = 2000
    val_fraction: float = 0.1
    seed: int = 0
    num_workers: int = 4

    #: Temporal-ensembling decay used at inference. Lower = smoother.
    temporal_ensemble_coeff: float = 0.01

    image_size: int = 128
    augment: bool = True


@dataclass
class Config:
    """Everything, in one object.

    The robot is chosen by name; the workspace, the object sizes and where they
    spawn all come from that choice rather than being repeated here, so no two
    of them can drift apart.
    """

    robot: str = "panda"
    sim: SimConfig = field(default_factory=SimConfig)
    ik: IKConfig | None = None
    hand: HandConfig = field(default_factory=HandConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def spec(self) -> RobotSpec:
        return get_robot(self.robot)

    def __post_init__(self) -> None:
        if self.ik is None:
            # The posture term depends on whether the arm has joints to spare.
            self.ik = IKConfig(posture_cost=self.spec.ik_posture_cost)

    @property
    def gripper(self) -> GripperConfig:
        """How far the jaws open and close, from the robot."""
        low, high = self.spec.gripper_gap_range
        return GripperConfig(min_command_gap=low, max_command_gap=high)

    @property
    def workspace(self):
        """Region the gripper may be commanded into, from the robot."""
        return self.spec.workspace

    @property
    def layout(self):
        """Where the objects spawn, from the robot."""
        return self.spec.layout

    # Old name, still used in a few places.
    @property
    def randomization(self):
        return self.spec.layout
