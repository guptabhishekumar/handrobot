"""Live teleoperation: drive the simulated arm with your hand, record what you do.

The loop is paced by the webcam. One camera frame produces one control step, so
if frames drop the robot simply moves more slowly rather than jumping -- and the
recorded dataset stays at a fixed 30 Hz because every recorded step is one
control period of simulated time.

Controls
--------
``space``  toggle the clutch. The arm only follows your hand while engaged;
           release to reposition your hand without moving the robot.
``n``      start a new episode (resets the scene and begins recording).
``s``      save the current episode manually.
``d``      discard the current episode and reset.
``h``      send the arm back to its home pose.
``[`` ``]``  make the arm less or more sensitive to your hand.
``v``      swap the lower simulator panel (follow, front, wide, wrist).
``w``      show or hide the wrist picture-in-picture.
``?``      show or hide the key list.
``q``      quit.

Everything happens in one window: the camera on the left, the simulator on the
right. MuJoCo's separate 3D window is deliberately not used -- on macOS it
requires the ``mjpython`` launcher, under which OpenCV cannot open a window at
all, so the two are mutually exclusive. One window that always works beats two
that never do.

An episode that reaches the success condition is saved automatically.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from handrobot.config import Config
from handrobot.data.dataset import EpisodeWriter
from handrobot.filters import RateLimiter
from handrobot.gripper import GripperCalibration
from handrobot.hands.tracker import HandTracker, Webcam
from handrobot.hands.types import INDEX_TIP, THUMB_TIP, HandPose
from handrobot.retarget.ik import ArmIK
from handrobot.retarget.mapper import HandToGripper
from handrobot.retarget.grasp import GraspFrames
from handrobot.sim.env import PickPlaceEnv
from handrobot.viz.hud import (
    REFERENCE_HEIGHT,
    HudState,
    compose,
    draw_scene_overlays,
    panel_geometry,
    view_label,
)
from handrobot.viz.overlay import draw_hand_overlay, hand_is_clipped


#: Name of the operator's webcam stream inside a recorded episode. It is stored
#: for the film and excluded from what any policy is trained on.
OPERATOR_CAMERA = "operator_cam"
OPERATOR_WIDTH = 480
OPERATOR_HEIGHT = 360


@dataclass
class TeleopStats:
    """Counters shown in the on-screen panel."""

    episodes_saved: int = 0
    episodes_discarded: int = 0
    successes: int = 0
    frames_seen: int = 0
    frames_tracked: int = 0
    ik_failures: int = 0
    frames_clipped: int = 0
    recent_fps: list[float] = field(default_factory=list)
    #: How often each rejection reason has come up, so the operator can be told
    #: what to change rather than just that something is wrong.
    rejections: dict[str, int] = field(default_factory=dict)
    #: The last few seconds of frames, and why the failed ones failed. The
    #: session totals are the honest summary at the end, but they are the wrong
    #: thing to *display*: after ten good minutes a session average cannot move,
    #: so an operator whose tracking has just collapsed sees a healthy number.
    #: Three seconds is long enough to average out a blink and short enough to
    #: react to.
    live_window: int = 90
    recent_tracked: deque = field(default_factory=lambda: deque(maxlen=90))
    recent_rejections: deque = field(default_factory=lambda: deque(maxlen=90))
    #: Per-frame quality for the on-screen timeline: 1 tracked, 2 tracked at
    #: the frame edge, 0 lost.
    recent_quality: deque = field(default_factory=lambda: deque(maxlen=240))

    def note_rejection(self, reason: str | None) -> None:
        if reason:
            self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def note_frame(self, tracked: bool, reason: str | None, clipped: bool = False) -> None:
        """Record the outcome of one camera frame, for the totals and the window."""
        self.recent_tracked.append(bool(tracked))
        self.recent_quality.append(0 if not tracked else (2 if clipped else 1))
        if tracked:
            self.frames_tracked += 1
            self.recent_rejections.append(None)
        else:
            self.note_rejection(reason)
            self.recent_rejections.append(reason)

    @property
    def top_rejection(self) -> tuple[str, int] | None:
        if not self.rejections:
            return None
        return max(self.rejections.items(), key=lambda item: item[1])

    @property
    def tracking_rate(self) -> float:
        return self.frames_tracked / max(self.frames_seen, 1)

    @property
    def live_tracking_rate(self) -> float:
        """Fraction of the last few seconds of frames that produced a pose."""
        if not self.recent_tracked:
            return self.tracking_rate
        return sum(self.recent_tracked) / len(self.recent_tracked)

    @property
    def live_rejection(self) -> str | None:
        """The commonest recent reason for losing frames, while it is worth saying.

        Silent above 85% tracked: a handful of dropped frames is normal, and an
        interface that complains about normal is one the operator stops reading.
        """
        if self.live_tracking_rate >= 0.85:
            return None
        reasons: dict[str, int] = {}
        for reason in self.recent_rejections:
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        if not reasons:
            return None
        return max(reasons.items(), key=lambda item: item[1])[0]

    @property
    def fps(self) -> float:
        return float(np.mean(self.recent_fps)) if self.recent_fps else 0.0

    def tick(self, dt: float) -> None:
        self.recent_fps.append(1.0 / max(dt, 1e-6))
        if len(self.recent_fps) > 30:
            self.recent_fps.pop(0)


class TeleopSession:
    """Owns the environment, the retargeter and the recording state."""

    def __init__(
        self,
        env: PickPlaceEnv,
        mapper: HandToGripper,
        ik: ArmIK,
        config: Config,
        writer: EpisodeWriter | None = None,
        grasp: GraspFrames | None = None,
        calibration: GripperCalibration | None = None,
    ) -> None:
        self.env = env
        self.mapper = mapper
        self.ik = ik
        self.spec = config.spec
        self.grasp = grasp or GraspFrames(self.spec)
        self.calibration = calibration or GripperCalibration.cached(self.spec)
        self.gripper_index = self.spec.gripper_index
        # A ceiling on how fast the commanded joints may move. Nothing upstream
        # -- a bad detection, a solver hiccup, a flinch -- can make the arm
        # snap harder than a real servo would.
        self.rate_limiter = RateLimiter(
            self.spec.command_speed_limits(
                self.calibration.command_max - self.calibration.command_min
            ),
            config.sim.control_dt,
        )
        #: Height the gripper site should reach to close on the cube. Matches the
        #: scripted expert, so the on-screen guidance and the scripted
        #: demonstrations agree about what "at the cube" means.
        self.grasp_height = self.spec.cube_half_extent + self.spec.grasp_clearance
        self.config = config
        self.writer = writer
        self.stats = TeleopStats()

        self.recording = False
        self.episode_steps = 0
        self.last_success = False
        self._message = ""
        self._message_time = 0.0
        self.message = "press N to start an episode"
        #: Fades from 1 to 0 after a success, driving the on-screen banner. A
        #: saved success is the only thing that happens without being asked for,
        #: and an operator who misses it repeats an episode that already counted.
        self.flash = 0.0
        self._q = env.joint_positions.copy()
        self._seed: int | None = None
        self._operator_frame: np.ndarray | None = None
        self._homing = False

    #: Seconds a message stays on the strip. Messages report what just
    #: happened; one still sitting there a minute later is describing a
    #: different moment, and the operator has no way of knowing which.
    MESSAGE_SECONDS = 6.0

    @property
    def message(self) -> str:
        return self._message

    @message.setter
    def message(self, text: str) -> None:
        self._message = text
        self._message_time = time.perf_counter()

    @property
    def visible_message(self) -> str:
        """The message, while it is still describing the present."""
        if not self._message:
            return ""
        if time.perf_counter() - self._message_time > self.MESSAGE_SECONDS:
            return ""
        return self._message

    # -- episode control ----------------------------------------------------

    def reseed_rate_limiter(self) -> None:
        """Start the rate limiter from the last command, not the measured pose.

        It constrains the sequence of commands, so seeding it with measured
        joint angles would make the servo's own tracking error look like a
        commanded jump and clip the very first step of every episode.
        """
        self.rate_limiter.reset(self.env.commanded_positions)

    def new_episode(self, seed: int | None = None) -> None:
        # Carry the clutch state across the reset. Dropping it every episode
        # would make the operator press space forty times in a recording session
        # for no reason; the mapper re-anchors on the next tracked frame anyway.
        was_engaged = self.mapper.engaged
        self._seed = seed
        self.env.reset(seed=seed)
        self._q = self.env.joint_positions.copy()
        self.mapper.reset()
        self.mapper._engaged = was_engaged
        self.reseed_rate_limiter()
        self._homing = False
        if self.writer is not None:
            self.writer.discard()
        self.recording = self.writer is not None
        self.episode_steps = 0
        self.last_success = False
        self.message = (
            "recording - move your hand"
            if was_engaged
            else "recording - press space to engage the clutch"
        )

    def save_episode(self, success: bool) -> None:
        if self.writer is None or len(self.writer) == 0:
            self.message = "nothing recorded"
            return
        path = self.writer.finish(
            success=success, metadata={"teleop": True, "seed": self._seed}
        )
        self.stats.episodes_saved += 1
        self.stats.successes += int(success)
        self.recording = False
        if success:
            self.flash = 1.0
        self.message = (
            f"saved {path.name} ({'success' if success else 'failure'}) - press N for the next"
        )

    def discard_episode(self) -> None:
        if self.writer is not None:
            self.writer.discard()
        self.stats.episodes_discarded += 1
        self.recording = False
        self.message = "discarded"

    def go_home(self) -> None:
        """Drive back to the home pose and stay there until the clutch re-engages.

        The homing flag matters: without it the very next control step would run
        inverse kinematics towards the gripper's *current* position and undo the
        move before it finished.
        """
        self._q = self.env.model.key("home").ctrl.copy()
        self.mapper.reset()
        self.reseed_rate_limiter()
        self._homing = True
        self.message = "returning home - press space to take over"

    # -- one control step ---------------------------------------------------

    def set_operator_frame(self, frame: np.ndarray | None) -> None:
        """Supply the annotated webcam image recorded alongside this step."""
        self._operator_frame = frame

    @property
    def needs_images(self) -> bool:
        """Whether this step will be written to the dataset.

        Rendering the policy cameras costs about 8 ms each, which is a quarter of
        the frame budget at 30 Hz. Frames that will not be recorded -- clutch
        released, homing, episode already saved -- do not need them.
        """
        return bool(
            self.recording
            and self.writer is not None
            and self.mapper.engaged
            and not self._homing
            and self.episode_steps < self.config.sim.teleop_max_steps
        )

    def _observe(self) -> Observation:
        return self.env.observe() if self.needs_images else self.env.observe_state()

    def step(self, pose: HandPose | None) -> dict:
        """Advance the simulation by one control period."""
        observation_before = self._observe()
        position, _ = self.env.gripper_pose

        if pose is not None and self.mapper.engaged and not self.mapper.anchored:
            self.mapper.engage(pose, position)

        # Called unconditionally, including while homing or idle, because the
        # smoothing filter has to keep tracking the hand: the clutch anchors to
        # the *smoothed* position, so it must already be settled by the time the
        # operator engages.
        command = self.mapper(pose, position)

        if self._homing:
            return self._step_homing(observation_before)

        if not self.mapper.engaged and not self.mapper.has_target:
            # Nothing has been commanded yet this episode. Hold the joints where
            # they are rather than solving towards a canonical orientation, which
            # would make the arm twitch on its own before the operator touches it.
            return self._step_idle(observation_before)
        # The approach pitch is dictated by the target, not by the operator; see
        # handrobot.retarget.reach for why.
        rotation = self.grasp.frame_for(command.position, command.jaw_azimuth)
        result = self.ik.solve(command.position, rotation, self._q)
        if not result.ok:
            # Re-solve from the home pose rather than a stale one, which matters
            # when the arm has to swing right across the workspace.
            retry = self.ik.solve(
                command.position, rotation, self.env._home_ctrl.copy(), iterations=60
            )
            result = retry if retry.position_error < result.position_error else result
        if not result.ok:
            self.stats.ik_failures += 1
        self._q = result.q.copy()
        self._q[self.gripper_index] = self.calibration.gap_to_command(command.jaw_gap)
        self._q = self.rate_limiter(self._q)

        action = self._q.copy()
        self._record(observation_before, action)

        # observe=False: the images for this step were already rendered above,
        # and the observation this returns is discarded.
        step_result = self.env.step(action, observe=False)
        if step_result.success and not self.last_success:
            self.last_success = True
            if self.recording:
                self.save_episode(success=True)
            else:
                self.message = "success"

        return {
            "command": command,
            "ik": result,
            "success": step_result.success,
            "alignment": self.alignment(),
        }

    def _step_idle(self, observation) -> dict:
        """Hold the current joint command; nothing is being demonstrated."""
        self.env.step(self._q, observe=False)
        return {"command": self.mapper.hold(self.env.gripper_pose[0]), "ik": None,
                "success": False, "alignment": self.alignment()}

    def _step_homing(self, observation) -> dict:
        """Move to the home pose at a limited speed; homing is not a demonstration."""
        self._q = self.rate_limiter(self.env._home_ctrl.copy())
        self.env.step(self._q, observe=False)
        # Only the arm joints: a gripper's measured joint value and its
        # commanded value are not in the same units on every robot.
        n = self.spec.n_arm_joints
        if np.max(np.abs(self.env.joint_positions[:n] - self.env._home_ctrl[:n])) < 0.03:
            self._homing = False
            self.message = "at home - press space to engage the clutch"
        return {"command": self.mapper.hold(self.env.gripper_pose[0]), "ik": None,
                "success": False, "alignment": self.alignment()}

    def _record(self, observation_before, action: np.ndarray) -> None:
        """Append one step to the current episode, if one is being recorded."""
        if not (self.recording and self.writer is not None):
            return
        if not self.mapper.engaged:
            # Frames where the operator has the clutch released are not part of
            # the demonstration; recording them would teach the policy to stop.
            return
        limit = self.config.sim.teleop_max_steps
        if self.episode_steps >= limit:
            self.recording = False
            self.message = f"episode hit the {limit}-step limit - press s to save or d to drop"
            return
        images = dict(observation_before.images)
        if not images:
            # Defensive: needs_images and the recording conditions must agree, or
            # an episode would be written with no camera data at all.
            raise RuntimeError("recording a step whose images were never rendered")
        if OPERATOR_CAMERA in self.writer.cameras:
            images[OPERATOR_CAMERA] = (
                self._operator_frame
                if self._operator_frame is not None
                else np.zeros((OPERATOR_HEIGHT, OPERATOR_WIDTH, 3), np.uint8)
            )
        self.writer.add(observation_before.joint_positions, action, images)
        self.episode_steps += 1

    def alignment(self) -> dict[str, float]:
        """Where the gripper is relative to the cube and the bin.

        Purely an operator aid, shown on screen. It is never a policy input:
        a policy that could read the cube's true position would learn to use it
        and would fall apart the moment it had to rely on its cameras.
        """
        gripper, _ = self.env.gripper_pose
        cube = self.env.cube_position
        bin_position = self.env.bin_position

        # Aim at the cube until it has been picked up, then at the bin.
        holding = cube[2] > 2.0 * self.spec.cube_half_extent
        goal = (
            np.array([bin_position[0], bin_position[1],
                      bin_position[2] + self.spec.release_clearance
                      + 2 * self.spec.cube_half_extent])
            if holding
            else np.array([cube[0], cube[1], self.grasp_height])
        )
        return {
            "cube_planar": float(np.linalg.norm(gripper[:2] - cube[:2])),
            "cube_height": float(gripper[2] - cube[2]),
            "bin_planar": float(np.linalg.norm(gripper[:2] - bin_position[:2])),
            "holding": holding,
            "goal_name": "bin" if holding else "cube",
            "goal": goal,
            "goal_error": float(np.linalg.norm(gripper - goal)),
            "hand_move": self.mapper.hand_move_for(goal - gripper),
        }

    def engage(self, pose: HandPose | None) -> None:
        if pose is None:
            self.message = "no hand visible - cannot engage"
            return
        self._homing = False
        position, _ = self.env.gripper_pose
        self.mapper.engage(pose, position)
        self.message = "clutch engaged"

    def disengage(self) -> None:
        self.mapper.disengage()
        self.message = "clutch released"


#: Interface sizes, all 16:9. The composed frame is what everything is drawn
#: into; the window showing it can be any size, because OpenCV scales it. A
#: larger frame does not mean a larger window -- it means text and overlays that
#: stay sharp when the window is large, and a recording that is worth keeping.
UI_PRESETS: dict[str, tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
    "8k": (7680, 4320),
}

#: Size of the composed interface when nothing else is asked for.
WINDOW_WIDTH, WINDOW_HEIGHT = UI_PRESETS["720p"]

#: Seconds a success banner takes to fade out.
FLASH_SECONDS = 1.6

#: The order ``v`` cycles the lower simulator panel through.
SIM_VIEWS = ("chase_cam", "front_cam", "hero_cam", "wrist_cam")


def window_size(ui: str | int | None) -> tuple[int, int]:
    """Resolve a size for the composed interface.

    Accepts a preset name, a height in pixels, or nothing. A bare height is
    completed to 16:9 and rounded to an even number of pixels, because half a
    pixel of panel is a row of grey down the middle of the window.
    """
    if ui is None:
        return UI_PRESETS["720p"]
    if isinstance(ui, str):
        key = ui.strip().lower()
        if key in UI_PRESETS:
            return UI_PRESETS[key]
        if key.isdigit():
            ui = int(key)
        else:
            raise ValueError(
                f"unknown interface size {ui!r}; choose from "
                f"{', '.join(UI_PRESETS)} or give a height in pixels"
            )
    height = int(ui)
    if height < 360:
        raise ValueError(f"interface height {height} is too small to read; 360 is the minimum")
    height -= height % 2
    width = int(round(height * 16 / 9))
    return width - width % 2, height


class RenderPacer:
    """Decides how often the simulator panels are redrawn.

    The control loop is paced by the webcam, and everything in one period --
    detection, inverse kinematics, physics, three rendered views, compositing --
    has to fit inside it. Rendering is the only part that can be spent
    selectively, and it is also the only part the operator does not feel
    directly: a panel held for one extra frame looks the same, while a control
    period that overruns shows up as an arm that lags the hand.

    So the cadence is measured, not assumed. A fast machine redraws every frame;
    a slow one degrades a view at a time instead of dropping control frames. The
    hysteresis and the settle count exist so the cadence cannot oscillate, which
    is far more visible than either cadence on its own.
    """

    def __init__(self, budget_ms: float = 1000.0 / 30.0, minimum: int = 1,
                 maximum: int = 4, settle: int = 15) -> None:
        self.budget_ms = float(budget_ms)
        self.minimum = int(minimum)
        self.maximum = int(maximum)
        self.settle = int(settle)
        self.cadence = self.minimum
        self._average: float | None = None
        self._since_change = 0

    @property
    def average_ms(self) -> float:
        return float(self._average) if self._average is not None else 0.0

    def update(self, loop_ms: float) -> int:
        """Fold in one measured control period and return the cadence to use."""
        loop_ms = float(loop_ms)
        self._average = loop_ms if self._average is None else 0.85 * self._average + 0.15 * loop_ms
        self._since_change += 1
        if self._since_change < self.settle:
            return self.cadence
        if self._average > self.budget_ms * 1.08 and self.cadence < self.maximum:
            self.cadence += 1
            self._since_change = 0
        elif self._average < self.budget_ms * 0.72 and self.cadence > self.minimum:
            self.cadence -= 1
            self._since_change = 0
        return self.cadence


def render_size(width: int, height: int, limit: tuple[int, int]) -> tuple[int, int]:
    """Largest render of this shape that fits the offscreen buffer.

    Scaled rather than cropped: a panel that fits by being cut off is a panel
    showing a different part of the scene than the one the overlays were
    projected for.
    """
    limit_h, limit_w = limit
    factor = min(1.0, limit_w / max(width, 1), limit_h / max(height, 1))
    return max(1, int(width * factor)), max(1, int(height * factor))


class Interface:
    """Everything the operator sees, assembled from one camera frame.

    One stage and a column of tiles. Every tool that watches several cameras at
    once -- a drone controller, a robot tablet, a streaming desk -- lays them
    out this way, because attention is not divisible: two half-sized views are
    not twice as useful as one, they are two things nobody is looking at
    properly. The stage is whatever the operator is working from, the tiles are
    there to be glanced at, and ``v`` swaps them.

    The overlays follow the same rule. Everything drawn on the stage has to earn
    its lines: the reach outline appears when the clutch is engaged and matters,
    the frame border appears when the hand is actually near it, the distance
    word appears when the distance is wrong. A picture covered in permanent
    guides is a picture nobody reads.

    Split from the loop deliberately: the loop owns the two things that cannot
    run without hardware -- a webcam and a window -- and this owns the pixels,
    so the whole interface can be driven from synthetic poses in a test.
    """

    #: What the stage can show, in the order ``v`` cycles them.
    STAGES = ("camera",) + SIM_VIEWS
    #: The tiles, in the order they are stacked. Whichever is on the stage is
    #: dropped from the column rather than shown twice.
    TILES = ("camera", "top_cam", "chase_cam", "wrist_cam")

    def __init__(
        self,
        config: Config,
        env: PickPlaceEnv,
        mapper: HandToGripper,
        camera_size: tuple[int, int] = (640, 480),
        ui: str | int | None = None,
        sim_view: str = "chase_cam",
    ) -> None:
        from handrobot.viz.roi import ReachEnvelope

        self.config = config
        self.env = env
        self.mapper = mapper
        self.width, self.height = window_size(ui)
        self.scale = max(0.25, self.height / REFERENCE_HEIGHT)

        self.stage = "camera"
        self.preferred_view = sim_view if sim_view in SIM_VIEWS else "chase_cam"
        self.show_help = False
        #: The tile column can be dropped entirely, which is the operator asking
        #: for every pixel in the window on the one view they are working from.
        self.show_tiles = True

        self.layout = panel_geometry(self.width, self.height, self.scale, tiles=3)
        limit = env.max_render_size
        # Panels are rendered at the size they are drawn at, clamped to the
        # model's offscreen buffer -- MuJoCo refuses to build a renderer larger
        # than it, so an unclamped 4K stage is not a slow interface but an
        # exception on the first frame.
        self.stage_render = render_size(
            self.layout["stage_width"], self.layout["stage_height"], limit
        )
        self.full_render = render_size(self.width, self.height, limit)
        self.tile_render = render_size(
            self.layout["column_width"], self.layout["tile_height"], limit
        )

        camera_width, camera_height = camera_size
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)

        self.pacer = RenderPacer(budget_ms=1000.0 / config.sim.control_hz)
        self.envelope = ReachEnvelope(
            workspace=config.workspace,
            camera_to_robot=HandToGripper.CAMERA_TO_ROBOT,
            intrinsics=None,
        )
        self._rendered: dict[str, np.ndarray] = {}
        self._frame_index = 0
        self._view_stamps: list[float] = []

    # -- what is on screen --------------------------------------------------

    @property
    def lower_view(self) -> str:
        """The simulator view the operator is working from."""
        return self.stage if self.stage in SIM_VIEWS else self.preferred_view

    @property
    def tile_names(self) -> list[str]:
        if not self.show_tiles:
            return []
        return [name for name in self.TILES if name != self.stage]

    @property
    def stage_size(self) -> tuple[int, int]:
        """Pixels the stage is rendered at, which depends on the column being there."""
        return self.stage_render if self.show_tiles else self.full_render

    def preview_size(self) -> tuple[int, int]:
        """Pixels the webcam preview is drawn at.

        Never more than twice the webcam's own resolution: past that the
        upscale invents nothing and only costs the memory bandwidth the control
        loop needs.
        """
        if self.stage == "camera":
            box = self.layout["stage_width"] if self.show_tiles else self.width
        else:
            box = self.layout["column_width"]
        width = min(box, 2 * self.camera_width)
        return width, max(1, round(width * self.camera_height / max(self.camera_width, 1)))

    def handle_key(self, key: int, session: "TeleopSession") -> bool:
        """Consume a key that only changes what is displayed. True if it was ours."""
        if key == ord("v"):
            self.stage = self.STAGES[(self.STAGES.index(self.stage) + 1) % len(self.STAGES)]
            if self.stage in SIM_VIEWS:
                self.preferred_view = self.stage
            self._rendered.clear()
            session.message = (
                "stage: your hand" if self.stage == "camera"
                else f"stage: {view_label(self.stage).lower()}"
            )
            return True
        if key == ord("w"):
            # The wrist is a tile like any other; hiding it is cycling past it.
            self.stage = "wrist_cam" if self.stage != "wrist_cam" else "camera"
            self._rendered.clear()
            session.message = f"stage: {'wrist view' if self.stage == 'wrist_cam' else 'your hand'}"
            return True
        if key == ord("t"):
            self.show_tiles = not self.show_tiles
            self._rendered.clear()
            session.message = "tiles on" if self.show_tiles else "tiles off - full window"
            return True
        if key in (ord("?"), ord("/")):
            self.show_help = not self.show_help
            return True
        return False

    # -- pixels -------------------------------------------------------------

    def ribbon_inset(self, preview_height: int) -> int:
        """Rows of the preview the status ribbon will end up lying over.

        The ribbon is drawn on the composed frame, after this; anything put in
        those rows is drawn where the operator cannot see it.
        """
        stage_height = max(1, self.layout["stage_height"])
        return int(round(self.layout["ribbon_height"] * preview_height / stage_height))

    def _preview(self, frame_rgb: np.ndarray, landmarks, pose, clipped: bool,
                 intrinsics, command, on_stage: bool, hand_move) -> np.ndarray:
        import cv2

        from handrobot.viz.roi import draw_envelope, draw_frame_margin
        from handrobot.viz.hud import draw_guidance_arrow

        width, height = self.preview_size()
        preview = cv2.cvtColor(
            cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_RGB2BGR,
        )
        unit = max(1.0, width / 640.0)
        inset = self.ribbon_inset(height) if on_stage else 0

        # Only when the hand is actually at the border it warns about.
        if clipped:
            draw_frame_margin(preview, clipped=True)

        if self.mapper.engaged and intrinsics is not None:
            self.envelope.intrinsics = intrinsics
            anchor_robot = self.mapper.robot_anchor
            if anchor_robot is None:
                anchor_robot = (
                    command.position if command is not None else self.env.gripper_pose[0]
                )
            plane_x = float(anchor_robot[0] if command is None else command.position[0])
            polygons = self.envelope.polygons(
                self.mapper.hand_anchor, anchor_robot, self.mapper.position_gain, plane_x
            )
            if polygons:
                zoom = width / max(self.camera_width, 1)
                draw_envelope(preview, [p * zoom for p in polygons],
                              saturated=getattr(self.mapper, "saturated", False),
                              bottom_inset=inset)

        draw_hand_overlay(preview, landmarks, pose, engaged=self.mapper.engaged)

        if pose is not None:
            low, high = self.config.hand.depth_comfort
            if not low <= pose.depth <= high:
                word = "MOVE BACK FROM THE CAMERA" if pose.depth < low else "COME CLOSER"
                cv2.putText(preview, word, (round(16 * unit), round(30 * unit)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6 * unit, (70, 190, 250),
                            max(1, round(unit)), cv2.LINE_AA)

        # The correction, drawn at the hand rather than in the corner.
        if on_stage and self.mapper.engaged and pose is not None and landmarks is not None:
            pinch = landmarks.image[[THUMB_TIP, INDEX_TIP], :2].mean(axis=0)
            draw_guidance_arrow(
                preview, (pinch[0] * width, pinch[1] * height),
                None if hand_move is None else hand_move * 1000, unit,
                bottom_inset=inset,
            )
        return preview

    def _refresh(self, now: float, needed: list[str]) -> None:
        import cv2

        for name in needed:
            width, height = (self.stage_size if name == self.stage else self.tile_render)
            self._rendered[name] = cv2.cvtColor(
                self.env.render(name, height, width), cv2.COLOR_RGB2BGR
            )
        self._view_stamps.append(now)
        if len(self._view_stamps) > 30:
            self._view_stamps.pop(0)

    @property
    def view_hz(self) -> float | None:
        if len(self._view_stamps) < 2:
            return None
        span = self._view_stamps[-1] - self._view_stamps[0]
        return (len(self._view_stamps) - 1) / span if span > 0 else None

    def render(self, session: "TeleopSession", frame_rgb: np.ndarray, pose, landmarks,
               info: dict, tracker, elapsed: float, now: float | None = None) -> np.ndarray:
        """Compose one interface frame from the state of one control period."""
        now = time.perf_counter() if now is None else now
        alignment = info.get("alignment") or {}
        command = info.get("command")
        clipped = landmarks is not None and hand_is_clipped(landmarks)

        preview = self._preview(
            frame_rgb, landmarks, pose, clipped, getattr(tracker, "intrinsics", None),
            command, self.stage == "camera", alignment.get("hand_move"),
        )

        cameras = [name for name in ([self.stage] + self.tile_names) if name != "camera"]
        if not self._rendered or self._frame_index % self.pacer.cadence == 0:
            self._refresh(now, cameras)
        self._frame_index += 1

        tcp = self.env.gripper_pose[0]
        goal = alignment.get("goal")
        drawn = {
            name: draw_scene_overlays(self._rendered[name], self.env, name, tcp, goal,
                                      show_arrow=(name == self.stage))
            for name in cameras
        }

        stage = preview if self.stage == "camera" else drawn[self.stage]
        stage_label = ("YOUR HAND" if self.stage == "camera" else view_label(self.stage))
        tiles = [
            (preview if name == "camera" else drawn[name],
             "YOUR HAND" if name == "camera" else view_label(name))
            for name in self.tile_names
        ]

        state = HudState(
            engaged=self.mapper.engaged,
            recording=session.recording,
            episode_steps=session.episode_steps,
            saved=session.stats.episodes_saved,
            successes=session.stats.successes,
            tracking=session.stats.live_tracking_rate,
            fps=session.stats.fps,
            sensitivity=self.mapper.position_gain,
            message=session.visible_message,
            goal_name=alignment.get("goal_name", "cube"),
            goal_distance=alignment.get("goal_error"),
            hand_move=alignment.get("hand_move"),
            saturated=getattr(self.mapper, "saturated", False),
            rejection=session.stats.live_rejection,
            holding=alignment.get("holding", False),
            hands_seen=getattr(tracker, "hands_seen", 0),
            followed_hand=getattr(tracker, "followed_hand", None),
            tracked_now=pose is not None,
            step_limit=self.config.sim.teleop_max_steps,
            flash=session.flash,
            loop_ms=self.pacer.average_ms,
            loop_budget_ms=self.pacer.budget_ms,
            tracking_history=tuple(session.stats.recent_quality),
            jaw_gap=None if command is None else float(command.jaw_gap),
            jaw_range=(self.config.gripper.min_command_gap,
                       self.config.gripper.max_command_gap),
            # Measured off the loaded model, not off the robot description: the
            # scene is the authority on how wide the puck actually is, and the
            # gauge is only useful if its tick is the real object.
            object_width=2.0 * self.env.cube_half_extent,
        )
        frame = compose(stage, tiles, state, width=self.width, height=self.height,
                        stage_label=stage_label, help_open=self.show_help)
        self.pacer.update(elapsed * 1000.0)
        return frame


def run_teleop(
    output: Path | str | None,
    device: int = 0,
    seed: int = 0,
    world_z_sign: float = 1.0,
    config: Config | None = None,
    sim_view: str = "chase_cam",
    stereo_device: int | None = None,
    stereo_baseline: float = 0.12,
    ui: str | int | None = None,
) -> TeleopStats:
    """Open the camera, open the simulator, and loop until the operator quits."""
    import cv2

    config = config or Config()
    # Resolve the interface size before anything is opened: a typo in --ui
    # should not cost the operator a camera permission prompt and a model load.
    frame_width, frame_height = window_size(ui)

    env = PickPlaceEnv(config=config, seed=seed)
    ik = ArmIK(config.ik, config.spec)
    mapper = HandToGripper(config.hand, config.workspace, config.gripper)

    writer = None
    policy_cameras = [c.name for c in config.sim.policy_cameras]
    if output is not None:
        writer = EpisodeWriter(
            output,
            cameras=policy_cameras + [OPERATOR_CAMERA],
            source="human",
            policy_cameras=policy_cameras,
        )

    session = TeleopSession(env, mapper, ik, config, writer)
    episode_seed = seed
    last_time = time.perf_counter()

    stereo = None
    if stereo_device is not None:
        from handrobot.hands.stereo import StereoRig

        stereo = StereoRig(stereo_device, stereo_baseline, config.hand)
        print(f"stereo depth on: second camera {stereo_device}, "
              f"baseline {stereo_baseline * 1000:.0f} mm")

    window = "handrobot teleop"
    try:
        with Webcam(device) as camera, HandTracker(
            camera.width, camera.height, config.hand, world_z_sign
        ) as tracker:
            interface = Interface(
                config, env, mapper, (camera.width, camera.height), ui=ui, sim_view=sim_view
            )
            # A resizable window, because the composed frame can be far larger
            # than the screen: the interface is drawn at full resolution and the
            # window shows as much of it as the display can.
            cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(window, *UI_PRESETS["720p"])
            if (frame_width, frame_height) != UI_PRESETS["720p"]:
                print(f"interface composed at {frame_width}x{frame_height}; "
                      "drag the window to any size")

            session.new_episode(seed=episode_seed)
            while True:
                frame = camera.read()
                if frame is None:
                    break
                now = time.perf_counter()
                elapsed = now - last_time
                session.stats.tick(elapsed)
                last_time = now
                session.stats.frames_seen += 1
                session.flash = max(0.0, session.flash - elapsed / FLASH_SECONDS)

                pose, landmarks = tracker.detect(frame)
                if stereo is not None:
                    pose = stereo.refine(pose, camera.width)
                clipped = landmarks is not None and hand_is_clipped(landmarks)
                session.stats.note_frame(pose is not None, tracker.last_rejection, clipped)
                if clipped:
                    session.stats.frames_clipped += 1

                # The operator panel of the film records the overlay alone, at a
                # modest size; the on-screen preview is composed separately.
                operator_rgb = cv2.resize(frame, (OPERATOR_WIDTH, OPERATOR_HEIGHT))
                operator_bgr = cv2.cvtColor(operator_rgb, cv2.COLOR_RGB2BGR)
                draw_hand_overlay(operator_bgr, landmarks, pose, engaged=mapper.engaged)
                session.set_operator_frame(cv2.cvtColor(operator_bgr, cv2.COLOR_BGR2RGB))

                info = session.step(pose)
                goal = (info.get("alignment") or {}).get("goal")
                env.set_goal_marker(goal if mapper.engaged else None)

                cv2.imshow(
                    window,
                    interface.render(session, frame, pose, landmarks, info, tracker,
                                     elapsed, now=now),
                )

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if interface.handle_key(key, session):
                    continue
                if key == ord(" "):
                    if mapper.engaged:
                        session.disengage()
                        # Next engage may be with the other hand; re-choose then.
                        tracker.forget_hand()
                    else:
                        session.engage(pose)
                elif key == ord("n"):
                    episode_seed += 1
                    session.new_episode(seed=episode_seed)
                elif key == ord("s"):
                    session.save_episode(success=session.last_success)
                elif key == ord("d"):
                    session.discard_episode()
                    episode_seed += 1
                    session.new_episode(seed=episode_seed)
                elif key in (ord("["), ord("]")):
                    gain = mapper.adjust_gain(-0.2 if key == ord("[") else 0.2)
                    session.message = f"sensitivity {gain:.1f}x"
                elif key == ord("h"):
                    session.go_home()
    finally:
        if stereo is not None:
            stereo.close()
        try:
            import cv2 as _cv2

            _cv2.destroyAllWindows()
        except Exception:
            pass
        env.close()

    return session.stats
