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
``v``      swap the lower simulator panel (chase, front, hero, wrist).
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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from handrobot.config import Config
from handrobot.data.dataset import EpisodeWriter
from handrobot.filters import RateLimiter
from handrobot.gripper import GripperCalibration
from handrobot.hands.tracker import HandTracker, Webcam
from handrobot.hands.types import HandPose
from handrobot.retarget.ik import ArmIK
from handrobot.retarget.mapper import HandToGripper
from handrobot.retarget.grasp import GraspFrames
from handrobot.sim.env import PickPlaceEnv
from handrobot.viz.hud import GAP, STRIP_HEIGHT, HudState, compose, draw_scene_overlays
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

    def note_rejection(self, reason: str | None) -> None:
        if reason:
            self.rejections[reason] = self.rejections.get(reason, 0) + 1

    @property
    def top_rejection(self) -> tuple[str, int] | None:
        if not self.rejections:
            return None
        return max(self.rejections.items(), key=lambda item: item[1])

    @property
    def tracking_rate(self) -> float:
        return self.frames_tracked / max(self.frames_seen, 1)

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
        self.message = "press n to start an episode"
        self._q = env.joint_positions.copy()
        self._seed: int | None = None
        self._operator_frame: np.ndarray | None = None
        self._homing = False

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
        self.message = (
            f"saved {path.name} ({'success' if success else 'failure'}) - press n for the next"
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


def run_teleop(
    output: Path | str | None,
    device: int = 0,
    seed: int = 0,
    world_z_sign: float = 1.0,
    config: Config | None = None,
    sim_view: str = "front_cam",
) -> TeleopStats:
    """Open the camera, open the simulator, and loop until the operator quits."""
    import cv2

    config = config or Config()
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
    # "top+front" is the default because neither view alone is enough: the top
    # view shows horizontal alignment with the cube, which a side view cannot,
    # and the front view shows height, which a top view cannot.
    # The top view is always shown -- horizontal alignment is invisible from
    # any side view. The lower panel defaults to the chase camera, which rides
    # above and behind the gripper so the arm can never park itself in front of
    # the lens; "v" swaps it for the fixed views.
    views = ["chase_cam", "front_cam", "hero_cam", "wrist_cam"]
    view_index = views.index(sim_view) if sim_view in views else 0
    #: Render the simulator panel every Nth frame and hold it in between. It is
    #: for the operator's eyes, not for the policy, and at 30 Hz a fresh render
    #: every frame would eat a third of the control budget.
    view_every = 3
    last_view: tuple[np.ndarray, np.ndarray] | None = None
    frame_index = 0

    episode_seed = seed
    last_time = time.perf_counter()

    try:
        with Webcam(device) as camera, HandTracker(
            camera.width, camera.height, config.hand, world_z_sign
        ) as tracker:
            session.new_episode(seed=episode_seed)
            while True:
                frame = camera.read()
                if frame is None:
                    break
                now = time.perf_counter()
                session.stats.tick(now - last_time)
                last_time = now
                session.stats.frames_seen += 1

                pose, landmarks = tracker.detect(frame)
                if pose is not None:
                    session.stats.frames_tracked += 1
                else:
                    session.stats.note_rejection(tracker.last_rejection)
                if landmarks is not None and hand_is_clipped(landmarks):
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

                preview = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                draw_hand_overlay(preview, landmarks, pose, engaged=mapper.engaged)
                if last_view is None or frame_index % view_every == 0:
                    panel_w, panel_h = _panel_geometry()
                    last_view = (
                        cv2.cvtColor(env.render("top_cam", panel_h, panel_w), cv2.COLOR_RGB2BGR),
                        cv2.cvtColor(
                            env.render(views[view_index], panel_h, panel_w), cv2.COLOR_RGB2BGR
                        ),
                    )
                frame_index += 1

                alignment = info.get("alignment") or {}
                tcp = env.gripper_pose[0]
                top_panel = draw_scene_overlays(last_view[0], env, "top_cam", tcp, goal)
                # The chase panel keeps only the crosshair: an arrow there sweeps
                # around as the camera follows the gripper, and the operator
                # found it distracting rather than helpful.
                low_panel = draw_scene_overlays(
                    last_view[1], env, views[view_index], tcp, goal, show_arrow=False
                )
                cv2.imshow(
                    "handrobot teleop",
                    compose(
                        preview, top_panel, low_panel,
                        HudState(
                            engaged=mapper.engaged,
                            recording=session.recording,
                            episode_steps=session.episode_steps,
                            saved=session.stats.episodes_saved,
                            successes=session.stats.successes,
                            tracking=session.stats.tracking_rate,
                            fps=session.stats.fps,
                            sensitivity=mapper.position_gain,
                            message=session.message,
                            goal_name=alignment.get("goal_name", "cube"),
                            goal_distance=alignment.get("goal_error"),
                            hand_move=alignment.get("hand_move"),
                            saturated=getattr(mapper, "saturated", False),
                            rejection=(session.stats.top_rejection or (None, 0))[0]
                            if session.stats.tracking_rate < 0.85 else None,
                            holding=alignment.get("holding", False),
                            hands_seen=tracker.hands_seen,
                            followed_hand=(
                                {"Left": "right", "Right": "left"}.get(
                                    tracker._followed.handedness
                                )
                                if tracker._followed is not None
                                else None
                            ),
                        ),
                    ),
                )

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
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
                elif key == ord("v"):
                    view_index = (view_index + 1) % len(views)
                    last_view = None
                    session.message = f"view: {views[view_index]}"
                elif key in (ord("["), ord("]")):
                    gain = mapper.adjust_gain(-0.2 if key == ord("[") else 0.2)
                    session.message = f"sensitivity {gain:.1f}x"
                elif key == ord("h"):
                    session.go_home()
    finally:
        try:
            import cv2 as _cv2

            _cv2.destroyAllWindows()
        except Exception:
            pass
        env.close()

    return session.stats


#: Size of the composed interface window.
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


def _panel_geometry(width: int = WINDOW_WIDTH, height: int = WINDOW_HEIGHT) -> tuple[int, int]:
    """Pixel size of each simulator panel, so renders match it exactly."""
    body = height - STRIP_HEIGHT
    camera_width = int(width * 0.52)
    return width - camera_width - GAP, (body - GAP) // 2
