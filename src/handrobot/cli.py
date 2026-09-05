"""Command line entry point: ``python -m handrobot <command>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from handrobot.config import Config, TrainConfig
from handrobot.paths import CHECKPOINT_DIR, DATA_DIR, OUTPUT_DIR


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0, help="base random seed")


def _add_robot(parser: argparse.ArgumentParser) -> None:
    from handrobot.robots import DEFAULT_ROBOT, ROBOTS

    parser.add_argument(
        "--robot", type=str, default=DEFAULT_ROBOT, choices=sorted(ROBOTS),
        help="which arm to drive; the Panda reaches far more of the table and "
             "handles a larger object, the SO-101 is the one you can buy",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="handrobot",
        description="Teach a simulated SO-101 arm with a webcam and your bare hands.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="print environment, device and asset status")

    p = sub.add_parser("handcheck", help="webcam diagnostic: verify hand tracking and depth")
    p.add_argument("--device", type=int, default=0, help="camera index")
    p.add_argument("--hfov", type=float, default=None,
                   help="camera horizontal field of view in degrees; only the "
                        "forward/back axis depends on it, so raise it if that "
                        "axis feels too sensitive and lower it if too sluggish")
    p.add_argument("--seconds", type=float, default=0.0,
                   help="exit automatically after this long (0 = run until q)")

    p = sub.add_parser("teleop", help="drive the arm with your hand and record demonstrations")
    p.add_argument("--out", type=Path, default=None,
                   help="dataset directory to append episodes to "
                        "(default: runs/demos/<robot>_human)")
    p.add_argument("--device", type=int, default=0, help="camera index")
    p.add_argument("--no-record", action="store_true", help="drive without saving anything")
    p.add_argument(
        "--ui", type=str, default="720p",
        help="interface size: 720p, 1080p, 1440p, 4k, 8k, or a height in pixels. "
             "The window is resizable whatever this is; a larger frame keeps the "
             "text and overlays sharp when the window is large.",
    )
    p.add_argument("--view", type=str, default="chase_cam",
                   choices=["chase_cam", "front_cam", "hero_cam", "wrist_cam"],
                   help="lower simulator panel (the top view is always shown); "
                        "cycle it with v")
    p.add_argument("--flip-z", action="store_true",
                   help="invert the hand depth axis (see handcheck)")
    p.add_argument("--hand", type=str, default=None, choices=["left", "right"],
                   help="which of your hands drives the arm; without this it "
                        "follows whichever hand you raise first")
    p.add_argument("--hfov", type=float, default=None,
                   help="camera horizontal field of view in degrees; only the "
                        "forward/back axis depends on it, so raise it if that "
                        "axis feels too sensitive and lower it if too sluggish")
    p.add_argument("--stereo-device", type=int, default=None,
                   help="second camera index; turns hand depth from a size "
                        "guess into stereo triangulation")
    p.add_argument("--baseline", type=float, default=0.12,
                   help="distance between the two cameras in metres "
                        "(measure it with a ruler)")
    _add_robot(p)
    _add_common(p)

    p = sub.add_parser("collect-scripted",
                       help="generate demonstrations with the scripted expert")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--out", type=Path, default=None,
                   help="default: runs/demos/<robot>_scripted")
    p.add_argument("--keep-failures", action="store_true",
                   help="also store episodes that did not succeed")
    p.add_argument("--task", type=str, default="bin",
                   help="objective: a task name (bin, push, lift, touch) or any "
                        "of its natural phrasings, e.g. 'push it to the ring'")
    _add_robot(p)
    _add_common(p)

    p = sub.add_parser("train", help="train ACT on a demonstration dataset")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=CHECKPOINT_DIR / "act")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default=None, help="mps, cuda or cpu")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--model", type=str, default="act", choices=["act", "diffusion"],
                   help="policy class: ACT (default) or a diffusion policy, "
                        "for the ablation")
    _add_common(p)

    p = sub.add_parser("eval", help="measure a controller's success rate")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="ACT checkpoint; omit to evaluate the scripted expert")
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--open-loop", action="store_true",
                   help="replay whole chunks instead of temporal ensembling")
    p.add_argument("--out", type=Path, default=None, help="write results as JSON")
    p.add_argument("--task", type=str, default="bin",
                   help="objective: a task name (bin, push, lift, touch) or any "
                        "of its natural phrasings, e.g. 'push it to the ring'")
    _add_robot(p)
    _add_common(p)

    p = sub.add_parser(
        "diagnose",
        help="work out why a policy is failing: grasp accuracy, camera use, motion",
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--device", type=str, default=None)
    _add_robot(p)

    p = sub.add_parser("demo", help="render a video of a controller solving the task")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--out", type=Path, default=OUTPUT_DIR / "demo.mp4")
    p.add_argument("--camera", type=str, default="hero_cam")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--task", type=str, default="bin",
                   help="objective: a task name (bin, push, lift, touch) or any "
                        "of its natural phrasings, e.g. 'push it to the ring'")
    _add_robot(p)
    _add_common(p)

    p = sub.add_parser(
        "film",
        help="build the three-panel film: your hands, the robot copying, the robot alone",
    )
    p.add_argument("--data", type=Path, required=True, help="dataset holding the episode")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="ACT checkpoint for the third panel; omit for two panels")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR / "film.mp4")
    p.add_argument("--device", type=str, default=None)
    _add_robot(p)

    p = sub.add_parser("replay", help="render an episode from a dataset")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--out", type=Path, default=OUTPUT_DIR / "replay.mp4")

    p = sub.add_parser(
        "dexhand",
        help="neural retargeting: your fingers drive a 16-joint LEAP hand live",
    )
    p.add_argument("--device", type=int, default=0, help="camera index")
    p.add_argument("--hand", type=str, default="right", choices=["left", "right"])
    p.add_argument("--retrain", action="store_true",
                   help="retrain the base network before starting")
    p.add_argument("--record", action="store_true",
                   help="first record YOUR hand for a minute and train on it; "
                        "this is what makes the mirroring truly yours")

    p = sub.add_parser("dataset", help="summarise a dataset")
    p.add_argument("--data", type=Path, required=True)

    p = sub.add_parser(
        "export-lerobot",
        help="convert a dataset to the Hugging Face LeRobot format",
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="directory for the LeRobot dataset")
    p.add_argument("--repo-id", type=str, required=True,
                   help="Hub-style id, e.g. yourname/handrobot-panda")
    p.add_argument("--keep-failures", action="store_true")
    _add_robot(p)

    return parser


# -- commands ---------------------------------------------------------------


def command_info(args: argparse.Namespace) -> int:
    import mujoco
    import numpy as np
    import torch

    from handrobot import __version__
    from handrobot.paths import HAND_LANDMARKER_TASK
    from handrobot.robots import DEFAULT_ROBOT, ROBOTS

    print(f"handrobot {__version__}")
    print(f"  mujoco       {mujoco.__version__}")
    print(f"  torch        {torch.__version__}  mps={torch.backends.mps.is_available()}")
    try:
        import mediapipe

        print(f"  mediapipe    {mediapipe.__version__}")
    except Exception as exc:
        print(f"  mediapipe    NOT AVAILABLE ({exc})")
    print(f"  hand model   {'ok  ' if HAND_LANDMARKER_TASK.exists() else 'MISSING'} "
          f"{HAND_LANDMARKER_TASK}")
    print()
    for name, spec in sorted(ROBOTS.items()):
        mark = "*" if name == DEFAULT_ROBOT else " "
        ok = spec.arm_xml.exists() and spec.scene_xml.exists()
        size = np.round(spec.workspace.size, 2)
        print(f" {mark}{name:8s} {'ok  ' if ok else 'MISSING'}  "
              f"{spec.n_arm_joints} joints   "
              f"workspace {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m   "
              f"object {spec.cube_half_extent * 2000:.0f} mm")
    print("\n  * default. Choose with --robot.")
    return 0


def command_handcheck(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from handrobot.config import HandConfig
    from handrobot.diagnostics import run_handcheck

    config = HandConfig()
    if args.hfov is not None:
        config = replace(config, assumed_hfov_deg=args.hfov)
    return run_handcheck(device=args.device, seconds=args.seconds, config=config)


def command_teleop(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from handrobot.teleop import run_teleop

    config = Config(robot=args.robot)
    overrides = {}
    if args.hfov is not None:
        overrides["assumed_hfov_deg"] = args.hfov
    if args.hand is not None:
        overrides["prefer_hand"] = args.hand
    if overrides:
        config = Config(
            robot=config.robot, sim=config.sim, ik=config.ik,
            hand=replace(config.hand, **overrides), train=config.train,
        )

    out = args.out if args.out is not None else DATA_DIR / f"{args.robot}_human"
    stats = run_teleop(
        output=None if args.no_record else out,
        device=args.device,
        seed=args.seed,
        world_z_sign=-1.0 if args.flip_z else 1.0,
        config=config,
        sim_view=args.view,
        stereo_device=args.stereo_device,
        stereo_baseline=args.baseline,
        ui=args.ui,
    )
    print(
        f"\nsaved {stats.episodes_saved} episodes "
        f"({stats.successes} successful, {stats.episodes_discarded} discarded)"
    )
    if not args.no_record and stats.episodes_saved:
        print(f"episodes are in {out}")
        print(f"next: .venv/bin/python -m handrobot train --data {out} "
              f"--out runs/checkpoints/{args.robot}_mine")
    print(f"hand tracked on {100 * stats.tracking_rate:.0f}% of frames, "
          f"{stats.ik_failures} unreachable targets")
    if stats.frames_seen:
        clipped = 100 * stats.frames_clipped / stats.frames_seen
        if clipped > 15:
            print(f"your hand touched the edge of the frame on {clipped:.0f}% of frames - "
                  "sit back so the whole hand stays inside the picture")
    top = stats.top_rejection
    if top is not None and stats.tracking_rate < 0.85:
        print(f"most frames were lost to: {top[0]} ({top[1]} frames)")
    return 0


def command_collect_scripted(args: argparse.Namespace) -> int:
    from handrobot.data.dataset import EpisodeWriter
    from handrobot.rollout import ScriptedController, run_episode
    from handrobot.scripted import ScriptedExpert
    from handrobot.sim import PickPlaceEnv

    config = Config(robot=args.robot)
    cameras = [c.name for c in config.sim.policy_cameras]
    out = args.out if args.out is not None else DATA_DIR / f"{args.robot}_scripted"
    env = PickPlaceEnv(config=config, seed=args.seed, task=args.task)
    controller = ScriptedController(ScriptedExpert(config))
    writer = EpisodeWriter(out, cameras, source="scripted")

    kept = successes = 0
    for i in range(args.episodes):
        result = run_episode(env, controller, writer=writer, seed=args.seed + i)
        successes += int(result.success)
        if result.success or args.keep_failures:
            writer.finish(success=result.success, metadata={
                "seed": args.seed + i,
                "task": env.task.name,
                "instruction": env.task.instruction,
            })
            kept += 1
        else:
            writer.discard()
        print(f"  episode {i + 1:3d}/{args.episodes}  "
              f"{'success' if result.success else 'FAIL'}  {result.steps:3d} steps")
    env.close()

    print(f"\n{successes}/{args.episodes} succeeded, {kept} episodes written to {out}")
    return 0


def command_train(args: argparse.Namespace) -> int:
    from handrobot.policy.train import train

    config = TrainConfig()
    if args.steps is not None:
        config.steps = args.steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.chunk_size is not None:
        config.chunk_size = args.chunk_size
    if args.lr is not None:
        config.learning_rate = args.lr
    config.seed = args.seed
    config.augment = not args.no_augment

    summary = train(args.data, args.out, config, device_name=args.device, resume=args.resume, model=args.model)
    print(f"\ntrained {summary['steps']} steps on {summary['episodes']} episodes "
          f"in {summary['wall_seconds'] / 60:.1f} min")
    print(f"best validation L1: {summary['best_val_l1']}")
    print(f"checkpoints in {args.out}")
    return 0


def _build_controller(args: argparse.Namespace, config: Config):
    from handrobot.rollout import PolicyController, ScriptedController

    if args.checkpoint is None:
        from handrobot.scripted import ScriptedExpert

        return ScriptedController(ScriptedExpert(config)), "scripted expert"

    from handrobot.policy.inference import ChunkedActor, load_checkpoint
    from handrobot.policy.train import resolve_device

    device = resolve_device(getattr(args, "device", None))
    policy, stats, extra = load_checkpoint(args.checkpoint, device)
    actor = ChunkedActor(
        policy,
        stats,
        device,
        image_size=config.train.image_size,
        temporal_ensemble_coeff=(
            None if getattr(args, "open_loop", False) else config.train.temporal_ensemble_coeff
        ),
        query_every=policy.config.chunk_size // 2,
    )
    label = f"ACT checkpoint {args.checkpoint} (step {extra.get('step', '?')}) on {device}"
    return PolicyController(actor), label


def command_eval(args: argparse.Namespace) -> int:
    from handrobot.rollout import evaluate_controller
    from handrobot.sim import PickPlaceEnv

    config = Config(robot=args.robot)
    controller, label = _build_controller(args, config)
    print(f"evaluating {label} on the {config.spec.name} over {args.episodes} episodes\n")

    env = PickPlaceEnv(config=config, seed=args.seed, task=args.task)
    results = evaluate_controller(env, controller, args.episodes, seed=args.seed)
    env.close()

    print(f"\nsuccess rate: {results['successes']}/{results['episodes']} "
          f"= {100 * results['success_rate']:.1f}%")
    print(f"mean episode length: {results['mean_steps']:.1f} steps")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"controller": label, **results}, indent=2))
        print(f"wrote {args.out}")
    return 0


def command_diagnose(args: argparse.Namespace) -> int:
    from handrobot.policy.diagnose import GRASP_TOLERANCE, VISION_THRESHOLD, diagnose

    print(f"diagnosing {args.checkpoint}\n")
    report = diagnose(args.checkpoint, episodes=args.episodes, device_name=args.device,
                      config=Config(robot=args.robot))

    print("  grasp error       "
          f"{report.mean_grasp_error * 1000:6.1f} mm   "
          f"(needs under {GRASP_TOLERANCE * 1000:.0f} mm)   "
          f"{'ok' if report.grasp_is_accurate else 'TOO HIGH'}")
    print("    per target      "
          + "  ".join(f"{e * 1000:.1f}" for e in report.grasp_errors) + " mm")
    print("  camera use        "
          f"{report.vision_sensitivity:6.2f} rad  "
          f"(needs over {VISION_THRESHOLD:.2f})       "
          f"{'ok' if report.uses_vision else 'IGNORING THE CAMERAS'}")
    print("  motion range      "
          f"{report.action_range:6.2f} rad  "
          f"{'ok' if report.moves else 'THE ARM IS FROZEN'}")
    print(f"  success rate      {100 * report.success_rate:6.1f} %    "
          f"over {report.episodes} episodes")

    print()
    for line in report.verdict():
        print(f"  - {line}")
    return 0


def command_demo(args: argparse.Namespace) -> int:
    import imageio.v2 as imageio

    from handrobot.rollout import run_episode
    from handrobot.sim import PickPlaceEnv

    config = Config(robot=args.robot)
    controller, label = _build_controller(args, config)
    env = PickPlaceEnv(config=config, seed=args.seed, task=args.task)

    frames: list[np.ndarray] = []
    successes = 0
    for i in range(args.episodes):
        result = run_episode(
            env, controller, seed=args.seed + i, render_camera=args.camera,
            render_size=(config.sim.render_height, config.sim.render_width),
        )
        successes += int(result.success)
        if not result.frames:
            print(f"  episode {i + 1}/{args.episodes}: produced no frames, skipping")
            continue
        frames.extend(result.frames)
        # Hold the final frame briefly so each attempt reads as a separate take.
        frames.extend([result.frames[-1]] * 12)
        print(f"  episode {i + 1}/{args.episodes}: "
              f"{'success' if result.success else 'FAIL'}")
    env.close()

    if not frames:
        print("no frames were rendered")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=int(config.sim.control_hz), quality=8,
                     macro_block_size=1)
    print(f"\n{label}: {successes}/{args.episodes} succeeded")
    print(f"wrote {args.out} ({len(frames)} frames)")
    return 0


def command_film(args: argparse.Namespace) -> int:
    from handrobot.viz.film import build_film

    result = build_film(
        data_root=args.data,
        episode_index=args.episode,
        checkpoint=args.checkpoint,
        output=args.out,
        config=Config(robot=args.robot),
        device_name=args.device,
    )
    print(f"panels: {' | '.join(result['panels'])}")
    print(f"demonstration succeeded: {result['episode_success']}")
    if result["policy_success"] is not None:
        print(f"policy succeeded:        {result['policy_success']}")
    print(f"wrote {result['output']} ({result['frames']} frames)")
    return 0


def command_replay(args: argparse.Namespace) -> int:
    import imageio.v2 as imageio

    from handrobot.data.dataset import list_episodes, load_episode

    paths = list_episodes(args.data)
    if not paths:
        print(f"no episodes in {args.data}")
        return 1
    if not 0 <= args.episode < len(paths):
        print(f"episode {args.episode} out of range (0..{len(paths) - 1})")
        return 1

    episode = load_episode(paths[args.episode])
    cameras = sorted(episode.policy_cameras or episode.images)
    frames = [np.hstack([episode.images[c][t] for c in cameras]) for t in range(len(episode))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=30, quality=8, macro_block_size=1)
    print(f"episode {args.episode}: {len(episode)} steps, success={episode.success}, "
          f"source={episode.source}")
    print(f"wrote {args.out}")
    return 0


def command_dexhand(args: argparse.Namespace) -> int:
    if args.retrain:
        from handrobot.dexhand.retarget_net import CHECKPOINT, train_retargeter

        train_retargeter(path=CHECKPOINT)
    if args.record:
        from handrobot.dexhand.record import record_hand
        from handrobot.dexhand.retarget_net import train_personal

        print("[1/3] recording your hand -- follow the prompts on screen; "
              "the clock pauses whenever your hand is not tracked")
        result = record_hand(device=args.device, hand=args.hand)
        if result["poses"] < 200:
            print("too few poses captured; keep your hand in frame and try again")
            return 1
        print(f"\n[2/3] training on your {result['poses']} poses "
              "(about half a minute, progress below)...")
        try:
            trained = train_personal()
        except ValueError as error:
            print(f"not installed: {error}")
            return 1
        print(f"personal network installed: your fingertips now match to "
              f"{trained['holdout_tips_mm']:.1f} mm "
              f"(base network was {trained['base_tips_mm']:.1f} mm)")
        print("\n[3/3] starting the live mirror...")
    from handrobot.dexhand.live import run_dexhand

    return run_dexhand(device=args.device, hand=args.hand)


def command_export_lerobot(args: argparse.Namespace) -> int:
    from handrobot.data.lerobot_export import export_lerobot

    try:
        export_lerobot(args.data, args.out, args.repo_id,
                       robot_type=args.robot,
                       successful_only=not args.keep_failures)
    except ImportError as error:
        print(error)
        return 1
    return 0


def command_dataset(args: argparse.Namespace) -> int:
    from handrobot.data.dataset import read_meta

    meta = read_meta(args.data)
    print(f"{args.data}")
    print(f"  episodes    {meta['num_episodes']} ({meta['num_successful']} successful)")
    print(f"  frames      {meta['num_frames']}")
    print(f"  cameras     {meta['cameras']}")
    if meta.get("policy_cameras") and meta["policy_cameras"] != meta["cameras"]:
        print(f"  policy      {meta['policy_cameras']}")
    by_source: dict[str, int] = {}
    for entry in meta["episodes"]:
        by_source[entry["source"]] = by_source.get(entry["source"], 0) + 1
    for source, count in sorted(by_source.items()):
        print(f"  source      {source}: {count}")
    if meta["episodes"]:
        lengths = [e["length"] for e in meta["episodes"]]
        print(f"  length      min {min(lengths)}  mean {sum(lengths) / len(lengths):.0f}  "
              f"max {max(lengths)}")
    return 0


COMMANDS = {
    "info": command_info,
    "handcheck": command_handcheck,
    "teleop": command_teleop,
    "collect-scripted": command_collect_scripted,
    "train": command_train,
    "eval": command_eval,
    "diagnose": command_diagnose,
    "demo": command_demo,
    "film": command_film,
    "replay": command_replay,
    "dexhand": command_dexhand,
    "dataset": command_dataset,
    "export-lerobot": command_export_lerobot,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
