"""Canonical filesystem locations for models, assets and generated data."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

ASSETS_DIR = PROJECT_ROOT / "assets"
SO101_DIR = ASSETS_DIR / "so101"

#: Arm only. Used for inverse kinematics, where free-floating scene objects
#: would otherwise show up as solvable degrees of freedom.
ARM_XML = SO101_DIR / "so101.xml"

#: Arm plus table, cube, bin and cameras. Used for simulation.
SCENE_XML = SO101_DIR / "scene_task.xml"

HAND_LANDMARKER_TASK = ASSETS_DIR / "models" / "hand_landmarker.task"

#: Everything a session produces lives under here, one subfolder per kind:
#:
#:     runs/
#:       demos/<name>/      episodes you recorded (teleop) or generated (scripted)
#:       checkpoints/<name>/  trained policies: best.pt, last.pt, training log
#:       results/           evaluation JSON, demo videos, films
#:
#: One folder to look in, one folder to back up, one folder to delete.
RUNS_DIR = PROJECT_ROOT / "runs"
DATA_DIR = RUNS_DIR / "demos"
CHECKPOINT_DIR = RUNS_DIR / "checkpoints"
OUTPUT_DIR = RUNS_DIR / "results"
