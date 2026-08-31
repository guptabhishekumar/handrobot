#!/usr/bin/env bash
# Everything except the webcam: verify the install, prove the task is solvable,
# generate demonstrations, train, score the result, and render a video.
#
#   ./scripts/quickstart.sh                  # 150 episodes, 12k steps (~60 min)
#   ./scripts/quickstart.sh 40 3000          # a fast smoke run        (~12 min)
#   ./scripts/quickstart.sh 150 12000 so101  # the other arm
set -euo pipefail
cd "$(dirname "$0")/.."

EPISODES="${1:-150}"
STEPS="${2:-12000}"
ROBOT="${3:-panda}"
PY=".venv/bin/python"

echo "==> environment"
$PY -m handrobot info

echo; echo "==> scripted expert, ${EPISODES} episodes"
$PY -m handrobot eval --robot "$ROBOT" --episodes 20

echo; echo "==> collecting demonstrations"
$PY -m handrobot collect-scripted --robot "$ROBOT" --episodes "$EPISODES" --out "runs/demos/${ROBOT}_scripted"
$PY -m handrobot dataset --data "runs/demos/${ROBOT}_scripted"

echo; echo "==> training ACT for ${STEPS} steps"
$PY -m handrobot train --data "runs/demos/${ROBOT}_scripted" --out "runs/checkpoints/$ROBOT" --steps "$STEPS"

echo; echo "==> scoring the policy on unseen layouts"
$PY -m handrobot eval --robot "$ROBOT" --checkpoint "runs/checkpoints/$ROBOT/best.pt" \
    --episodes 50 --seed 900000 --out "runs/results/${ROBOT}_eval.json"

echo; echo "==> rendering"
$PY -m handrobot demo --robot "$ROBOT" --checkpoint "runs/checkpoints/$ROBOT/best.pt" \
    --episodes 4 --seed 900000 --out "runs/results/${ROBOT}_demo.mp4"
$PY -m handrobot film --robot "$ROBOT" --data "runs/demos/${ROBOT}_scripted" --episode 0 \
    --checkpoint "runs/checkpoints/$ROBOT/best.pt" --out runs/results/film.mp4

echo; echo "done. everything is in runs/"
