#!/usr/bin/env bash
# Download the MediaPipe hand landmarker weights.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p assets/models
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
curl -sSL -o assets/models/hand_landmarker.task "$URL"
echo "saved assets/models/hand_landmarker.task ($(du -h assets/models/hand_landmarker.task | cut -f1))"
