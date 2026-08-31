# Quick start

Five minutes to a robot copying your hand. One hour to a robot trained by you.

## 0. Requirements

- macOS or Linux, Python **3.12** (MediaPipe has no newer wheels)
- A webcam. Nothing else.

## 1. Install

```bash
make setup          # venv + deps + tracker model, prints all-ok
```

<sub>No `uv`? `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt -e . && ./scripts/fetch_models.sh`</sub>

## 2. Point the camera

```bash
make check          # move your hand, press q; aim for 90%+ tracked
```

macOS asks for camera permission on first run: System Settings > Privacy & Security > Camera.

## 3. Drive the robot

```bash
make teleop
```

| key | action |
|---|---|
| `space` | clutch: arm follows your hand only while engaged |
| `n` / `s` / `d` | new episode / save / discard |
| `[` `]` | sensitivity down / up |
| `q` | quit |

Pinch to close the gripper. Puck into the blue bin = episode saved.
Record 40-60 successes (about ten minutes).

## 4. Train and score

```bash
make train          # ACT on your demos, ~25 min on an M-series Mac
make eval           # success rate on 50 layouts it has never seen
```

## 5. Showpieces

```bash
make demo           # the policy solving the task alone
make film           # side-by-side video: you vs. the robot alone
make dexhand        # a 16-joint hand mirrors your fingers live
make dexhand-record # 1 min of capture, then it trains on YOUR hand
```

## No camera? Prove everything anyway

```bash
make baseline       # scripted demos -> training -> 100% eval, fully reproducible
make test           # 394 tests, headless, ~2 min
```

Troubleshooting and the full story: [README.md](README.md).
