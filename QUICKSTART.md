# Quick start

From a fresh machine to a robot arm copying your hand, then doing the task alone.

```
clone ──▶ install ──▶ check camera ──▶ play ──▶ train ──▶ watch it solo
 30 s       3 min         1 min        10 min   25 min      2 min
```

---

## 0. What you need

| | |
|---|---|
| **OS** | macOS or Linux |
| **Python** | **3.12 only** — MediaPipe publishes no wheels for 3.13+. `uv` installs 3.12 for you, so you do not need it beforehand. |
| **Tools** | `git`, `curl` |
| **Hardware** | any webcam. No robot, no gloves, no GPU. |
| **Disk** | ~3 GB (torch + MuJoCo + MediaPipe) |

No webcam? Skip to [No camera](#no-camera-run-the-whole-thing-anyway) — everything still trains and scores.

---

## 1. Clone

```bash
git clone https://github.com/guptabhishekumar/handrobot.git
cd handrobot
```

Every command below is run from this folder.

## 2. Get `uv` (skip if you have it)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS + Linux
# or, on macOS:  brew install uv
```

Then reopen the terminal, or `source $HOME/.local/bin/env`.

## 3. Install

```bash
make setup
```

That one command:

1. creates `.venv` on Python 3.12 (downloading 3.12 if you lack it),
2. installs the package and dev deps,
3. downloads the MediaPipe hand-landmarker weights (~8 MB) into `assets/models/`,
4. prints a status report.

**It worked if the report ends like this** — versions will differ, the `ok`s must not:

```
handrobot 0.1.0
  mujoco       3.x.x
  torch        2.x.x  mps=True
  mediapipe    0.10.35
  hand model   ok    .../assets/models/hand_landmarker.task

 *panda    ok    7 joints   workspace ...
  so101    ok    5 joints   workspace ...
```

Re-run that report any time with `.venv/bin/python -m handrobot info`.

<details>
<summary><b>No <code>uv</code>, or you want plain pip</b> (click)</summary>

You need a real Python 3.12 first (`brew install python@3.12`, or deadsnakes on Ubuntu):

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ".[dev]"
./scripts/fetch_models.sh
.venv/bin/python -m handrobot info
```

</details>

> **No activation needed.** Every `make` target calls `.venv/bin/python` directly.
> If you prefer, `source .venv/bin/activate` and then use `python -m handrobot ...`.

## 4. Prove it works before touching the camera

```bash
make test                                        # 520 tests, headless, ~6 min
.venv/bin/python -m handrobot eval --episodes 25 # scripted expert: expect 100%
```

The scripted expert exists so the physics, the grasp and the scoring are known-good
*before* you spend ten minutes recording.

## 5. Point the camera

```bash
make check
```

Move your hand around, open and close a pinch, press `q`.

- **Aim for 90%+ tracked.** More light on your hand and a plainer background both help.
- If it says *"depth axis is inverted"*, add `--flip-z` to teleop in the next step.
- macOS asks for camera permission on first run. If no prompt appears, enable your
  terminal under **System Settings → Privacy & Security → Camera**.
- Wrong camera picked? `.venv/bin/python -m handrobot handcheck --device 1`

## 6. Play

```bash
make teleop
```

or double-click **`HandRobot.command`** in Finder for a menu.

One window opens: your hand on the left, the simulator on the right, a guidance
strip along the bottom.

**How to play**

1. Press `space` — the clutch engages and the arm starts following your hand.
   Press `space` again to freeze it while you move your hand back somewhere comfy.
   (Same as lifting a mouse off the desk. Use it constantly.)
2. Read the strip: `LEFT 35   DOWN 8   PUSH 22` — millimetres to the puck.
   Move until it says `LINED UP WITH THE CUBE`.
3. **Pinch** thumb to index — the gripper closes. Pause a beat before descending
   and again before closing; smoothing lag vanishes the moment you hold still.
4. Carry the puck to the blue bin. Success **saves the episode automatically**.
5. `n` for a fresh layout. Repeat.

`AT THE EDGE OF REACH` means you asked for somewhere the arm cannot go — move back
toward the middle and it resumes instantly.

**Keys**

| key | does |
|---|---|
| `space` | clutch on / off — arm only follows while engaged |
| `n` | new episode (resets scene, starts recording) |
| `s` | save the current episode by hand |
| `d` | discard it and reset |
| `h` | send the arm home |
| `[` `]` | less / more sensitive to your hand |
| `v` | put the next view on the big stage |
| `w` | put the wrist view on the stage |
| `t` | hide the tiles — the stage takes the whole window |
| `?` | show the key list on screen |
| `q` | quit |

**What you are looking at**

- **Green outline** on your camera — every hand position the arm can reach right
  now. Inside it the arm follows; outside it the arm stops. It moves when you
  re-clutch and shrinks when you raise sensitivity.
- **Gauge on the right** — how far your hand is from the camera. Stay in the
  green band; it says `COME CLOSER` or `MOVE BACK` when you drift out.
- **Thin border** — where hand tracking starts to fail. Turns red before it does.
- **Tiles** on the right — top, follow and wrist views. `v` swaps any of them
  onto the big stage; `t` hides them for a full-window view.
- **Hairline** along the very top — it marks only the frames tracking lost
  (red) or your hand touched the frame edge (amber). Clean means clean.
- **JAW gauge** in the ribbon — how far the jaws are open, with a tick at the
  puck's real width. It has to pass the tick to go around the puck.

Bigger screen? `.venv/bin/python -m handrobot teleop --ui 1080p` (also `1440p`,
`4k`, `8k`, or a height in pixels). The window stays resizable either way.

Set the sensitivity in the first thirty seconds: twitchy → `[`, having to reach
across the room → `]`. It never jerks the arm.

**Aim for 40–60 successes, about ten minutes.** They land in `runs/demos/panda_human/`.

## 7. Train on what you recorded

```bash
make train    # runs/demos/panda_human -> runs/checkpoints/mine, ~25 min on an M-series Mac
make eval     # success rate on 50 layouts it has never seen
```

Training prints an L1 action error; the number that matters is the eval success rate.

<details>
<summary><b>Score is low, or you are short of demos</b> (click)</summary>

```bash
# why is it failing? separates grasp misses / memorisation / diverged training
.venv/bin/python -m handrobot diagnose --checkpoint runs/checkpoints/mine/best.pt

# top the dataset up with scripted episodes (tagged 'scripted', same folder)
.venv/bin/python -m handrobot collect-scripted --episodes 150 --out runs/demos/panda_human
.venv/bin/python -m handrobot dataset --data runs/demos/panda_human   # see the mix
```

</details>

## 8. Watch it work

```bash
make demo     # the policy alone, no hand anywhere
make film     # three panels: your webcam | the robot copying you | the policy solo
```

Videos land in `runs/results/`.

---

## No camera? Run the whole thing anyway

```bash
make baseline                    # scripted demos -> train -> eval -> video (~60 min)
./scripts/quickstart.sh 40 3000  # same pipeline, fast smoke run (~12 min)
./scripts/quickstart.sh 150 12000 so101   # the other arm
```

## Extras

```bash
make dexhand          # 16-joint LEAP hand mirrors your fingers live
make dexhand-record   # 1 min of guided capture, then it retrains on YOUR hand
make multitask        # all 4 tasks (bin / push / lift / touch), one conditioned policy
```

`teleop`, `collect-scripted`, `eval`, `demo`, `film` and `diagnose` take `--robot so101`
for the ~£200 arm instead of the default Panda.
Run `make` with no target to list every launcher.

---

## When it breaks

| symptom | fix |
|---|---|
| `make: uv: command not found` | do step 2, reopen the terminal |
| `hand model MISSING` in `info` | `./scripts/fetch_models.sh` |
| `mediapipe NOT AVAILABLE` | wrong Python. It must be 3.12 — rebuild `.venv` |
| no camera prompt, black window | System Settings → Privacy & Security → Camera, tick your terminal |
| wrong camera used | add `--device 1` (2, 3, …) |
| tracking under 90% | more light on the hand, plainer background, hand fully in frame |
| *"depth axis is inverted"* | `.venv/bin/python -m handrobot teleop --flip-z` |
| arm too twitchy / too sluggish | `[` and `]` while running |
| `train` says no data | you recorded to a custom folder — pass `--data <that folder>` |
| Linux: `ImportError: libGL.so.1` | `sudo apt install -y libgl1 libglib2.0-0` |

Everything a run produces is under `runs/` — one folder to inspect, back up, or delete.

The full story, the numbers, and how it works: **[README.md](README.md)**.
