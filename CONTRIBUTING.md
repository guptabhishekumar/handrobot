# Contributing

Thanks for the interest. The bar for this repository is simple: nothing lands
that is not measured, and nothing is claimed that is not tested.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
./scripts/fetch_models.sh
.venv/bin/pytest -q          # ~2 minutes, must be green before and after
```

Python 3.12 exactly (MediaPipe has no newer wheels). Everything runs headless:
the tests that need a webcam stub it with a photograph, so CI and laptops
without cameras run the full suite.

## Ground rules

- **Every behavioural change carries a test that fails without it.** The suite
  leans on end-to-end pins: screen directions are asserted against rendered
  pixels, retargeting against forward kinematics, capture against a real
  photograph through the real tracker. Prefer that style over mocks.
- **Tuned constants need a measurement.** Filter cutoffs, deadband radii, rate
  limits and workspace bounds in this codebase were each chosen from a swept
  measurement, and the docstring next to the value says what was measured. A PR
  that changes one should update that story.
- **No silent regressions in feel.** Anything touching `retarget/` or
  `teleop.py` should keep `tests/test_stability.py` and
  `tests/test_virtual_operator.py` green -- they are the difference between
  pleasant teleoperation and an arm that wanders.
- **Robot-agnostic or explicitly not.** Code reads the `RobotSpec`; a change
  that only works on one arm should say so and be tested on both anyway
  (`conftest.py` parameterises most fixtures over Panda and SO-101).

## Where help is genuinely wanted

- A third arm (the `RobotSpec` registry is the whole integration surface)
- Stereo or learned depth to replace the monocular estimate
- Language-conditioned or multi-task policies on top of the recorder
- Sim-to-real transfer for a physical SO-101

Open an issue first for anything large.
