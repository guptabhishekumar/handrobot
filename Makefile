# handrobot launchers. `make` with no target prints this menu.
PY := .venv/bin/python

.DEFAULT_GOAL := help
help:                ## show this menu
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[32m%-12s\033[0m %s\n", $$1, $$2}'

setup:               ## create venv, install, fetch tracker model
	uv venv --python 3.12 .venv
	uv pip install -e ".[dev]"
	./scripts/fetch_models.sh
	$(PY) -m handrobot info

warmup:              ## build the measured tables and the retargeting network
	$(PY) -m handrobot warmup

check:               ## webcam diagnostic: can it see your hand?
	$(PY) -m handrobot handcheck

teleop:              ## drive the arm with your hand, record demos
	$(PY) -m handrobot teleop

demo:                ## watch the trained policy solve the task alone
	$(PY) -m handrobot demo

film:                ## render the side-by-side showcase video
	$(PY) -m handrobot film

dexhand:             ## LEAP hand mirrors your fingers (neural retargeting)
	$(PY) -m handrobot dexhand

dexhand-record:      ## 1 minute of guided capture, then trains on YOUR hand
	$(PY) -m handrobot dexhand --record

train:               ## train ACT on your recorded demonstrations
	$(PY) -m handrobot train --data runs/demos/panda_human --out runs/checkpoints/mine

eval:                ## score a policy on 50 unseen layouts
	$(PY) -m handrobot eval --checkpoint runs/checkpoints/mine/best.pt --episodes 50

multitask:           ## collect all 4 tasks and train one conditioned policy
	for T in bin push lift touch; do \
	  $(PY) -m handrobot collect-scripted --task $$T --episodes 120 --out runs/demos/panda_multitask; done
	$(PY) -m handrobot train --data runs/demos/panda_multitask --out runs/checkpoints/multitask --steps 16000

baseline:            ## full reproducible pipeline without a camera (~60 min)
	./scripts/quickstart.sh

test:                ## run the full suite (~2 min, headless)
	$(PY) -m pytest -q

.PHONY: help setup warmup check teleop demo film dexhand dexhand-record train eval multitask baseline test
