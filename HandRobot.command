#!/bin/zsh
# handrobot launcher - double-click to drive the robot with your hand.
# Lives in the repository root and finds everything relative to itself,
# so it works from any clone location.
cd "$(dirname "$0")" || exit 1

if [ ! -x .venv/bin/python ]; then
  echo "no .venv found - run 'make setup' once first"
  read "anykey?  press ENTER to close: "
  exit 1
fi

clear
echo "  ┌─────────────────────────────────────────────┐"
echo "  │            h a n d r o b o t                │"
echo "  │   your hand, a webcam, a robot that learns  │"
echo "  ├─────────────────────────────────────────────┤"
echo "  │   [1] Teleop     drive the arm (default)    │"
echo "  │   [2] Demo       the trained policy, alone  │"
echo "  │   [3] Dexhand    16-joint finger mirror     │"
echo "  │   [4] Film       render the showcase video  │"
echo "  └─────────────────────────────────────────────┘"
echo ""
read "choice?  press ENTER for teleop, or 1-4: "

case "$choice" in
  2) .venv/bin/python -m handrobot demo ;;
  3) .venv/bin/python -m handrobot dexhand ;;
  4) .venv/bin/python -m handrobot film --data runs/demos/panda_human --episode 0 ;;
  *) .venv/bin/python -m handrobot teleop ;;
esac

echo ""
read "done?  finished - press ENTER to close: "
