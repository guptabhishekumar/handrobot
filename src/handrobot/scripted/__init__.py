"""A scripted solver for the pick-and-place task.

It exists for three reasons: to prove the task is physically solvable before a
human records anything, to give the learned policy a baseline to be measured
against, and to let the whole data-to-training pipeline be tested end to end
without a camera.
"""

from handrobot.scripted.expert import ScriptedExpert, Waypoint

__all__ = ["ScriptedExpert", "Waypoint"]
