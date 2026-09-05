"""On-screen overlays and video composition."""

from handrobot.viz.film import Panel, build_film, compose, replay_episode
from handrobot.viz.hud import HudState, draw_strip
from handrobot.viz.overlay import draw_hand_overlay
from handrobot.viz.roi import draw_depth_band, draw_envelope

__all__ = [
    "draw_hand_overlay",
    "draw_envelope",
    "draw_depth_band",
    "draw_strip",
    "HudState",
    "Panel",
    "compose",
    "replay_episode",
    "build_film",
]
