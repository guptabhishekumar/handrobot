"""Differentiable forward kinematics for the LEAP hand, in PyTorch.

Neural retargeting here is trained end to end: the network proposes joint
angles, this module turns them into fingertip and knuckle positions, and the
loss compares those against the human hand's landmarks. That only works if the
kinematics is differentiable -- MuJoCo's is not -- so the chain is rebuilt in
torch from the very MJCF the simulator uses.

Every joint on this hand is a hinge with axis (0, 0, -1) at its body origin,
which keeps the maths small: each link is a fixed rotation-plus-offset followed
by a rotation about z by minus the joint angle.

``tests/test_dexhand.py`` asserts this FK matches MuJoCo's to a micron at
random joint angles; if the model file ever changes, that fails before any
network trains on wrong geometry.
"""

from __future__ import annotations

import numpy as np
import torch

from handrobot.paths import ASSETS_DIR

LEAP_XML = ASSETS_DIR / "leap" / "right_hand.xml"

#: The joints, in MuJoCo order. Four per finger: knuckle flex, sideways
#: rotation, middle flex, tip flex -- and the thumb's own four.
JOINT_NAMES = (
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
)

#: Keypoints the retargeting matches, four per digit, base to tip.
KEYPOINT_NAMES = (
    "if_knuckle", "if_mid", "if_distal", "if_tip",
    "mf_knuckle", "mf_mid", "mf_distal", "mf_tip",
    "rf_knuckle", "rf_mid", "rf_distal", "rf_tip",
    "th_base", "th_mid", "th_distal", "th_tip",
)

#: Fingertip pad centres, in each distal body's own frame, read off the tip
#: geoms in the MJCF.
TIP_OFFSETS = {
    "if_ds": (-0.0013, -0.0336, 0.0145),
    "mf_ds": (-0.0013, -0.0336, 0.0145),
    "rf_ds": (-0.0013, -0.0336, 0.0145),
    "th_ds": (-0.0013, -0.0456, -0.0145),
}

#: Body whose origin marks each keypoint (the tip keypoints add their offset).
KEYPOINT_BODIES = (
    "if_bs", "if_md", "if_ds", "if_ds",
    "mf_bs", "mf_md", "mf_ds", "mf_ds",
    "rf_bs", "rf_md", "rf_ds", "rf_ds",
    "th_mp", "th_px", "th_ds", "th_ds",
)


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class LeapFK:
    """Batched, differentiable palm-frame FK to the sixteen keypoints."""

    def __init__(self, xml_path=LEAP_XML) -> None:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(xml_path))
        palm = model.body("palm").id

        self.joint_low = torch.tensor(model.jnt_range[:, 0], dtype=torch.float32)
        self.joint_high = torch.tensor(model.jnt_range[:, 1], dtype=torch.float32)

        # For every keypoint, the chain from the palm down: a list of
        # (fixed rotation, fixed offset, joint index or None) per link.
        self.chains: list[list[tuple[torch.Tensor, torch.Tensor, int | None]]] = []
        for body_name, keypoint in zip(KEYPOINT_BODIES, KEYPOINT_NAMES):
            body = model.body(body_name).id
            links = []
            while body != palm:
                rotation = torch.tensor(
                    _quat_to_matrix(model.body_quat[body]), dtype=torch.float32
                )
                offset = torch.tensor(model.body_pos[body], dtype=torch.float32)
                joint = None
                if model.body_jntnum[body] == 1:
                    joint_id = int(model.body_jntadr[body])
                    axis = model.jnt_axis[joint_id]
                    if not np.allclose(axis, (0, 0, -1)):
                        raise ValueError(f"unexpected joint axis {axis} on {body_name}")
                    joint = joint_id
                links.append((rotation, offset, joint))
                body = int(model.body_parentid[body])
            links.reverse()
            self.chains.append(links)

        self.tip_offset = {
            name: torch.tensor(value, dtype=torch.float32)
            for name, value in TIP_OFFSETS.items()
        }
        self.keypoint_tip_body = [
            KEYPOINT_BODIES[i] if KEYPOINT_NAMES[i].endswith("_tip") else None
            for i in range(len(KEYPOINT_NAMES))
        ]

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        """Joint angles (B, 16) to keypoints (B, 16, 3), in the palm frame."""
        batch = q.shape[0]
        cos, sin = torch.cos(-q), torch.sin(-q)  # axis is (0, 0, -1)

        keypoints = []
        for index, links in enumerate(self.chains):
            rotation = torch.eye(3, dtype=q.dtype, device=q.device).expand(batch, 3, 3)
            position = torch.zeros(batch, 3, dtype=q.dtype, device=q.device)
            for fixed_rotation, offset, joint in links:
                position = position + torch.einsum(
                    "bij,j->bi", rotation, offset.to(q.dtype)
                )
                rotation = rotation @ fixed_rotation.to(q.dtype)
                if joint is not None:
                    c, s = cos[:, joint], sin[:, joint]
                    zero = torch.zeros_like(c)
                    one = torch.ones_like(c)
                    spin = torch.stack([
                        torch.stack([c, -s, zero], dim=1),
                        torch.stack([s, c, zero], dim=1),
                        torch.stack([zero, zero, one], dim=1),
                    ], dim=1)
                    rotation = rotation @ spin
            tip_body = self.keypoint_tip_body[index]
            if tip_body is not None:
                position = position + torch.einsum(
                    "bij,j->bi", rotation, self.tip_offset[tip_body].to(q.dtype)
                )
            keypoints.append(position)
        return torch.stack(keypoints, dim=1)

    def clamp(self, q: torch.Tensor) -> torch.Tensor:
        return torch.clamp(q, self.joint_low.to(q.dtype), self.joint_high.to(q.dtype))
