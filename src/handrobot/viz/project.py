"""Projecting world points into a rendered camera image.

The interface draws guidance directly onto the simulator panels -- a crosshair
on the gripper, an arrow to the goal -- which requires knowing where a 3D point
lands in pixels. That is the standard pinhole model. MuJoCo gives the camera's
pose in ``cam_xpos``/``cam_xmat`` (columns are the camera's axes in world
coordinates: x right, y up, and it looks along -z) and its vertical field of
view in degrees, from which the focal length in pixels is

    f = (H / 2) / tan(fovy / 2)

and a point at ``v`` in camera coordinates lands at

    u = W/2 + f * v_x / -v_z          (right of centre)
    v = H/2 - f * v_y / -v_z          (below centre)

``tests/test_projection.py`` checks this against where a rendered probe
actually appears, so the maths here cannot silently drift from the renderer.
"""

from __future__ import annotations

import numpy as np


def project_point(env, camera: str, point: np.ndarray,
                  height: int, width: int) -> tuple[float, float] | None:
    """Pixel position of a world point in a camera's image, or ``None`` if behind."""
    import mujoco

    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if camera_id < 0:
        return None
    if camera == "chase_cam":
        env.update_chase_camera()
    mujoco.mj_camlight(env.model, env.data)

    position = env.data.cam_xpos[camera_id]
    rotation = env.data.cam_xmat[camera_id].reshape(3, 3)
    v = rotation.T @ (np.asarray(point, dtype=float) - position)
    if v[2] > -1e-6:
        return None  # behind the camera

    focal = 0.5 * height / np.tan(0.5 * np.radians(env.model.cam_fovy[camera_id]))
    u = 0.5 * width + focal * v[0] / -v[2]
    w = 0.5 * height - focal * v[1] / -v[2]
    return float(u), float(w)
