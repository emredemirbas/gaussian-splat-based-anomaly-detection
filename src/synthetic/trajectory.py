"""Orbital camera trajectory around the chimney in COLMAP world frame (Y-down).

Produces world-to-camera rotations and translations along with the orbit
centers so the writer can emit a COLMAP-compatible images.bin.
"""

from __future__ import annotations

import numpy as np


def _rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Inverse of `math_utils.qvec2rotmat`: 3x3 rotation -> (qw,qx,qy,qz).

    Uses the standard branch-on-trace algorithm. Returns a unit quaternion
    with qw >= 0.
    """
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def orbital_trajectory(
    n_frames: int,
    radius: float,
    H: float,
    height_amp: float = 1.5,
    n_loops: int = 1,
    pitch_down: float = 0.0,
) -> dict:
    """Build a smooth orbital trajectory around the chimney axis.

    Returns a dict with keys:
        centers     : (N, 3) camera centers in world frame (COLMAP)
        rotations   : (N, 3, 3) world-to-camera rotations R
        translations: (N, 3) world-to-camera translations t = -R @ C
        qvecs       : (N, 4) (qw, qx, qy, qz)
    """
    phi = 2.0 * np.pi * np.arange(n_frames) / n_frames * n_loops
    cx = radius * np.cos(phi)
    cz = radius * np.sin(phi)
    # Camera at mid-height of chimney with a vertical sinusoidal oscillation.
    cy = -(0.5 * H) + height_amp * np.sin(2.0 * phi)
    centers = np.column_stack([cx, cy, cz]).astype(np.float64)

    rotations = np.empty((n_frames, 3, 3), dtype=np.float64)
    translations = np.empty((n_frames, 3), dtype=np.float64)
    qvecs = np.empty((n_frames, 4), dtype=np.float64)

    world_down = np.array([0.0, 1.0, 0.0])  # +Y is down in COLMAP frame

    for i in range(n_frames):
        C = centers[i]
        L = np.array([0.0, C[1] + pitch_down, 0.0])  # look at axis point at same height
        forward = L - C
        forward /= np.linalg.norm(forward) + 1e-12

        # right = down x forward, then re-orthogonalize down.
        right = np.cross(world_down, forward)
        rn = np.linalg.norm(right)
        if rn < 1e-9:
            # Degenerate (camera directly above/below axis); fall back on x-axis.
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= rn
        down = np.cross(forward, right)
        down /= np.linalg.norm(down) + 1e-12

        # OpenCV/COLMAP camera frame: x-right, y-down, z-forward (rows of R_w2c).
        R_w2c = np.stack([right, down, forward], axis=0)
        t_w2c = -R_w2c @ C

        rotations[i] = R_w2c
        translations[i] = t_w2c
        qvecs[i] = _rotmat_to_qvec(R_w2c)

        # Sanity check: C = -R^T t.
        C_back = -R_w2c.T @ t_w2c
        assert np.linalg.norm(C_back - C) < 1e-9, "trajectory self-test failed"

    return {
        "centers": centers,
        "rotations": rotations,
        "translations": translations,
        "qvecs": qvecs,
    }
