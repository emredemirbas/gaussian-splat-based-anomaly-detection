"""Build PLY structured arrays for the synthetic Gaussian-splat + sparse output.

Field layout matches the 3DGS standard and is compatible with both the
anomaly detection pipeline (main.py) and the gsplat_viewer:

    position : x, y, z                (float32)
    normals  : nx, ny, nz             (float32)
    color DC : f_dc_0, f_dc_1, f_dc_2 (float32, SH DC coefficients)
    color SH : f_rest_0..f_rest_44    (float32, higher-order SH — zeros for synthetic)
    opacity  : opacity                (float32, logit space)
    scale    : scale_0..scale_2       (float32, log space)
    rotation : rot_0..rot_3           (float32, quaternion)

The actual file writing is delegated to `io_utils.PlyIO.save`.
"""

from __future__ import annotations

import os
import sys
import numpy as np

# Make repo root importable when this module is loaded as `synthetic.ply_writer`.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from io_utils import PlyIO  # noqa: E402


# --------------------------------------------------------------------------- #
#  Build the full 3DGS-standard dtype (62 fields, matches real data)           #
# --------------------------------------------------------------------------- #

_SH_REST_COUNT = 45  # 3 * ((max_sh_degree + 1)^2 - 1)  for degree=3

_fields: list[tuple[str, str]] = [
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
    ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
]
_fields += [(f"f_rest_{i}", "<f4") for i in range(_SH_REST_COUNT)]
_fields += [
    ("opacity", "<f4"),
    ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
    ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
]

GSPLAT_DTYPE = np.dtype(_fields)

SPARSE_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])


def build_gsplat_array(
    positions: np.ndarray,    # (N, 3)
    f_dc: np.ndarray,         # (N, 3)
    opacity: np.ndarray,      # (N,) logit space
    log_scale: np.ndarray,    # (N, 3) log space
    normals: np.ndarray | None = None,  # (N, 3) optional surface normals
) -> np.ndarray:
    n = positions.shape[0]
    arr = np.zeros(n, dtype=GSPLAT_DTYPE)

    # Position
    arr["x"] = positions[:, 0]
    arr["y"] = positions[:, 1]
    arr["z"] = positions[:, 2]

    # Normals (optional — zero if not provided)
    if normals is not None:
        arr["nx"] = normals[:, 0]
        arr["ny"] = normals[:, 1]
        arr["nz"] = normals[:, 2]

    # SH DC color
    arr["f_dc_0"] = f_dc[:, 0]
    arr["f_dc_1"] = f_dc[:, 1]
    arr["f_dc_2"] = f_dc[:, 2]

    # Higher-order SH coefficients — all zero for synthetic data.
    # (np.zeros already handles this.)

    # Opacity (logit space)
    arr["opacity"] = opacity

    # Scale (log space)
    arr["scale_0"] = log_scale[:, 0]
    arr["scale_1"] = log_scale[:, 1]
    arr["scale_2"] = log_scale[:, 2]

    # Rotation — identity quaternion
    arr["rot_0"] = 1.0
    arr["rot_1"] = 0.0
    arr["rot_2"] = 0.0
    arr["rot_3"] = 0.0

    return arr


def build_sparse_array(positions: np.ndarray) -> np.ndarray:
    n = positions.shape[0]
    arr = np.zeros(n, dtype=SPARSE_DTYPE)
    arr["x"] = positions[:, 0]
    arr["y"] = positions[:, 1]
    arr["z"] = positions[:, 2]
    return arr


def save_ply(path: str, arr: np.ndarray) -> None:
    PlyIO.save(path, arr, as_text=False)
