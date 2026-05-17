"""Geometry sampling for the synthetic chimney scene.

COLMAP world frame conventions (must match `io_utils.get_colmap_camera_centers`
and `main.py`'s `GroundFilter(vertical_axis=1, axis_points_up=False)`):
  +Y points DOWN.
  Ground plane lies near y = 0.
  Chimney axis runs along -y: base at y = 0, apex at y = -H.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ChimneySamples:
    positions: np.ndarray   # (N, 3) float32 in COLMAP frame
    normals: np.ndarray     # (N, 3) float32 outward surface normals
    uv: np.ndarray          # (N, 2) float32 surface coords: (theta, h)
    is_band: np.ndarray     # (N,)   bool — point falls in a darker horizontal band


def r_of_h(h: np.ndarray, H: float, r_base: float, r_top: float) -> np.ndarray:
    """Radius at height h (h in [0, H]) for a tapered cylinder."""
    return r_base + (r_top - r_base) * (h / H)


def sample_tapered_cylinder(
    H: float,
    r_base: float,
    r_top: float,
    n_points: int,
    n_bands: int = 6,
    band_width: float = 0.30,
    band_bump: float = 0.02,
    rng: np.random.Generator | None = None,
) -> ChimneySamples:
    """Uniformly sample the lateral surface of a tapered cylinder.

    Returns positions in COLMAP frame: base at y=0, apex at y=-H.
    """
    rng = rng if rng is not None else np.random.default_rng()

    h = rng.uniform(0.0, H, size=n_points)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
    r = r_of_h(h, H, r_base, r_top)

    is_band = np.zeros(n_points, dtype=bool)
    if n_bands > 0:
        band_centers = (np.arange(n_bands) + 0.5) * (H / n_bands)
        for hc in band_centers:
            is_band |= np.abs(h - hc) < (band_width * 0.5)

    r_eff = r + np.where(is_band, band_bump, 0.0)

    x = r_eff * np.cos(theta)
    z = r_eff * np.sin(theta)
    y = -h

    positions = np.column_stack([x, y, z]).astype(np.float32)

    # Outward normal of a tapered cylinder surface. Side tilts inward by
    # slope = (r_top - r_base) / H as h grows; in COLMAP Y-down that means the
    # outward normal has a +y component (toward the wider base, which is at y=0).
    slope = (r_top - r_base) / H
    nx = np.cos(theta)
    nz = np.sin(theta)
    ny = np.full_like(nx, slope, dtype=np.float64)
    n = np.column_stack([nx, ny, nz])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    normals = n.astype(np.float32)

    uv = np.column_stack([theta, h]).astype(np.float32)
    return ChimneySamples(positions=positions, normals=normals, uv=uv, is_band=is_band)


def sample_ground_plane(
    extent: float,
    n_points: int,
    noise_sigma: float = 0.02,
    center_xz: tuple[float, float] = (0.0, 0.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a noisy ground plane around y=0 in COLMAP frame."""
    rng = rng if rng is not None else np.random.default_rng()
    half = 0.5 * extent
    x = rng.uniform(-half, half, size=n_points) + center_xz[0]
    z = rng.uniform(-half, half, size=n_points) + center_xz[1]
    y = rng.normal(0.0, noise_sigma, size=n_points)
    return np.column_stack([x, y, z]).astype(np.float32)


def sample_background_clutter(
    n_blobs: int,
    points_per_blob: int,
    chimney_radius_max: float,
    field_extent: float,
    height_range: tuple[float, float] = (-3.0, 0.0),
    blob_sigma: float = 0.6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Place Gaussian blobs (mock trees/buildings) outside the chimney."""
    rng = rng if rng is not None else np.random.default_rng()
    half = 0.5 * field_extent

    centers = []
    while len(centers) < n_blobs:
        cx = rng.uniform(-half, half)
        cz = rng.uniform(-half, half)
        if np.hypot(cx, cz) > chimney_radius_max + 2.0:
            cy = rng.uniform(height_range[0], height_range[1])
            centers.append((cx, cy, cz))

    pts = []
    for cx, cy, cz in centers:
        local = rng.normal(0.0, blob_sigma, size=(points_per_blob, 3))
        local[:, 1] *= 0.6  # flatter vertically
        pts.append(local + np.array([cx, cy, cz]))
    return np.concatenate(pts, axis=0).astype(np.float32)
