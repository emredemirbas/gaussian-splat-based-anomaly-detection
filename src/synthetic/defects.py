"""Defect generation for the synthetic chimney.

Defect types
------------
1. **Corrosion**      – elliptical rust-coloured patches (color change).
2. **Cracks**         – narrow dark linear features with elongated scale and
                        slight inward displacement.
3. **Missing bolts**  – small circular voids with low opacity, dark color,
                        and inward displacement.
4. **Deformed panels** – rectangular patches displaced outward/inward with
                         enlarged Gaussian scales.

The pipeline's Stage-3 RX detector reads `f_dc_0/1/2` (DC spherical-harmonic
coefficients), position, opacity, and scale.  These defect types are designed
so that each one creates anomalies in a *different* subset of those features,
enabling thorough end-to-end testing.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


# --------------------------------------------------------------------------- #
#  Constants                                                                   #
# --------------------------------------------------------------------------- #

SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))
RUST_RGB = np.array([0.55, 0.25, 0.10], dtype=np.float32)
CRACK_RGB = np.array([0.08, 0.06, 0.05], dtype=np.float32)  # near-black
BOLT_HOLE_RGB = np.array([0.05, 0.04, 0.04], dtype=np.float32)  # void dark


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB in [0, 1] to 3DGS SH DC coefficients."""
    return ((rgb.astype(np.float32) - 0.5) / SH_C0).astype(np.float32)


def _wrap_angle(d: np.ndarray) -> np.ndarray:
    """Wrap angle differences to (-pi, pi]."""
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def logit(p: float) -> float:
    """Logit function: log(p / (1 - p))."""
    return float(np.log(p / (1.0 - p)))


# =========================================================================== #
#  1.  CORROSION  (existing, unchanged)                                        #
# =========================================================================== #

@dataclass
class CorrosionPatch:
    theta_c: float    # center azimuth [0, 2*pi)
    h_c: float        # center height [0, H]
    a_theta: float    # arc-length extent (metres) along circumference
    a_h: float        # vertical extent (metres)
    strength: float   # blend weight at center, in (0, 1]

    def to_dict(self) -> dict:
        return {**asdict(self), "type": "corrosion"}


def generate_corrosion_patches(
    n_patches: int,
    H: float,
    r_mean: float,
    rng: np.random.Generator | None = None,
    strength_range: tuple[float, float] = (0.6, 0.95),
    a_theta_range: tuple[float, float] = (0.4, 1.2),
    a_h_range: tuple[float, float] = (0.5, 1.5),
    h_margin: float = 0.5,
) -> list[CorrosionPatch]:
    rng = rng if rng is not None else np.random.default_rng()
    patches: list[CorrosionPatch] = []
    for _ in range(n_patches):
        patches.append(CorrosionPatch(
            theta_c=float(rng.uniform(0.0, 2.0 * np.pi)),
            h_c=float(rng.uniform(h_margin, H - h_margin)),
            a_theta=float(rng.uniform(*a_theta_range)),
            a_h=float(rng.uniform(*a_h_range)),
            strength=float(rng.uniform(*strength_range)),
        ))
    return patches


def apply_corrosion(
    uv: np.ndarray,
    r_at_h: np.ndarray,
    base_rgb: np.ndarray,
    patches: list[CorrosionPatch],
) -> tuple[np.ndarray, np.ndarray]:
    """Blend base RGB toward rust where corrosion patches cover the surface.

    Returns (rgb, corroded_mask).
    """
    theta = uv[:, 0]
    h = uv[:, 1]

    total_w = np.zeros(theta.shape[0], dtype=np.float32)

    for p in patches:
        dtheta = _wrap_angle(theta - p.theta_c)
        s_arc = r_at_h * dtheta
        s_h = h - p.h_c
        d2 = (s_arc / p.a_theta) ** 2 + (s_h / p.a_h) ** 2
        w = (p.strength * np.exp(-d2)).astype(np.float32)
        total_w += w

    total_w = np.clip(total_w, 0.0, 1.0)
    rgb = (1.0 - total_w[:, None]) * base_rgb + total_w[:, None] * RUST_RGB[None, :]
    return rgb.astype(np.float32), total_w > 0.15


# =========================================================================== #
#  2.  CRACKS                                                                  #
# =========================================================================== #

@dataclass
class CrackDefect:
    """A crack is a linear feature on the surface from (theta_s, h_s)
    to (theta_e, h_e) with a given half-width (metres on surface)."""
    theta_s: float
    h_s: float
    theta_e: float
    h_e: float
    width: float       # half-width in metres on the surface
    depth: float       # inward radial displacement (metres)
    scale_elongation: float  # factor to elongate one scale axis (e.g. 4.0)

    def to_dict(self) -> dict:
        return {**asdict(self), "type": "crack"}


def generate_cracks(
    n_cracks: int,
    H: float,
    r_mean: float,
    rng: np.random.Generator | None = None,
    width_range: tuple[float, float] = (0.03, 0.08),
    depth_range: tuple[float, float] = (0.01, 0.03),
    length_range: tuple[float, float] = (1.0, 4.0),
    scale_elongation_range: tuple[float, float] = (3.0, 5.0),
    h_margin: float = 1.0,
) -> list[CrackDefect]:
    rng = rng if rng is not None else np.random.default_rng()
    cracks: list[CrackDefect] = []
    for _ in range(n_cracks):
        # Start point
        theta_s = float(rng.uniform(0.0, 2.0 * np.pi))
        h_s = float(rng.uniform(h_margin, H - h_margin))
        # Random direction and length
        length = float(rng.uniform(*length_range))
        angle = float(rng.uniform(-np.pi / 4, np.pi / 4))  # mostly vertical
        dh = length * np.cos(angle)
        dtheta = length * np.sin(angle) / r_mean  # convert metres to radians
        h_e = float(np.clip(h_s + dh, h_margin, H - h_margin))
        theta_e = theta_s + dtheta
        cracks.append(CrackDefect(
            theta_s=theta_s,
            h_s=h_s,
            theta_e=theta_e,
            h_e=h_e,
            width=float(rng.uniform(*width_range)),
            depth=float(rng.uniform(*depth_range)),
            scale_elongation=float(rng.uniform(*scale_elongation_range)),
        ))
    return cracks


def apply_cracks(
    uv: np.ndarray,
    r_at_h: np.ndarray,
    normals: np.ndarray,
    positions: np.ndarray,
    rgb: np.ndarray,
    log_scale: np.ndarray,
    cracks: list[CrackDefect],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply crack defects.  Modifies color, positions, and scale.

    Returns (positions, rgb, log_scale, crack_mask).
    """
    theta = uv[:, 0]
    h = uv[:, 1]
    n = len(theta)
    crack_mask = np.zeros(n, dtype=bool)

    positions = positions.copy()
    rgb = rgb.copy()
    log_scale = log_scale.copy()

    for cr in cracks:
        # Line segment in surface coordinates (arc-length, height).
        # For each point, compute distance to the line segment.
        # Convert angles to arc-length using local radius.
        p_arc = r_at_h * _wrap_angle(theta - cr.theta_s)
        p_h = h - cr.h_s
        seg_arc = r_at_h * _wrap_angle(np.full(n, cr.theta_e - cr.theta_s))
        seg_h = np.full(n, cr.h_e - cr.h_s)

        # Parameter t along the segment (clamped to [0, 1])
        seg_len2 = seg_arc**2 + seg_h**2
        t = np.clip(
            (p_arc * seg_arc + p_h * seg_h) / (seg_len2 + 1e-12),
            0.0, 1.0,
        )
        # Closest point on segment
        proj_arc = t * seg_arc
        proj_h = t * seg_h
        dist = np.sqrt((p_arc - proj_arc)**2 + (p_h - proj_h)**2)

        # Weight: 1 inside half-width, smooth falloff outside
        w = np.clip(1.0 - dist / (cr.width + 1e-12), 0.0, 1.0).astype(np.float32)
        affected = w > 0.05
        crack_mask |= affected

        if not np.any(affected):
            continue

        # --- Color: blend toward near-black ---
        rgb[affected] = (
            (1.0 - w[affected, None]) * rgb[affected]
            + w[affected, None] * CRACK_RGB[None, :]
        )

        # --- Position: displace inward along normal ---
        positions[affected] -= (
            w[affected, None] * cr.depth * normals[affected]
        )

        # --- Scale: elongate one axis (axis 1 = "vertical" in local frame) ---
        # Increase scale_1 (log space), shrink scale_0
        log_scale[affected, 1] += w[affected] * np.log(cr.scale_elongation)
        log_scale[affected, 0] -= w[affected] * np.log(2.0)  # shrink perpendicular

    return positions, rgb, log_scale, crack_mask


# =========================================================================== #
#  3.  MISSING BOLTS                                                           #
# =========================================================================== #

@dataclass
class MissingBoltDefect:
    """A circular void where a bolt was removed."""
    theta_c: float    # center azimuth
    h_c: float        # center height
    radius: float     # surface-distance radius (metres)
    opacity_drop: float  # how much to reduce opacity in logit space

    def to_dict(self) -> dict:
        return {**asdict(self), "type": "missing_bolt"}


def generate_missing_bolts(
    n_bolts: int,
    H: float,
    n_bands: int,
    rng: np.random.Generator | None = None,
    radius_range: tuple[float, float] = (0.05, 0.15),
    opacity_drop_range: tuple[float, float] = (2.0, 4.0),
    h_margin: float = 0.5,
) -> list[MissingBoltDefect]:
    """Generate missing bolt defects at band locations (structural joints)."""
    rng = rng if rng is not None else np.random.default_rng()
    bolts: list[MissingBoltDefect] = []

    # Place bolts at band positions (where structural joints would be)
    if n_bands > 0:
        band_centers = (np.arange(n_bands) + 0.5) * (H / n_bands)
    else:
        band_centers = np.array([H * 0.5])

    for _ in range(n_bolts):
        # Pick a random band and a random azimuth
        h_c = float(rng.choice(band_centers))
        theta_c = float(rng.uniform(0.0, 2.0 * np.pi))
        bolts.append(MissingBoltDefect(
            theta_c=theta_c,
            h_c=h_c,
            radius=float(rng.uniform(*radius_range)),
            opacity_drop=float(rng.uniform(*opacity_drop_range)),
        ))
    return bolts


def apply_missing_bolts(
    uv: np.ndarray,
    r_at_h: np.ndarray,
    normals: np.ndarray,
    positions: np.ndarray,
    rgb: np.ndarray,
    opacity: np.ndarray,
    bolts: list[MissingBoltDefect],
    inward_depth: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply missing bolt defects.  Modifies color, opacity, and positions.

    Returns (positions, rgb, opacity, bolt_mask).
    """
    theta = uv[:, 0]
    h = uv[:, 1]
    n = len(theta)
    bolt_mask = np.zeros(n, dtype=bool)

    positions = positions.copy()
    rgb = rgb.copy()
    opacity = opacity.copy()

    for b in bolts:
        dtheta = _wrap_angle(theta - b.theta_c)
        s_arc = r_at_h * dtheta
        s_h = h - b.h_c
        dist = np.sqrt(s_arc**2 + s_h**2)

        # Sharp circular boundary with soft edge
        w = np.clip(1.0 - dist / (b.radius + 1e-12), 0.0, 1.0).astype(np.float32)
        affected = w > 0.05
        bolt_mask |= affected

        if not np.any(affected):
            continue

        # --- Color: dark void ---
        rgb[affected] = (
            (1.0 - w[affected, None]) * rgb[affected]
            + w[affected, None] * BOLT_HOLE_RGB[None, :]
        )

        # --- Opacity: drop (in logit space) ---
        opacity[affected] -= w[affected] * b.opacity_drop

        # --- Position: displace inward (recessed hole) ---
        positions[affected] -= (
            w[affected, None] * inward_depth * normals[affected]
        )

    return positions, rgb, opacity, bolt_mask


# =========================================================================== #
#  4.  DEFORMED PANELS                                                         #
# =========================================================================== #

@dataclass
class DeformedPanelDefect:
    """A rectangular patch of displaced (dented or bulging) surface."""
    theta_c: float      # center azimuth
    h_c: float          # center height
    w_theta: float      # angular half-extent (radians)
    w_h: float          # vertical half-extent (metres)
    displacement: float  # radial displacement: + outward, − inward (metres)
    scale_factor: float  # scale enlargement factor for affected splats

    def to_dict(self) -> dict:
        return {**asdict(self), "type": "deformed_panel"}


def generate_deformed_panels(
    n_panels: int,
    H: float,
    rng: np.random.Generator | None = None,
    w_theta_range: tuple[float, float] = (0.15, 0.4),
    w_h_range: tuple[float, float] = (0.5, 1.5),
    displacement_range: tuple[float, float] = (-0.10, 0.10),
    scale_factor_range: tuple[float, float] = (1.5, 3.0),
    h_margin: float = 1.0,
) -> list[DeformedPanelDefect]:
    rng = rng if rng is not None else np.random.default_rng()
    panels: list[DeformedPanelDefect] = []
    for _ in range(n_panels):
        displacement = float(rng.uniform(*displacement_range))
        # Ensure displacement is non-trivial (avoid near-zero)
        if abs(displacement) < 0.03:
            displacement = 0.05 * (1.0 if displacement >= 0 else -1.0)
        panels.append(DeformedPanelDefect(
            theta_c=float(rng.uniform(0.0, 2.0 * np.pi)),
            h_c=float(rng.uniform(h_margin, H - h_margin)),
            w_theta=float(rng.uniform(*w_theta_range)),
            w_h=float(rng.uniform(*w_h_range)),
            displacement=displacement,
            scale_factor=float(rng.uniform(*scale_factor_range)),
        ))
    return panels


def apply_deformed_panels(
    uv: np.ndarray,
    normals: np.ndarray,
    positions: np.ndarray,
    log_scale: np.ndarray,
    panels: list[DeformedPanelDefect],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply deformed panel defects.  Modifies positions and scale.

    Returns (positions, log_scale, panel_mask).
    """
    theta = uv[:, 0]
    h = uv[:, 1]
    n = len(theta)
    panel_mask = np.zeros(n, dtype=bool)

    positions = positions.copy()
    log_scale = log_scale.copy()

    for p in panels:
        dtheta = _wrap_angle(theta - p.theta_c)
        # Normalized distance in patch coordinates
        u = dtheta / (p.w_theta + 1e-12)
        v = (h - p.h_c) / (p.w_h + 1e-12)
        d2 = u**2 + v**2

        # Smooth Gaussian falloff
        w = np.exp(-0.5 * d2).astype(np.float32)
        # Only affect points well within the patch
        w[d2 > 9.0] = 0.0  # cut off beyond 3-sigma
        affected = w > 0.05
        panel_mask |= affected

        if not np.any(affected):
            continue

        # --- Position: displace along outward normal ---
        positions[affected] += (
            w[affected, None] * p.displacement * normals[affected]
        )

        # --- Scale: enlarge all axes ---
        log_scale[affected] += w[affected, None] * np.log(p.scale_factor)

    return positions, log_scale, panel_mask
