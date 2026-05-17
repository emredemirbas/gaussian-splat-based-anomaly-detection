"""CLI: generate a synthetic chimney scene with multiple defect types.

Defect types:
    1. Corrosion patches  – rust-coloured elliptical regions (color change)
    2. Cracks             – narrow dark linear features (color + scale + position)
    3. Missing bolts      – small circular voids (opacity + color + position)
    4. Deformed panels    – dented/bulging surface patches (position + scale)

Outputs (in --out-dir):
    images.bin            COLMAP-compatible trajectory (consumed by io_utils)
    sparse.ply            sparse points used by Stage 1 filters
    gsplat.ply            full Gaussian-splat point cloud used by Stage 2/3
    ground_truth_labels.npy   per-chimney-point defect label (0=normal, 1-4=defect)
    meta.json             seed + ground-truth defect parameters

Example:
    python -m synthetic.generate --out-dir data/synthetic/synth01 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from synthetic import geometry, defects, trajectory, colmap_writer, ply_writer
from io_utils import get_colmap_camera_centers  # for round-trip verification


# ---- defaults shared by chimney body ---------------------------------------- #

CHIMNEY_BASE_RGB = np.array([0.78, 0.78, 0.76], dtype=np.float32)  # weathered concrete
BAND_RGB = np.array([0.50, 0.50, 0.48], dtype=np.float32)          # darker bands
GROUND_RGB = np.array([0.35, 0.30, 0.22], dtype=np.float32)        # dirt/grass
CLUTTER_RGB = np.array([0.20, 0.35, 0.18], dtype=np.float32)       # vegetation green


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate synthetic chimney + multi-defect data for the inspection pipeline."
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_background", action="store_true", help="Skip generating ground and clutter")

    # Chimney
    ap.add_argument("--height", type=float, default=30.0)
    ap.add_argument("--r-base", type=float, default=2.5)
    ap.add_argument("--r-top", type=float, default=1.5)
    ap.add_argument("--n-bands", type=int, default=6)

    # Point counts
    ap.add_argument("--n-chimney", type=int, default=200_000)
    ap.add_argument("--n-ground", type=int, default=30_000)
    ap.add_argument("--n-clutter-blobs", type=int, default=6)
    ap.add_argument("--n-clutter-per-blob", type=int, default=800)

    # Trajectory
    ap.add_argument("--n-frames", type=int, default=120)
    ap.add_argument("--orbit-radius", type=float, default=12.0)
    ap.add_argument("--height-amp", type=float, default=2.0)
    ap.add_argument("--n-loops", type=int, default=1)

    # Defects
    ap.add_argument("--n-corrosion-patches", type=int, default=5)
    ap.add_argument("--n-cracks", type=int, default=3)
    ap.add_argument("--n-missing-bolts", type=int, default=4)
    ap.add_argument("--n-deformed-panels", type=int, default=2)

    # Sparse sub-sampling
    ap.add_argument("--sparse-chimney-frac", type=float, default=0.015)

    return ap.parse_args()


# --------------------------------------------------------------------------- #
#  Defect label constants                                                      #
# --------------------------------------------------------------------------- #

LABEL_NORMAL = 0
LABEL_CORROSION = 1
LABEL_CRACK = 2
LABEL_MISSING_BOLT = 3
LABEL_DEFORMED_PANEL = 4

LABEL_NAMES = {
    LABEL_NORMAL: "normal",
    LABEL_CORROSION: "corrosion",
    LABEL_CRACK: "crack",
    LABEL_MISSING_BOLT: "missing_bolt",
    LABEL_DEFORMED_PANEL: "deformed_panel",
}


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rng_chimney, rng_corrosion, rng_crack, rng_bolt, rng_panel, rng_ground, rng_clutter = (
        np.random.default_rng(s)
        for s in rng.integers(0, 2**31 - 1, size=7)
    )

    # ================================================================== #
    #  Chimney geometry                                                    #
    # ================================================================== #
    chimney = geometry.sample_tapered_cylinder(
        H=args.height,
        r_base=args.r_base,
        r_top=args.r_top,
        n_points=args.n_chimney,
        n_bands=args.n_bands,
        rng=rng_chimney,
    )

    base_rgb = np.where(
        chimney.is_band[:, None],
        BAND_RGB[None, :],
        CHIMNEY_BASE_RGB[None, :],
    ).astype(np.float32)

    n_chim = chimney.positions.shape[0]
    r_mean = 0.5 * (args.r_base + args.r_top)

    # Start with chimney defaults for per-point properties
    positions = chimney.positions.copy()
    rgb = base_rgb.copy()
    opacity = np.full(n_chim, defects.logit(0.90), dtype=np.float32)
    log_scale = np.full((n_chim, 3), np.log(0.02), dtype=np.float32)

    # Ground-truth labels: 0 = normal by default
    gt_labels = np.zeros(n_chim, dtype=np.int32)

    r_at_h = geometry.r_of_h(chimney.uv[:, 1], args.height, args.r_base, args.r_top)

    # ================================================================== #
    #  1.  Corrosion                                                       #
    # ================================================================== #
    corrosion_patches = defects.generate_corrosion_patches(
        n_patches=args.n_corrosion_patches,
        H=args.height,
        r_mean=r_mean,
        rng=rng_corrosion,
    )
    rgb, corroded_mask = defects.apply_corrosion(
        uv=chimney.uv, r_at_h=r_at_h, base_rgb=rgb, patches=corrosion_patches,
    )
    gt_labels[corroded_mask] = LABEL_CORROSION

    # ================================================================== #
    #  2.  Cracks                                                          #
    # ================================================================== #
    crack_list = defects.generate_cracks(
        n_cracks=args.n_cracks,
        H=args.height,
        r_mean=r_mean,
        rng=rng_crack,
    )
    positions, rgb, log_scale, crack_mask = defects.apply_cracks(
        uv=chimney.uv,
        r_at_h=r_at_h,
        normals=chimney.normals,
        positions=positions,
        rgb=rgb,
        log_scale=log_scale,
        cracks=crack_list,
    )
    # Crack overwrites corrosion where both overlap
    gt_labels[crack_mask & (gt_labels == LABEL_NORMAL)] = LABEL_CRACK
    gt_labels[crack_mask] = LABEL_CRACK

    # ================================================================== #
    #  3.  Missing bolts                                                   #
    # ================================================================== #
    bolt_list = defects.generate_missing_bolts(
        n_bolts=args.n_missing_bolts,
        H=args.height,
        n_bands=args.n_bands,
        rng=rng_bolt,
    )
    positions, rgb, opacity, bolt_mask = defects.apply_missing_bolts(
        uv=chimney.uv,
        r_at_h=r_at_h,
        normals=chimney.normals,
        positions=positions,
        rgb=rgb,
        opacity=opacity,
        bolts=bolt_list,
    )
    gt_labels[bolt_mask] = LABEL_MISSING_BOLT

    # ================================================================== #
    #  4.  Deformed panels                                                 #
    # ================================================================== #
    panel_list = defects.generate_deformed_panels(
        n_panels=args.n_deformed_panels,
        H=args.height,
        rng=rng_panel,
    )
    positions, log_scale, panel_mask = defects.apply_deformed_panels(
        uv=chimney.uv,
        normals=chimney.normals,
        positions=positions,
        log_scale=log_scale,
        panels=panel_list,
    )
    gt_labels[panel_mask] = LABEL_DEFORMED_PANEL

    # ================================================================== #
    #  Ground + clutter (background / non-structure)                       #
    # ================================================================== #
    if not args.no_background:
        field_extent = max(4.0 * args.orbit_radius, 50.0)
        ground_pos = geometry.sample_ground_plane(
            extent=field_extent, n_points=args.n_ground, noise_sigma=0.03, rng=rng_ground,
        )
        ground_rgb = np.tile(GROUND_RGB, (ground_pos.shape[0], 1))

        clutter_pos = geometry.sample_background_clutter(
            n_blobs=args.n_clutter_blobs,
            points_per_blob=args.n_clutter_per_blob,
            chimney_radius_max=args.r_base,
            field_extent=field_extent,
            height_range=(-3.0, -0.1),
            rng=rng_clutter,
        )
        clutter_rgb = np.tile(CLUTTER_RGB, (clutter_pos.shape[0], 1))
    else:
        ground_pos = np.empty((0, 3), dtype=np.float32)
        ground_rgb = np.empty((0, 3), dtype=np.float32)
        clutter_pos = np.empty((0, 3), dtype=np.float32)
        clutter_rgb = np.empty((0, 3), dtype=np.float32)

    # ================================================================== #
    # Assemble full GSplat array
    all_positions = np.concatenate([positions, ground_pos, clutter_pos], axis=0)
    all_rgb = np.concatenate([rgb, ground_rgb, clutter_rgb], axis=0)
    f_dc = defects.rgb_to_sh_dc(all_rgb)

    n_grnd = ground_pos.shape[0]
    n_clut = clutter_pos.shape[0]
    n_total = n_chim + n_grnd + n_clut

    # Extend opacity
    all_opacity = np.empty(n_total, dtype=np.float32)
    all_opacity[:n_chim] = opacity
    all_opacity[n_chim:n_chim + n_grnd] = defects.logit(0.50)
    all_opacity[n_chim + n_grnd:] = defects.logit(0.70)

    # Extend log_scale
    all_log_scale = np.empty((n_total, 3), dtype=np.float32)
    all_log_scale[:n_chim] = log_scale
    all_log_scale[n_chim:n_chim + n_grnd] = np.log(0.05)
    all_log_scale[n_chim + n_grnd:] = np.log(0.08)

    # Extend normals: chimney has real outward normals, ground/clutter just point UP (-Y)
    all_normals = np.empty((n_total, 3), dtype=np.float32)
    all_normals[:n_chim] = chimney.normals
    all_normals[n_chim:] = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    gsplat = ply_writer.build_gsplat_array(all_positions, f_dc, all_opacity, all_log_scale, all_normals)
    ply_writer.save_ply(os.path.join(args.out_dir, "gsplat.ply"), gsplat)

    # ================================================================== #
    #  Sparse PLY (subsample chimney + all ground + all clutter)           #
    # ================================================================== #
    n_sparse_chim = max(1, int(args.sparse_chimney_frac * n_chim))
    sparse_idx = rng.choice(n_chim, size=n_sparse_chim, replace=False)
    sparse_pos = np.concatenate(
        [positions[sparse_idx], ground_pos, clutter_pos], axis=0,
    )
    sparse_arr = ply_writer.build_sparse_array(sparse_pos)
    ply_writer.save_ply(os.path.join(args.out_dir, "sparse.ply"), sparse_arr)

    # ================================================================== #
    #  Camera trajectory + images.bin                                      #
    # ================================================================== #
    traj = trajectory.orbital_trajectory(
        n_frames=args.n_frames,
        radius=args.orbit_radius,
        H=args.height,
        height_amp=args.height_amp,
        n_loops=args.n_loops,
    )
    names = [f"frame_{i:04d}.jpg" for i in range(args.n_frames)]
    images_bin = os.path.join(args.out_dir, "images.bin")
    colmap_writer.write_images_bin(images_bin, traj["qvecs"], traj["translations"], names)

    # Round-trip verification
    loaded_centers = get_colmap_camera_centers(images_bin)
    err = float(np.max(np.linalg.norm(loaded_centers - traj["centers"], axis=1)))
    if err > 1e-6:
        raise RuntimeError(f"COLMAP round-trip mismatch: max error {err}")

    # ================================================================== #
    #  Ground-truth labels                                                 #
    # ================================================================== #
    np.save(os.path.join(args.out_dir, "ground_truth_labels.npy"), gt_labels)

    # ================================================================== #
    #  Metadata                                                            #
    # ================================================================== #
    defect_counts = {
        name: int((gt_labels == label).sum())
        for label, name in LABEL_NAMES.items()
    }

    meta = {
        "seed": args.seed,
        "frame": "COLMAP world (+Y down)",
        "label_encoding": {str(k): v for k, v in LABEL_NAMES.items()},
        "chimney": {
            "height": args.height,
            "r_base": args.r_base,
            "r_top": args.r_top,
            "n_bands": args.n_bands,
            "base_index_range": [0, n_chim],
        },
        "ground_index_range": [n_chim, n_chim + n_grnd],
        "clutter_index_range": [n_chim + n_grnd, n_total],
        "defects": {
            "corrosion_patches": [p.to_dict() for p in corrosion_patches],
            "cracks": [c.to_dict() for c in crack_list],
            "missing_bolts": [b.to_dict() for b in bolt_list],
            "deformed_panels": [p.to_dict() for p in panel_list],
        },
        "defect_counts": defect_counts,
        "trajectory": {
            "n_frames": args.n_frames,
            "orbit_radius": args.orbit_radius,
            "height_amp": args.height_amp,
            "n_loops": args.n_loops,
            "round_trip_max_error_m": err,
        },
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ================================================================== #
    #  Summary                                                             #
    # ================================================================== #
    print(f"Wrote synthetic dataset to {args.out_dir}")
    print(f"  gsplat.ply  : {n_total} points ({n_chim} chimney / {n_grnd} ground / {n_clut} clutter)")
    print(f"  sparse.ply  : {sparse_pos.shape[0]} points")
    print(f"  images.bin  : {args.n_frames} frames (round-trip max err {err:.2e} m)")
    print(f"  Defect counts:")
    for label, name in LABEL_NAMES.items():
        count = defect_counts[name]
        pct = 100.0 * count / n_chim if n_chim > 0 else 0.0
        print(f"    {name:20s}: {count:6d} points ({pct:.2f}% of chimney)")
    print(f"  ground_truth_labels.npy : {n_chim} labels (chimney points only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
