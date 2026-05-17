"""
Unified entry point: runs the full filtering & anomaly detection pipeline,
computes cluster metadata, then launches the GSplat viewer with anomaly
navigation.

Usage:
    python run_pipeline.py \
        --images_bin path/to/images.bin \
        --sparse_ply path/to/points3D.ply \
        --gsplat_ply path/to/point_cloud.ply \
        --output_ply path/to/output.ply \
        [--hidpi] [other pipeline flags...]
"""

import sys
import os
import argparse
import numpy as np

# Ensure the project root is on the path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import io_utils
from main import _build_arg_parser, run_pipeline, _stack_xyz
from cluster_metadata import compute_cluster_metadata
from cluster_io import save_clusters


def build_arg_parser():
    """Extend the pipeline arg parser with viewer-specific arguments."""
    ap = _build_arg_parser()

    # --- Viewer options ---
    ap.add_argument("--hidpi", action="store_true",
                    help="Enable HiDPI scaling for the viewer interface.")
    ap.add_argument("--no_viewer", action="store_true",
                    help="Run pipeline only, do not launch the viewer.")
    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    # ------------------------------------------------------------------ #
    #  1. Run the full filtering + anomaly detection pipeline             #
    # ------------------------------------------------------------------ #
    results = run_pipeline(args)

    anomalies_data = results["anomalies_data"]
    anomaly_labels = results["anomaly_labels"]
    anomaly_scores = results["anomaly_scores_anomalies"]
    output_ply = results["output_ply"]

    # ------------------------------------------------------------------ #
    #  2. Compute cluster metadata (only if clustering was performed)     #
    # ------------------------------------------------------------------ #
    clusters = []
    if anomaly_labels is not None and len(anomaly_labels) > 0:
        anomaly_xyz = _stack_xyz(anomalies_data)
        cam_centers = io_utils.get_colmap_camera_centers(args.images_bin)

        clusters = compute_cluster_metadata(
            gsplat_xyz=anomaly_xyz,
            labels=anomaly_labels,
            scores=anomaly_scores,
            cam_centers=cam_centers,
        )

        # Save cluster metadata alongside the output PLY
        cluster_path = output_ply.replace(".ply", "_clusters.npz")
        save_clusters(cluster_path, clusters)
    else:
        print("[run_pipeline] No clusters to compute (clustering was skipped or no anomalies found).")

    # ------------------------------------------------------------------ #
    #  3. Launch the GSplat viewer                                        #
    # ------------------------------------------------------------------ #
    if args.no_viewer:
        print("\n[run_pipeline] --no_viewer specified, skipping viewer launch.")
        return

    if len(clusters) == 0:
        print("\n[run_pipeline] No anomaly clusters found. Launching viewer without navigation.")

    # Resolve PLY path to absolute BEFORE changing working directory
    output_ply_abs = os.path.abspath(output_ply)

    # Add the parent directory to path so we can import gsplat_viewer as a package
    parent_dir = os.path.dirname(_PROJECT_ROOT)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Change working directory to the viewer so shaders are found
    viewer_dir = os.path.join(os.path.dirname(_PROJECT_ROOT), "gsplat_viewer")
    os.chdir(viewer_dir)

    # Insert viewer_dir at the very front of sys.path so its internal imports (like `import util`)
    # resolve correctly to the viewer.
    if viewer_dir not in sys.path:
        sys.path.insert(0, viewer_dir)

    # Build argv for the viewer (it parses --hidpi itself)
    saved_argv = sys.argv
    sys.argv = ["gsplat_viewer"]
    if args.hidpi:
        sys.argv.append("--hidpi")

    import gsplat_viewer.main as viewer_main

    # Print the PLY path so the user knows what to load
    print(f"\n[run_pipeline] Launching GSplat viewer.")
    print(f"[run_pipeline] Load this file in the viewer:")
    print(f"  {output_ply_abs}")
    if len(clusters) > 0:
        print(f"[run_pipeline] {len(clusters)} anomaly clusters found (metadata saved to *_clusters.npz)")

    sys.argv = saved_argv

    # Launch viewer with auto-load
    viewer_main.launch_with_clusters(output_ply_abs, clusters, args.hidpi)


if __name__ == "__main__":
    main()
