"""Visualize synthetic defects and anomaly detection results.

Generates a multi-panel figure showing:
  1. Ground-truth defect labels on the chimney (3D + unwrapped surface)
  2. Pipeline anomaly detections vs. ground truth (TP/FP/FN)
  3. Anomaly score heatmap

Usage:
    python3 visualize_results.py \
        --gsplat_ply data/synthetic/synth_multi/gsplat.ply \
        --output_ply data/synthetic/synth_multi/output.ply \
        --gt_labels  data/synthetic/synth_multi/ground_truth_labels.npy \
        --meta       data/synthetic/synth_multi/meta.json \
        --save_dir   data/synthetic/synth_multi/figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from io_utils import PlyIO

# Label constants
LABEL_NAMES = {
    0: "Normal",
    1: "Corrosion",
    2: "Crack",
    3: "Missing Bolt",
    4: "Deformed Panel",
}

LABEL_COLORS = {
    0: "#B0B0B0",     # light grey — normal
    1: "#E8570E",     # orange — corrosion (rust)
    2: "#1A1A2E",     # near-black — crack
    3: "#FFD700",     # gold — missing bolt
    4: "#6A0DAD",     # purple — deformed panel
}

DETECTION_COLORS = {
    "TP": "#00C853",  # green — true positive
    "FP": "#FF1744",  # red — false positive
    "FN": "#FF9100",  # amber — false negative
    "TN": "#B0B0B0",  # grey — true negative (not shown)
}


def _stack_xyz(arr):
    return np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float64)


def _subsample(mask, max_points=15000, rng=None):
    """Subsample indices where mask is True, keeping at most max_points."""
    idx = np.where(mask)[0]
    if len(idx) <= max_points:
        return idx
    rng = rng or np.random.default_rng(0)
    return rng.choice(idx, size=max_points, replace=False)


# --------------------------------------------------------------------------- #
#  Figure 1 — Ground Truth: 3D view + unwrapped surface                       #
# --------------------------------------------------------------------------- #

def plot_ground_truth(gsplat_xyz, gt_labels_full, n_chimney, meta, save_path=None):
    """3D chimney with GT defect labels + unwrapped surface view."""
    fig = plt.figure(figsize=(20, 9))
    fig.suptitle("Ground-Truth Defect Labels on Synthetic Chimney",
                 fontsize=16, fontweight="bold", y=0.98)

    chimney_xyz = gsplat_xyz[:n_chimney]
    chimney_labels = gt_labels_full[:n_chimney]

    rng = np.random.default_rng(42)

    # --- Panel 1: 3D scatter (two views) ---
    for subplot_idx, (elev, azim, title) in enumerate([
        (15, 45, "3D View (front)"),
        (15, 135, "3D View (side)"),
    ], start=1):
        ax = fig.add_subplot(1, 3, subplot_idx, projection="3d")

        # Draw normal points (subsampled)
        normal_idx = _subsample(chimney_labels == 0, max_points=8000, rng=rng)
        ax.scatter(
            chimney_xyz[normal_idx, 0],
            chimney_xyz[normal_idx, 2],
            -chimney_xyz[normal_idx, 1],  # flip Y for display (COLMAP Y-down)
            c=LABEL_COLORS[0], s=0.3, alpha=0.15, rasterized=True,
        )

        # Draw each defect type on top
        for label_id in [1, 2, 3, 4]:
            mask = chimney_labels == label_id
            if not np.any(mask):
                continue
            idx = _subsample(mask, max_points=5000, rng=rng)
            ax.scatter(
                chimney_xyz[idx, 0],
                chimney_xyz[idx, 2],
                -chimney_xyz[idx, 1],
                c=LABEL_COLORS[label_id],
                s=4.0 if label_id != 3 else 15.0,  # bolts are tiny, make bigger
                alpha=0.9,
                label=LABEL_NAMES[label_id],
                rasterized=True,
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Height")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=12)
        if subplot_idx == 1:
            ax.legend(loc="upper left", fontsize=8, markerscale=3)

    # --- Panel 3: Unwrapped surface (theta vs height) ---
    ax2 = fig.add_subplot(1, 3, 3)

    # Compute theta and height from XYZ
    theta = np.arctan2(chimney_xyz[:, 2], chimney_xyz[:, 0])  # atan2(z, x)
    height = -chimney_xyz[:, 1]  # COLMAP Y-down → flip

    # Plot normal background
    normal_idx = _subsample(chimney_labels == 0, max_points=10000, rng=rng)
    ax2.scatter(
        np.degrees(theta[normal_idx]), height[normal_idx],
        c=LABEL_COLORS[0], s=0.2, alpha=0.1, rasterized=True,
    )

    # Plot defects
    for label_id in [1, 2, 3, 4]:
        mask = chimney_labels == label_id
        if not np.any(mask):
            continue
        idx = _subsample(mask, max_points=5000, rng=rng)
        ax2.scatter(
            np.degrees(theta[idx]), height[idx],
            c=LABEL_COLORS[label_id],
            s=3.0 if label_id != 3 else 12.0,
            alpha=0.8,
            label=f"{LABEL_NAMES[label_id]} ({int(mask.sum())} pts)",
            rasterized=True,
        )

    ax2.set_xlabel("Azimuth (degrees)", fontsize=11)
    ax2.set_ylabel("Height (m)", fontsize=11)
    ax2.set_title("Unwrapped Chimney Surface", fontsize=12)
    ax2.legend(loc="upper right", fontsize=8, markerscale=3)
    ax2.set_xlim(-180, 180)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 2 — Detection Results: TP / FP / FN                                 #
# --------------------------------------------------------------------------- #

def plot_detection_results(
    output_xyz, scores, pred_anomaly_mask, gt_for_output, save_path=None,
):
    """Show TP/FP/FN on the surviving chimney points."""
    fig = plt.figure(figsize=(20, 9))
    fig.suptitle("Anomaly Detection Results vs. Ground Truth",
                 fontsize=16, fontweight="bold", y=0.98)

    gt_defect = gt_for_output > 0
    tp_mask = gt_defect & pred_anomaly_mask
    fp_mask = ~gt_defect & pred_anomaly_mask
    fn_mask = gt_defect & ~pred_anomaly_mask
    tn_mask = ~gt_defect & ~pred_anomaly_mask

    rng = np.random.default_rng(42)

    # --- Panel 1: 3D TP/FP/FN ---
    ax = fig.add_subplot(1, 3, 1, projection="3d")

    # TN (background)
    tn_idx = _subsample(tn_mask, max_points=6000, rng=rng)
    ax.scatter(
        output_xyz[tn_idx, 0], output_xyz[tn_idx, 2], -output_xyz[tn_idx, 1],
        c=DETECTION_COLORS["TN"], s=0.2, alpha=0.08, rasterized=True,
    )

    # FN
    fn_idx = _subsample(fn_mask, max_points=5000, rng=rng)
    if len(fn_idx) > 0:
        ax.scatter(
            output_xyz[fn_idx, 0], output_xyz[fn_idx, 2], -output_xyz[fn_idx, 1],
            c=DETECTION_COLORS["FN"], s=3.0, alpha=0.6,
            label=f"FN (missed, {int(fn_mask.sum())})", rasterized=True,
        )

    # FP
    fp_idx = np.where(fp_mask)[0]
    if len(fp_idx) > 0:
        ax.scatter(
            output_xyz[fp_idx, 0], output_xyz[fp_idx, 2], -output_xyz[fp_idx, 1],
            c=DETECTION_COLORS["FP"], s=15.0, alpha=0.9,
            label=f"FP (false alarm, {int(fp_mask.sum())})", rasterized=True,
        )

    # TP
    tp_idx = np.where(tp_mask)[0]
    if len(tp_idx) > 0:
        ax.scatter(
            output_xyz[tp_idx, 0], output_xyz[tp_idx, 2], -output_xyz[tp_idx, 1],
            c=DETECTION_COLORS["TP"], s=15.0, alpha=0.9,
            label=f"TP (detected, {int(tp_mask.sum())})", rasterized=True,
        )

    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Height")
    ax.view_init(elev=15, azim=45)
    ax.set_title("3D Detection Map", fontsize=12)
    ax.legend(loc="upper left", fontsize=9, markerscale=2)

    # --- Panel 2: Unwrapped TP/FP/FN ---
    ax2 = fig.add_subplot(1, 3, 2)

    theta = np.degrees(np.arctan2(output_xyz[:, 2], output_xyz[:, 0]))
    height = -output_xyz[:, 1]

    # Background
    ax2.scatter(theta[tn_idx], height[tn_idx],
                c=DETECTION_COLORS["TN"], s=0.2, alpha=0.08, rasterized=True)

    if len(fn_idx) > 0:
        ax2.scatter(theta[fn_idx], height[fn_idx],
                    c=DETECTION_COLORS["FN"], s=3.0, alpha=0.5,
                    label=f"FN ({int(fn_mask.sum())})", rasterized=True)
    if len(fp_idx) > 0:
        ax2.scatter(theta[fp_idx], height[fp_idx],
                    c=DETECTION_COLORS["FP"], s=12.0, alpha=0.9,
                    label=f"FP ({int(fp_mask.sum())})", rasterized=True)
    if len(tp_idx) > 0:
        ax2.scatter(theta[tp_idx], height[tp_idx],
                    c=DETECTION_COLORS["TP"], s=12.0, alpha=0.9,
                    label=f"TP ({int(tp_mask.sum())})", rasterized=True)

    ax2.set_xlabel("Azimuth (degrees)", fontsize=11)
    ax2.set_ylabel("Height (m)", fontsize=11)
    ax2.set_title("Unwrapped Detection Map", fontsize=12)
    ax2.legend(loc="upper right", fontsize=9, markerscale=2)
    ax2.set_xlim(-180, 180)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Score heatmap (unwrapped) ---
    ax3 = fig.add_subplot(1, 3, 3)

    # Use log scale for scores (heavy-tailed)
    log_scores = np.log1p(scores)

    sc = ax3.scatter(
        theta, height,
        c=log_scores, cmap="hot", s=0.5, alpha=0.6,
        vmin=np.percentile(log_scores, 5),
        vmax=np.percentile(log_scores, 99.5),
        rasterized=True,
    )
    cbar = plt.colorbar(sc, ax=ax3, shrink=0.8, pad=0.02)
    cbar.set_label("log(1 + score)", fontsize=10)

    ax3.set_xlabel("Azimuth (degrees)", fontsize=11)
    ax3.set_ylabel("Height (m)", fontsize=11)
    ax3.set_title("Anomaly Score Heatmap", fontsize=12)
    ax3.set_xlim(-180, 180)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Figure 3 — Per-defect-type detection breakdown                              #
# --------------------------------------------------------------------------- #

def plot_per_defect_detection(
    output_xyz, pred_anomaly_mask, gt_for_output, save_path=None,
):
    """One subplot per defect type showing detected vs. missed."""
    defect_types = [(1, "Corrosion"), (2, "Crack"), (3, "Missing Bolt"), (4, "Deformed Panel")]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Per-Defect-Type Detection Breakdown",
                 fontsize=16, fontweight="bold", y=1.02)

    theta = np.degrees(np.arctan2(output_xyz[:, 2], output_xyz[:, 0]))
    height = -output_xyz[:, 1]

    rng = np.random.default_rng(42)
    bg_idx = _subsample(gt_for_output == 0, max_points=5000, rng=rng)

    for ax, (label_id, name) in zip(axes, defect_types):
        gt_mask = gt_for_output == label_id
        n_gt = int(gt_mask.sum())

        # Background
        ax.scatter(theta[bg_idx], height[bg_idx],
                   c="#E0E0E0", s=0.3, alpha=0.15, rasterized=True)

        if n_gt == 0:
            ax.text(0.5, 0.5, "No points\nsurvived\nfiltering",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=14, color="#999", style="italic")
        else:
            # Missed (FN)
            fn = gt_mask & ~pred_anomaly_mask
            fn_idx = np.where(fn)[0]
            if len(fn_idx) > 0:
                ax.scatter(theta[fn_idx], height[fn_idx],
                           c=DETECTION_COLORS["FN"], s=5.0, alpha=0.7,
                           label=f"Missed ({int(fn.sum())})", rasterized=True)

            # Detected (TP)
            tp = gt_mask & pred_anomaly_mask
            tp_idx = np.where(tp)[0]
            if len(tp_idx) > 0:
                ax.scatter(theta[tp_idx], height[tp_idx],
                           c=DETECTION_COLORS["TP"], s=12.0, alpha=0.9,
                           label=f"Detected ({int(tp.sum())})", rasterized=True)

            recall = int(tp.sum()) / n_gt if n_gt > 0 else 0
            ax.text(0.02, 0.98,
                    f"GT: {n_gt}\nRecall: {recall:.1%}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

        ax.set_title(f"{name}", fontsize=13, fontweight="bold",
                     color=LABEL_COLORS[label_id])
        ax.set_xlabel("Azimuth (°)")
        ax.set_ylabel("Height (m)")
        ax.set_xlim(-180, 180)
        ax.grid(True, alpha=0.3)
        if n_gt > 0:
            ax.legend(loc="upper right", fontsize=8, markerscale=2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Visualize synthetic defects and detection results.")
    ap.add_argument("--gsplat_ply", required=True)
    ap.add_argument("--output_ply", required=True)
    ap.add_argument("--gt_labels", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--save_dir", default=None, help="Directory to save figures (default: next to output_ply)")
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()

    # ---- Load ------------------------------------------------------------ #
    with open(args.meta) as f:
        meta = json.load(f)

    gt_labels_chimney = np.load(args.gt_labels)
    n_chimney = meta["chimney"]["base_index_range"][1]
    n_total = meta["clutter_index_range"][1]

    gt_labels_full = np.zeros(n_total, dtype=np.int32)
    gt_labels_full[:n_chimney] = gt_labels_chimney

    gsplat_data = PlyIO.read(args.gsplat_ply)
    gsplat_xyz = _stack_xyz(gsplat_data)

    output_data = PlyIO.read(args.output_ply)
    output_xyz = _stack_xyz(output_data)
    scores = output_data["anomaly_score"].astype(np.float64)

    # Match output → gsplat
    from scipy.spatial import cKDTree
    tree = cKDTree(gsplat_xyz)
    _, matched_idx = tree.query(output_xyz)
    gt_for_output = gt_labels_full[matched_idx]

    # Determine predicted anomalies
    anomaly_path = args.output_ply.replace(".ply", "_only_anomalies.ply")
    if os.path.exists(anomaly_path):
        anomaly_data = PlyIO.read(anomaly_path)
        anomaly_xyz = _stack_xyz(anomaly_data)
        tree2 = cKDTree(output_xyz)
        _, anom_idx = tree2.query(anomaly_xyz)
        pred_anomaly_mask = np.zeros(len(output_data), dtype=bool)
        pred_anomaly_mask[anom_idx] = True
    else:
        threshold = np.percentile(scores, 99.0)
        pred_anomaly_mask = scores > threshold

    # ---- Output directory ------------------------------------------------ #
    save_dir = args.save_dir or os.path.join(os.path.dirname(args.output_ply), "figures")
    os.makedirs(save_dir, exist_ok=True)

    # ---- Generate figures ------------------------------------------------ #
    print("Generating visualizations...")

    plot_ground_truth(
        gsplat_xyz, gt_labels_full, n_chimney, meta,
        save_path=os.path.join(save_dir, "1_ground_truth.png"),
    )

    plot_detection_results(
        output_xyz, scores, pred_anomaly_mask, gt_for_output,
        save_path=os.path.join(save_dir, "2_detection_results.png"),
    )

    plot_per_defect_detection(
        output_xyz, pred_anomaly_mask, gt_for_output,
        save_path=os.path.join(save_dir, "3_per_defect_breakdown.png"),
    )

    print(f"\nAll figures saved to: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
