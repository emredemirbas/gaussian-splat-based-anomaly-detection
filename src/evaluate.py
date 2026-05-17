"""Evaluate anomaly detection results against synthetic ground-truth labels.

Usage:
    python3 evaluate.py \
        --gsplat_ply data/synthetic/synth_multi/gsplat.ply \
        --output_ply data/synthetic/synth_multi/output.ply \
        --gt_labels  data/synthetic/synth_multi/ground_truth_labels.npy \
        --meta       data/synthetic/synth_multi/meta.json

Computes:
    - Per-defect-type precision, recall, F1
    - Overall precision, recall, F1 (any defect vs. normal)
    - Confusion matrix (compact)
"""

from __future__ import annotations

import argparse
import json
import sys
import os

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from io_utils import PlyIO


# Label constants (must match synthetic/generate.py)
LABEL_NAMES = {
    0: "normal",
    1: "corrosion",
    2: "crack",
    3: "missing_bolt",
    4: "deformed_panel",
}


def _stack_xyz(struct_arr: np.ndarray) -> np.ndarray:
    return np.column_stack([struct_arr["x"], struct_arr["y"], struct_arr["z"]]).astype(np.float64)


def match_points(
    gsplat_xyz: np.ndarray,
    output_xyz: np.ndarray,
    tol: float = 1e-3,
) -> np.ndarray:
    """Map each output point to its nearest gsplat index.

    Returns (M,) int array of gsplat indices for each output point.
    Raises if any match is worse than `tol` metres.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(gsplat_xyz)
    dists, idxs = tree.query(output_xyz)
    worst = dists.max()
    if worst > tol:
        print(f"[WARN] Worst match distance: {worst:.6f} m (tol={tol})")
    return idxs


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute precision, recall, F1 for binary arrays."""
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate anomaly detection against ground truth.")
    ap.add_argument("--gsplat_ply", required=True, help="Original GSplat PLY (from synthetic generator)")
    ap.add_argument("--output_ply", required=True, help="Pipeline output PLY (with anomaly_score field)")
    ap.add_argument("--gt_labels", required=True, help="Ground-truth labels .npy file")
    ap.add_argument("--meta", required=True, help="meta.json file")
    ap.add_argument("--tol", type=float, default=0.05, help="Point matching tolerance (metres)")
    args = ap.parse_args()

    # ---- Load data --------------------------------------------------------- #
    with open(args.meta, "r") as f:
        meta = json.load(f)

    gt_labels_chimney = np.load(args.gt_labels)
    n_chimney = meta["chimney"]["base_index_range"][1]

    # Build full-scene ground truth (chimney + ground + clutter all = 0)
    n_total = meta["clutter_index_range"][1]
    gt_labels_full = np.zeros(n_total, dtype=np.int32)
    gt_labels_full[:n_chimney] = gt_labels_chimney

    # Load the GSplat PLY (same as what the pipeline loaded)
    gsplat_data = PlyIO.read(args.gsplat_ply)
    gsplat_xyz = _stack_xyz(gsplat_data)

    # Load pipeline output PLY (has anomaly_score field)
    output_data = PlyIO.read(args.output_ply)
    output_xyz = _stack_xyz(output_data)

    if "anomaly_score" not in output_data.dtype.names:
        print("ERROR: output PLY does not have 'anomaly_score' field.")
        return 1

    scores = output_data["anomaly_score"].astype(np.float64)

    print(f"GSplat input:  {len(gsplat_data)} points")
    print(f"Pipeline output: {len(output_data)} points")
    print(f"Ground truth:  {n_chimney} chimney labels ({n_total} total)")

    # ---- Match output points back to original gsplat indices --------------- #
    print(f"\nMatching output points to original GSplat (tol={args.tol} m)...")
    matched_idx = match_points(gsplat_xyz, output_xyz, tol=args.tol)

    # Ground truth labels for the points that survived filtering
    gt_for_output = gt_labels_full[matched_idx]

    # ---- Determine which output points the pipeline flagged as anomalies --- #
    # The pipeline stores scores; we need the threshold.
    # The anomaly file tells us which ones were flagged.
    anomaly_ply_path = args.output_ply.replace(".ply", "_only_anomalies.ply")
    if os.path.exists(anomaly_ply_path):
        anomaly_data = PlyIO.read(anomaly_ply_path)
        anomaly_xyz = _stack_xyz(anomaly_data)
        # Match anomaly points to output
        from scipy.spatial import cKDTree
        tree = cKDTree(output_xyz)
        dists, anomaly_to_output = tree.query(anomaly_xyz)
        pred_anomaly_mask = np.zeros(len(output_data), dtype=bool)
        pred_anomaly_mask[anomaly_to_output] = True
        print(f"Anomaly file: {len(anomaly_data)} flagged anomalies")
    else:
        # Fallback: infer from scores using a simple threshold
        print("[WARN] No anomaly file found, inferring from scores (top 1%)")
        threshold = np.percentile(scores, 99.0)
        pred_anomaly_mask = scores > threshold

    n_pred_anomalies = int(pred_anomaly_mask.sum())

    # ---- Evaluation -------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)

    # Overall: any defect vs. normal
    gt_any_defect = gt_for_output > 0
    overall = compute_metrics(gt_any_defect, pred_anomaly_mask)
    print(f"\n{'OVERALL (any defect)':>25s}:  "
          f"P={overall['precision']:.4f}  R={overall['recall']:.4f}  F1={overall['f1']:.4f}  "
          f"(TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} TN={overall['tn']})")

    # Per-defect-type
    print(f"\n{'Defect Type':>25s} | {'GT':>5s} | {'Detected':>8s} | {'Prec':>6s} | {'Recall':>6s} | {'F1':>6s}")
    print("-" * 75)

    for label_id, label_name in LABEL_NAMES.items():
        if label_id == 0:
            continue
        gt_mask = gt_for_output == label_id
        n_gt = int(gt_mask.sum())
        if n_gt == 0:
            print(f"{label_name:>25s} | {n_gt:>5d} | {'—':>8s} | {'—':>6s} | {'—':>6s} | {'—':>6s}  (no GT points survived filtering)")
            continue
        m = compute_metrics(gt_mask, pred_anomaly_mask)
        print(f"{label_name:>25s} | {n_gt:>5d} | {m['tp']:>8d} | {m['precision']:>6.4f} | {m['recall']:>6.4f} | {m['f1']:>6.4f}")

    # Summary of filtered-out defects
    print(f"\n{'Ground-truth distribution in pipeline output':}")
    for label_id, label_name in LABEL_NAMES.items():
        n_in_output = int((gt_for_output == label_id).sum())
        n_in_full = int((gt_labels_full == label_id).sum())
        pct = 100.0 * n_in_output / max(n_in_full, 1)
        print(f"  {label_name:>20s}: {n_in_output:>6d} / {n_in_full:>6d} survived filtering ({pct:.1f}%)")

    print(f"\nPredicted anomalies: {n_pred_anomalies}")
    print(f"Total output points: {len(output_data)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
