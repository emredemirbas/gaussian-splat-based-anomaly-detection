"""
Direct anomaly detection runner for clean synthetic data.
Bypasses the Stage 1 (Sparse Filter) and Stage 2 (Sphere Filter) steps
which are designed for isolating real-world objects from background clutter.

Usage:
    python run_ad_only.py \
        --gsplat_ply data/synthetic/synth_clean/gsplat.ply \
        --output_ply data/synthetic/synth_clean/output.ply \
        --pfa 0.01
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
import anomaly_detector
from cluster_metadata import compute_cluster_metadata
from cluster_io import save_clusters


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Direct Anomaly Detection Pipeline")
    parser.add_argument("--gsplat_ply", type=str, required=True,
                        help="Path to the input clean gsplat PLY")
    parser.add_argument("--output_ply", type=str, required=True,
                        help="Path to save the output PLY with anomaly scores")
    parser.add_argument("--pfa", type=float, default=0.01,
                        help="Probability of false alarm for CFAR threshold")
    parser.add_argument("--no_viewer", action="store_true",
                        help="Skip launching the 3D viewer")
    parser.add_argument("--hidpi", action="store_true",
                        help="Enable HiDPI mode in the viewer")
    return parser


def run_ad_only(args):
    # ------------------------------------------------------------------ #
    #  Load Data                                                         #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("  Load GSplat Data")
    print("=" * 60)
    
    gsplat_data = io_utils.PlyIO.read(args.gsplat_ply)
    n_points = gsplat_data.shape[0]
    print(f"Loaded GSplat cloud: {n_points} points")

    # ------------------------------------------------------------------ #
    #  STAGE 3: Anomaly Detection                                        #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  Anomaly Detection (RX)")
    print("=" * 60)

    from anomaly_detector import extract_features, RRXDetector, estimate_sigma, cfar_threshold_scores, cluster_and_color_anomalies
    from sklearn.preprocessing import RobustScaler

    # 1. Feature Extraction
    print("Extracting features (Position + Color + Opacity + Scale)...")
    X, feature_names = extract_features(gsplat_data, use_position=True, use_color=True, use_opacity=True, use_scale=True)
    print(f"Using features: {feature_names}")

    print("Scaling features using RobustScaler...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    print("Estimating RBF sigma...")
    sigma = estimate_sigma(X_scaled)
    print(f"Estimated sigma: {sigma:.4f}")

    # 2. RRX Detection
    print("Running Randomized RX Detector...")
    det = RRXDetector(n_features=300, sigma=sigma)
    scores = det.fit_score(X_scaled)

    # 3. CFAR Thresholding
    out_dir = os.path.dirname(args.output_ply)
    cfar_plot_path = os.path.join(out_dir, "output_cfar_plot.png")
    
    print("\nApplying CFAR thresholding...")
    is_anomaly, cfar_tau, fit_info = cfar_threshold_scores(
        scores, pfa=args.pfa, n_bins=100, plot=True,
        save_path=cfar_plot_path,
    )
    
    anomaly_indices = np.where(is_anomaly)[0]
    print(f"[CFAR] Anomalies: {len(anomaly_indices)} / {n_points} "
          f"({len(anomaly_indices)/n_points*100:.2f}%)")

    import numpy.lib.recfunctions as rfn
    gsplat_data = rfn.append_fields(gsplat_data, 'anomaly_score', scores.astype(np.float32), usemask=False, asrecarray=False)

    # ------------------------------------------------------------------ #
    #  Clustering                                                        #
    # ------------------------------------------------------------------ #
    out_anomalies_ply = os.path.join(out_dir, "output_ad_only_anomalies.ply")
    anomalies_data = gsplat_data[is_anomaly]
    
    anomaly_labels = None
    clusters = []
    
    if len(anomaly_indices) > 0:
        print("\nClustering and coloring anomalies...")
        anomalies_data, anomaly_labels = cluster_and_color_anomalies(
            anomalies_data, eps=0.5, min_samples=10
        )
        io_utils.PlyIO.save(out_anomalies_ply, anomalies_data, as_text=False)
        print(f"Saved {len(anomalies_data)} anomalies to separate file: {out_anomalies_ply}")

        # Compute cluster metadata for the viewer
        from main import _stack_xyz
        anomaly_positions = _stack_xyz(anomalies_data)
        
        # Load camera centers to calculate optimal viewing angles for each cluster
        images_bin_path = args.gsplat_ply.replace("gsplat.ply", "images.bin")
        if os.path.exists(images_bin_path):
            cam_centers = io_utils.get_colmap_camera_centers(images_bin_path)
        else:
            print(f"Warning: {images_bin_path} not found. Camera navigation may be suboptimal.")
            cam_centers = np.zeros((1, 3))
            
        anomaly_scores = scores[is_anomaly]
        clusters = compute_cluster_metadata(
            gsplat_xyz=anomaly_positions, 
            labels=anomaly_labels,
            scores=anomaly_scores,
            cam_centers=cam_centers
        )

    print(f"Anomaly detection complete. Total points: {n_points}.")
    io_utils.PlyIO.save(args.output_ply, gsplat_data, as_text=False)
    print(f"\nSaved main PLY with anomaly scores: {args.output_ply}")

    if len(clusters) > 0:
        out_clusters_npz = os.path.join(out_dir, "output_clusters.npz")
        save_clusters(out_clusters_npz, clusters)

    return clusters


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_ply), exist_ok=True)
    clusters = run_ad_only(args)

    # ------------------------------------------------------------------ #
    #  Viewer Launch                                                     #
    # ------------------------------------------------------------------ #
    if args.no_viewer:
        print("\n[run_ad_only] --no_viewer specified, skipping viewer launch.")
        return

    output_ply_abs = os.path.abspath(args.output_ply)

    parent_dir = os.path.dirname(_PROJECT_ROOT)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    viewer_dir = os.path.join(os.path.dirname(_PROJECT_ROOT), "gsplat_viewer")
    os.chdir(viewer_dir)

    if viewer_dir not in sys.path:
        sys.path.insert(0, viewer_dir)

    saved_argv = sys.argv
    sys.argv = ["gsplat_viewer"]
    if args.hidpi:
        sys.argv.append("--hidpi")

    import gsplat_viewer.main as viewer_main

    print(f"\n[run_ad_only] Launching GSplat viewer.")
    print(f"[run_ad_only] Auto-loading file:")
    print(f"  {output_ply_abs}")
    if len(clusters) > 0:
        print(f"[run_ad_only] {len(clusters)} anomaly clusters found (metadata saved to *_clusters.npz)")

    sys.argv = saved_argv

    # Launch viewer with auto-load
    viewer_main.launch_with_clusters(output_ply_abs, clusters, args.hidpi)


if __name__ == "__main__":
    main()
