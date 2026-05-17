import argparse
import numpy as np

import io_utils
import math_utils
from ellipsoid_filter import EllipsoidFilter
from ground_filter import GroundFilter
from sphere_filter import SphereFilter
from outlier_filter import StatisticalOutlierFilter

def _stack_xyz(struct_arr: np.ndarray) -> np.ndarray:
    return np.column_stack([struct_arr["x"], struct_arr["y"], struct_arr["z"]]).astype(np.float64)

def _build_arg_parser():
    """Build and return the argument parser for the pipeline."""
    ap = argparse.ArgumentParser(description="Two-stage trajectory-guided filtering: sparse → sphere-based GSplat filter.")

    # --- Input / Output ---
    ap.add_argument("--images_bin", required=True, help="Path to COLMAP images.bin")
    ap.add_argument("--sparse_ply", required=True, help="Input sparse reconstruction PLY (points3D.ply)")
    ap.add_argument("--gsplat_ply", required=True, help="Input Gaussian Splatting PLY")
    ap.add_argument("--output_ply", required=True, help="Output filtered GSplat PLY")
    ap.add_argument("--output_sparse_ply", default=None, help="Optional: save filtered sparse PLY for debugging")

    # --- Trajectory Smoothing ---
    ap.add_argument("--smooth", type=float, default=10e7, help="Trajectory smoothing lambda (0 disables)")
    ap.add_argument("--planar_tau", type=float, default=1e-2, help="Planarity threshold for lambda3/lambda2")

    # --- SDF Fine Filter ---
    ap.add_argument("--sigma", type=float, default=20.0, help="RBF kernel sigma")
    ap.add_argument("--step", type=float, default=0.2, help="Enhancement step (controls thickness)")
    ap.add_argument("--reg", type=float, default=1e-6, help="Regularization for kernel system")
    ap.add_argument("--batch", type=int, default=20000, help="Batch size for SDF prediction")
    ap.add_argument("--sdf_threshold", type=float, default=0.0, help="SDF inclusion threshold (use >0 to keep points slightly outside trajectory)")

    ap.add_argument("--viz", action="store_true", help="Visualize 2D SDF if planar")

    # --- Ground Filtering ---
    ap.add_argument("--ground_percentile", type=float, default=5.0, help="Percentile for initial ground seed censoring")
    ap.add_argument("--ground_margin", type=float, default=0.02, help="Safety margin g (metres) for ground classification")
    ap.add_argument("--ground_ransac_dist", type=float, default=0.10, help="RANSAC inlier distance threshold (metres)")
    ap.add_argument("--ground_slope", type=float, default=0.9, help="Min |C| of normalised normal for slope validation")
    ap.add_argument("--ground_inlier_ratio", type=float, default=0.15, help="Min inlier ratio to accept a RANSAC plane")
    ap.add_argument("--ground_ransac_iters", type=int, default=1000, help="Number of RANSAC iterations")

    # --- Ellipsoid Coarse Filter ---
    ap.add_argument("--inflate_factor", type=float, default=1.50, help="Ellipsoid inflation factor (>1 = larger)")
    ap.add_argument("--ellipsoid_percentile", type=float, default=95.0, help="Percentile for ellipsoid radius computation")
    ap.add_argument("--no_ellipsoid_percentile", action="store_true", help="Use max instead of percentile for ellipsoid radii")
    ap.add_argument("--min_radius", type=float, default=0.5, help="Minimum ellipsoid semi-axis radius")

    # --- Statistical Outlier Removal ---
    ap.add_argument("--sor_neighbors", type=int, default=50, help="Number of nearest neighbours for SOR")
    ap.add_argument("--sor_threshold", type=float, default=1.5, help="Local density ratio threshold for SOR (1.0 = same as neighbours; higher = more lenient)")
    ap.add_argument("--sor_batch_size", type=int, default=100_000, help="Batch size for SOR KD-tree queries")
    ap.add_argument("--no_sor", action="store_true", help="Skip statistical outlier removal")

    # --- Sphere Filter ---
    ap.add_argument("--sphere_radius", type=float, default=0.5, help="Radius of spheres around each filtered sparse point (metres)")
    ap.add_argument("--sphere_batch_size", type=int, default=10000, help="Batch size for sphere filter KD-tree queries (lower = less memory)")

    # --- Scale Filter (3DGS) ---
    ap.add_argument("--scale_threshold", type=float, default=0.5, help="Max Gaussian scale (linear, after exp); splats above this are removed")

    # --- CFAR Anomaly Thresholding ---
    ap.add_argument("--pfa", type=float, default=0.01, help="CFAR probability of false alarm (e.g. 0.01 = 1%%)")
    ap.add_argument("--n_bins", type=int, default=100, help="Number of histogram bins for CFAR distribution fitting")

    # --- Anomaly Clustering ---
    ap.add_argument("--cluster_eps", type=float, default=0.5, help="DBSCAN eps for clustering anomalies")
    ap.add_argument("--cluster_min_samples", type=int, default=10, help="DBSCAN min_samples for clustering anomalies")
    ap.add_argument("--no_cluster", action="store_true", help="Skip clustering and coloring of anomalies")

    return ap


def run_pipeline(args):
    """Run the full filtering + anomaly detection pipeline."""

    # ================================================================== #
    #                     STAGE 1 — FILTER SPARSE CLOUD                   #
    # ================================================================== #
    print("=" * 60)
    print("  STAGE 1: Filter Sparse Reconstruction")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    #  Step 1.1 — Load trajectory (COLMAP frame)                          #
    # ------------------------------------------------------------------ #
    cam_centers = io_utils.get_colmap_camera_centers(args.images_bin)

    if args.smooth and args.smooth > 0:
        cam_smooth = math_utils.smooth_path(cam_centers, lam=args.smooth)
    else:
        cam_smooth = cam_centers.copy()

    # ------------------------------------------------------------------ #
    #  Step 1.2 — PCA on trajectory                                       #
    # ------------------------------------------------------------------ #
    mu, V, eigvals = math_utils.compute_pca_frame(cam_smooth)
    planar, ratio = math_utils.is_planar_from_eigvals(eigvals, tau=args.planar_tau)

    print("=== PCA / Planarity ===")
    print(f"eigvals (desc): {eigvals}")
    print(f"ratio lambda3/lambda2: {ratio:.6e}")
    print(f"planar (tau={args.planar_tau}): {planar}")

    cam_pca = math_utils.apply_pca_transform(cam_smooth, mu, V)

    dim = 2 if planar else 3
    print(f"dim: {dim}")

    # Load sparse PLY
    sparse_data = io_utils.PlyIO.read(args.sparse_ply)
    sparse_xyz_colmap = _stack_xyz(sparse_data)  # COLMAP frame
    sparse_xyz_pca = math_utils.apply_pca_transform(sparse_xyz_colmap, mu, V)  # PCA frame

    print(f"\nLoaded sparse cloud: {len(sparse_data)} points")

    # ------------------------------------------------------------------ #
    #  Step 1.3 — Ellipsoid coarse filtering (PCA frame)                  #
    # ------------------------------------------------------------------ #
    ellipsoid = EllipsoidFilter(
        inflate_factor=args.inflate_factor,
        use_percentile=not args.no_ellipsoid_percentile,
        percentile=args.ellipsoid_percentile,
        min_radius=args.min_radius,
    )

    coarse_mask, ellipsoid_info = ellipsoid.filter(sparse_xyz_pca, cam_pca, dim=dim)

    sparse_data = sparse_data[coarse_mask]
    sparse_xyz_colmap = sparse_xyz_colmap[coarse_mask]
    print(f"Ellipsoid coarse filter kept {sparse_data.shape[0]} / {coarse_mask.size} points")

    # ------------------------------------------------------------------ #
    #  Step 1.4 — Ground filtering in COLMAP frame (Y-down)               #
    # ------------------------------------------------------------------ #
    g_filter = GroundFilter(
        ground_margin=args.ground_margin,
        slope_threshold=args.ground_slope,
        ransac_iterations=args.ground_ransac_iters,
        ransac_distance_threshold=args.ground_ransac_dist,
        inlier_ratio_threshold=args.ground_inlier_ratio,
        percentile=args.ground_percentile,
        vertical_axis=1,        # Y axis in COLMAP
        axis_points_up=False,   # Y points downward in COLMAP
    )

    _, _, _, ground_mask = g_filter.filter_ground(sparse_xyz_colmap)

    keep_non_ground = ~ground_mask
    sparse_data = sparse_data[keep_non_ground]
    sparse_xyz_colmap = sparse_xyz_colmap[keep_non_ground]

    removed_count = int(np.sum(ground_mask))
    print(f"Ground Filter removed {removed_count} points (kept {len(sparse_xyz_colmap)}).")

    # ------------------------------------------------------------------ #
    #  Step 1.5 — PCA transform → to PCA frame for SDF                   #
    # ------------------------------------------------------------------ #
    sparse_xyz_pca = math_utils.apply_pca_transform(sparse_xyz_colmap, mu, V)

    # ------------------------------------------------------------------ #
    #  Step 1.6 — SDF fine filtering in PCA frame                         #
    # ------------------------------------------------------------------ #
    from sdf import TrajectorySDF

    sdf = TrajectorySDF(
        sigma=args.sigma,
        step=args.step,
        reg=args.reg,
        dim=dim
    )
    sdf.fit(cam_pca[:, :dim])

    if args.viz and planar:
        sdf.visualize_2d()

    if len(sparse_xyz_pca) > 0:
        vals = sdf.predict(sparse_xyz_pca[:, :dim], batch_size=args.batch)
        keep = vals <= args.sdf_threshold

        sparse_data = sparse_data[keep]
        sparse_xyz_colmap = sparse_xyz_colmap[keep]
        print(f"SDF fine filter kept {sparse_data.shape[0]} / {len(vals)} points")
    else:
        print("SDF fine filter: 0 points (cloud is empty).")

    # ------------------------------------------------------------------ #
    #  Step 1.7 — Statistical Outlier Removal (COLMAP frame)              #
    # ------------------------------------------------------------------ #
    if not args.no_sor:
        sor = StatisticalOutlierFilter(
            k=args.sor_neighbors,
            threshold=args.sor_threshold,
        )
        sor_mask = sor.filter(sparse_xyz_colmap, batch_size=args.sor_batch_size)
        sparse_data = sparse_data[sor_mask]
        sparse_xyz_colmap = sparse_xyz_colmap[sor_mask]
        print(f"SOR kept {sparse_data.shape[0]} / {sor_mask.size} points")
    else:
        print("Statistical Outlier Removal: skipped (--no_sor).")

    # ------------------------------------------------------------------ #
    #  Step 1.8 — Transform filtered sparse points back to COLMAP frame   #
    # ------------------------------------------------------------------ #
    # sparse_xyz_colmap already holds the COLMAP-frame coordinates of the
    # surviving points (we tracked it alongside sparse_data), so no inverse
    # PCA transform is needed here.
    filtered_sparse_xyz = sparse_xyz_colmap

    print(f"\nStage 1 complete: {len(filtered_sparse_xyz)} sparse points survived.")

    # Optionally save the filtered sparse cloud
    if args.output_sparse_ply:
        io_utils.PlyIO.save(args.output_sparse_ply, sparse_data, as_text=False)
        print(f"Saved filtered sparse cloud: {args.output_sparse_ply}")

    # ================================================================== #
    #                     STAGE 2 — SPHERE-BASED GSPLAT FILTER            #
    # ================================================================== #
    print("\n" + "=" * 60)
    print("  STAGE 2: Sphere-Based Gaussian Splatting Filter")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    #  Step 2.1 — Load GSplat PLY                                         #
    # ------------------------------------------------------------------ #
    gsplat_data = io_utils.PlyIO.read(args.gsplat_ply)
    gsplat_xyz = _stack_xyz(gsplat_data)

    print(f"Loaded GSplat cloud: {len(gsplat_data)} points")

    # ------------------------------------------------------------------ #
    #  Step 2.2 — Sphere filter                                           #
    # ------------------------------------------------------------------ #
    sphere = SphereFilter(radius=args.sphere_radius)
    sphere.build(filtered_sparse_xyz)
    sphere_mask = sphere.filter(gsplat_xyz, batch_size=args.sphere_batch_size)

    gsplat_data = gsplat_data[sphere_mask]
    print(f"Sphere filter kept {gsplat_data.shape[0]} / {sphere_mask.size} GSplat points")

    # ------------------------------------------------------------------ #
    #  Step 2.3 — Scale filter (3DGS only)                                #
    # ------------------------------------------------------------------ #
    # Standard 3DGS PLY files store scale as log(scale) in scale_0, scale_1, scale_2
    scale_cols = ["scale_0", "scale_1", "scale_2"]
    if all(col in gsplat_data.dtype.names for col in scale_cols):
        print("Detected 3DGS scale properties. Applying scale filter...")

        # Extract scales and convert from log space
        scales = np.column_stack([gsplat_data[c] for c in scale_cols])
        actual_scales = np.exp(scales)

        # Find the maximum scale dimension for each Gaussian
        max_scales = np.max(actual_scales, axis=1)

        keep_scale = max_scales < args.scale_threshold
        gsplat_data = gsplat_data[keep_scale]
        print(f"Scale Filter removed {np.sum(~keep_scale)} oversized splats (kept {gsplat_data.shape[0]}).")
    else:
        print("No standard 3DGS scale properties found. Skipping scale filter.")

    # ================================================================== #
    #                     STAGE 3 — ANOMALY DETECTION                     #
    # ================================================================== #
    print("\n" + "=" * 60)
    print("  STAGE 3: Anomaly Detection (RX)")
    print("=" * 60)
    
    from anomaly_detector import extract_features, RRXDetector, estimate_sigma, cfar_threshold_scores, cluster_and_color_anomalies
    from sklearn.preprocessing import RobustScaler
    
    print("Extracting features (Position + Color + Opacity + Scale)...")
    X, feature_names = extract_features(gsplat_data, use_position=True, use_color=True, use_opacity=True, use_scale=True)
    print(f"Using features: {feature_names}")
    
    print("Scaling features using RobustScaler...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    print("Estimating RBF sigma...")
    sigma = estimate_sigma(X_scaled)
    print(f"Estimated sigma: {sigma:.4f}")
    
    print("Running Randomized RX Detector...")
    det = RRXDetector(n_features=300, sigma=sigma)
    scores = det.fit_score(X_scaled)
    
    print("\nApplying CFAR thresholding...")
    cfar_plot_path = args.output_ply.replace(".ply", "_cfar_plot.png")
    is_anomaly, cfar_tau, fit_info = cfar_threshold_scores(
        scores, pfa=args.pfa, n_bins=args.n_bins, plot=args.viz,
        save_path=cfar_plot_path,
    )
    
    out_anomalies_ply = args.output_ply.replace(".ply", "_only_anomalies.ply")
    anomalies_data = gsplat_data[is_anomaly]
    
    anomaly_labels = None
    if not args.no_cluster:
        print("\nClustering and coloring anomalies...")
        anomalies_data, anomaly_labels = cluster_and_color_anomalies(
            anomalies_data, eps=args.cluster_eps, min_samples=args.cluster_min_samples
        )
    else:
        print("\nSkipping clustering (--no_cluster).")

    io_utils.PlyIO.save(out_anomalies_ply, anomalies_data, as_text=False)
    print(f"Saved {len(anomalies_data)} anomalies to separate file: {out_anomalies_ply}")
    
    import numpy.lib.recfunctions as rfn
    # Append the anomaly score to the structured array for inspection in 3D viewers (like CloudCompare)
    gsplat_data = rfn.append_fields(gsplat_data, 'anomaly_score', scores.astype(np.float32), usemask=False, asrecarray=False)
    
    print(f"Anomaly detection complete. No points were removed; anomaly_score field added. Total points: {len(gsplat_data)}.")

    # ================================================================== #
    #  Save Output                                                        #
    # ================================================================== #
    io_utils.PlyIO.save(args.output_ply, gsplat_data, as_text=False)
    print(f"\nSaved main PLY with anomaly scores: {args.output_ply}")

    return {
        "gsplat_data": gsplat_data,
        "anomalies_data": anomalies_data,
        "anomaly_labels": anomaly_labels,
        "anomaly_scores_all": scores,
        "anomaly_scores_anomalies": scores[is_anomaly],
        "is_anomaly": is_anomaly,
        "output_ply": args.output_ply,
        "output_anomalies_ply": out_anomalies_ply,
    }


def main():
    """CLI entry point — parse args and run the pipeline."""
    ap = _build_arg_parser()
    args = ap.parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    main()
