"""
Compute metadata for each anomaly cluster: centroid, AABB bounding box,
and a camera view pose derived from the nearest COLMAP camera.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class ClusterInfo:
    """Metadata for a single anomaly cluster."""
    cluster_id: int
    centroid: np.ndarray       # (3,) world-space centre of mass
    bbox_min: np.ndarray       # (3,) AABB minimum corner
    bbox_max: np.ndarray       # (3,) AABB maximum corner
    view_position: np.ndarray  # (3,) camera position for viewing this cluster
    view_target: np.ndarray    # (3,) look-at target (= centroid)
    view_up: np.ndarray        # (3,) up vector
    n_points: int = 0
    mean_score: float = 0.0


def compute_cluster_metadata(
    gsplat_xyz: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    cam_centers: np.ndarray,
    view_distance_factor: float = 0.35,
) -> list[ClusterInfo]:
    """
    For each DBSCAN cluster (label >= 0), compute:
      - centroid  (mean XYZ of anomalous splats in that cluster)
      - AABB      (min/max XYZ)
      - view pose (derived from the nearest COLMAP camera, moved closer
        along the camera→centroid direction for a tighter, straighter view)

    Parameters
    ----------
    gsplat_xyz : (N, 3) float64
        Positions of the *anomalous* GSplat points (already filtered to
        anomalies only, matching ``labels``).
    labels : (N,) int
        DBSCAN labels for each anomalous point.  -1 = noise, >=0 = cluster.
    scores : (N,) float64
        Anomaly scores for each anomalous point.
    cam_centers : (M, 3) float64
        COLMAP camera centres in world coordinates.
    view_distance_factor : float
        The camera is placed at this fraction of the original
        camera-to-centroid distance.  E.g. 0.35 means 35 % of the way
        from the original camera position to the centroid (i.e. much
        closer).  A minimum distance of 1.5x the cluster bounding-box
        diagonal is enforced to prevent clipping into the cluster.

    Returns
    -------
    clusters : list of ClusterInfo, sorted by cluster_id ascending.
    """
    gsplat_xyz = np.asarray(gsplat_xyz, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    cam_centers = np.asarray(cam_centers, dtype=np.float64)

    unique_labels = sorted(set(labels[labels >= 0]))
    clusters: list[ClusterInfo] = []

    # COLMAP convention: Y points downward
    up = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    for cid in unique_labels:
        mask = labels == cid
        pts = gsplat_xyz[mask]
        sc = scores[mask]

        centroid = pts.mean(axis=0)
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))

        # Find the nearest COLMAP camera to this cluster centroid
        dists = np.linalg.norm(cam_centers - centroid, axis=1)
        nearest_idx = int(np.argmin(dists))
        cam_pos = cam_centers[nearest_idx].copy()

        # Direction from camera toward centroid
        direction = centroid - cam_pos
        orig_dist = float(np.linalg.norm(direction))

        if orig_dist < 1e-8:
            # Degenerate: camera sits on the centroid — keep as-is
            view_position = cam_pos
        else:
            direction /= orig_dist  # unit vector

            # Desired distance: fraction of original, but at least
            # 1.5x bbox diagonal so we don't clip into the cluster
            desired_dist = max(orig_dist * view_distance_factor,
                               bbox_diag * 1.5)
            # Never go further than the original camera
            desired_dist = min(desired_dist, orig_dist)

            # Place camera along the same line, but closer to centroid
            view_position = centroid - direction * desired_dist

        clusters.append(ClusterInfo(
            cluster_id=cid,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            view_position=view_position,
            view_target=centroid.copy(),
            view_up=up.copy(),
            n_points=int(mask.sum()),
            mean_score=float(sc.mean()),
        ))

    print(f"[ClusterMetadata] Computed metadata for {len(clusters)} clusters")
    return clusters
