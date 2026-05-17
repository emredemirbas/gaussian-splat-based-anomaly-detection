"""
Serialize / deserialize ClusterInfo metadata to .npz files.

This allows the pipeline and viewer to be run separately if needed
(e.g. pipeline on a server, viewer on a workstation).
"""

import numpy as np
from cluster_metadata import ClusterInfo


def save_clusters(path: str, clusters: list[ClusterInfo]) -> None:
    """
    Save a list of ClusterInfo to a .npz file.

    Parameters
    ----------
    path : str
        Output file path (should end in .npz).
    clusters : list of ClusterInfo
    """
    n = len(clusters)
    cluster_ids = np.array([c.cluster_id for c in clusters], dtype=np.int64)
    centroids = np.array([c.centroid for c in clusters], dtype=np.float64)
    bbox_mins = np.array([c.bbox_min for c in clusters], dtype=np.float64)
    bbox_maxs = np.array([c.bbox_max for c in clusters], dtype=np.float64)
    view_positions = np.array([c.view_position for c in clusters], dtype=np.float64)
    view_targets = np.array([c.view_target for c in clusters], dtype=np.float64)
    view_ups = np.array([c.view_up for c in clusters], dtype=np.float64)
    n_points = np.array([c.n_points for c in clusters], dtype=np.int64)
    mean_scores = np.array([c.mean_score for c in clusters], dtype=np.float64)

    np.savez(
        path,
        cluster_ids=cluster_ids,
        centroids=centroids,
        bbox_mins=bbox_mins,
        bbox_maxs=bbox_maxs,
        view_positions=view_positions,
        view_targets=view_targets,
        view_ups=view_ups,
        n_points=n_points,
        mean_scores=mean_scores,
    )
    print(f"[ClusterIO] Saved {n} clusters to {path}")


def load_clusters(path: str) -> list[ClusterInfo]:
    """
    Load a list of ClusterInfo from a .npz file.

    Parameters
    ----------
    path : str
        Input file path (.npz).

    Returns
    -------
    clusters : list of ClusterInfo
    """
    data = np.load(path)
    n = len(data["cluster_ids"])
    clusters = []

    for i in range(n):
        clusters.append(ClusterInfo(
            cluster_id=int(data["cluster_ids"][i]),
            centroid=data["centroids"][i].copy(),
            bbox_min=data["bbox_mins"][i].copy(),
            bbox_max=data["bbox_maxs"][i].copy(),
            view_position=data["view_positions"][i].copy(),
            view_target=data["view_targets"][i].copy(),
            view_up=data["view_ups"][i].copy(),
            n_points=int(data["n_points"][i]),
            mean_score=float(data["mean_scores"][i]),
        ))

    print(f"[ClusterIO] Loaded {len(clusters)} clusters from {path}")
    return clusters
