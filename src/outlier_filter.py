import numpy as np
from scipy.spatial import cKDTree


class StatisticalOutlierFilter:
    """
    Locally-adaptive Statistical Outlier Removal for 3-D point clouds.

    For every point the mean Euclidean distance to its *k* nearest
    neighbours is computed.  Instead of comparing against a *global*
    threshold (which fails on multi-density clouds), each point's
    mean-distance is compared to the **average mean-distance of its own
    neighbours** — a simplified Local Outlier Factor (LOF).

    A point is flagged as an outlier when::

        mean_dist[i] / mean(mean_dist[neighbours_of_i]) > threshold

    A ratio of 1.0 means the point is exactly as dense as its
    neighbourhood.  Values above 1.0 indicate increasing isolation
    *relative to the local context*.

    This handles clouds with varying density naturally: a point in a
    globally sparse region is kept as long as its neighbours are
    equally sparse.

    Parameters
    ----------
    k : int
        Number of nearest neighbours to consider (excluding the query
        point itself).
    threshold : float
        Maximum allowed density ratio.  Points whose local ratio exceeds
        this value are removed.  Typical range: 1.2 – 3.0 (default 1.5).
    """

    def __init__(self, k: int = 50, threshold: float = 1.5):
        if k < 1:
            raise ValueError("k must be >= 1.")
        self.k = int(k)
        self.threshold = float(threshold)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def filter(
        self,
        points: np.ndarray,
        batch_size: int = 100_000,
    ) -> np.ndarray:
        """
        Return a boolean mask that is *True* for inliers and *False* for
        outliers.

        Parameters
        ----------
        points : (N, 3) float array
            The point cloud to filter.
        batch_size : int
            Number of points queried per KD-tree batch to limit peak
            memory.

        Returns
        -------
        mask : (N,) bool array
            True for every inlier point.
        """
        P = np.asarray(points, dtype=np.float64)
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError("points must be (N, 3).")

        N = P.shape[0]

        # Edge case: too few points to have k neighbours
        if N <= self.k:
            print(
                f"[SOR] Only {N} points but k={self.k}; "
                f"skipping outlier removal (keeping all)."
            )
            return np.ones(N, dtype=bool)

        tree = cKDTree(P)

        # k+1 because query includes the point itself (distance 0)
        k_query = self.k + 1

        mean_dists = np.empty(N, dtype=np.float64)
        neighbor_indices = np.empty((N, self.k), dtype=np.intp)

        n_batches = max(1, (N + batch_size - 1) // batch_size)

        for b in range(n_batches):
            lo = b * batch_size
            hi = min(lo + batch_size, N)
            dists, idxs = tree.query(P[lo:hi], k=k_query, workers=-1)
            # dists[:, 0] is always 0 (self-distance); skip it
            mean_dists[lo:hi] = dists[:, 1:].mean(axis=1)
            neighbor_indices[lo:hi] = idxs[:, 1:]

            if n_batches > 1:
                print(
                    f"[SOR] KNN batch {b + 1}/{n_batches}  "
                    f"({lo}–{hi})"
                )

        # ---- Local density ratio (simplified LOF) ---- #
        # For each point, average its neighbours' mean distances
        local_ref = mean_dists[neighbor_indices].mean(axis=1)

        # ratio > 1  →  point is more isolated than its local context
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(local_ref > 0, mean_dists / local_ref, 1.0)

        mask = ratio <= self.threshold
        n_removed = int((~mask).sum())

        print(
            f"[SOR] Local density ratio — threshold={self.threshold:.2f}"
        )
        print(
            f"[SOR] Ratio stats: "
            f"min={ratio.min():.3f}  median={np.median(ratio):.3f}  "
            f"max={ratio.max():.3f}"
        )
        print(
            f"[SOR] Removed {n_removed} outliers  "
            f"(kept {int(mask.sum())} / {N})"
        )

        return mask
