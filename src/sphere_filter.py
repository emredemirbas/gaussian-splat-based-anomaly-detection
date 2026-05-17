import numpy as np
from scipy.spatial import cKDTree


class SphereFilter:
    """
    Sphere-based point cloud filter using a KD-tree for efficient queries.

    Given a set of centre points (e.g. filtered sparse reconstruction) and a
    sphere radius, this filter keeps only those query points (e.g. Gaussian
    Splats) that lie inside at least one sphere.

    Algorithm
    ---------
    1. Build a ``cKDTree`` from the centre positions.
    2. For each query point, find whether any centre exists within
       ``radius`` using ``query_ball_point``.
    3. Return a boolean mask: True if the point has ≥1 neighbour.

    Parameters
    ----------
    radius : float
        Radius of each sphere (same units as the point coordinates).
    """

    def __init__(self, radius: float = 0.5):
        self.radius = float(radius)
        self._tree: cKDTree | None = None
        self._centers: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def build(self, centers: np.ndarray) -> None:
        """
        Build the KD-tree from sphere centre positions.

        Parameters
        ----------
        centers : (N, 3) float array
            Positions of the sphere centres (e.g. filtered sparse points
            in COLMAP frame).
        """
        C = np.asarray(centers, dtype=np.float64)
        if C.ndim != 2 or C.shape[1] != 3:
            raise ValueError("centers must be (N, 3).")
        self._centers = C
        self._tree = cKDTree(C)
        print(
            f"[SphereFilter] Built KD-tree from {C.shape[0]} centres "
            f"(radius = {self.radius:.4f})"
        )

    def filter(
        self,
        query_points: np.ndarray,
        batch_size: int = 100_000,
    ) -> np.ndarray:
        """
        Test which query points lie inside at least one sphere.

        The query is executed in batches of ``batch_size`` points to keep
        peak memory usage bounded (important for multi-million-point
        Gaussian Splatting clouds).

        Parameters
        ----------
        query_points : (M, 3) float array
            Points to test (e.g. GSplat positions in COLMAP frame).
        batch_size : int
            Number of query points processed per batch.  Default 100 000.

        Returns
        -------
        mask : (M,) bool array
            True for every query point inside at least one sphere.
        """
        if self._tree is None:
            raise RuntimeError("KD-tree not built. Call build() first.")

        Q = np.asarray(query_points, dtype=np.float64)
        if Q.ndim != 2 or Q.shape[1] != 3:
            raise ValueError("query_points must be (M, 3).")

        M = Q.shape[0]
        mask = np.empty(M, dtype=bool)

        n_batches = max(1, (M + batch_size - 1) // batch_size)

        for b in range(n_batches):
            lo = b * batch_size
            hi = min(lo + batch_size, M)
            neighbours = self._tree.query_ball_point(
                Q[lo:hi], self.radius, workers=-1,
            )
            mask[lo:hi] = np.array(
                [len(nb) > 0 for nb in neighbours], dtype=bool,
            )

            if n_batches > 1:
                print(
                    f"[SphereFilter] Batch {b + 1}/{n_batches}  "
                    f"({lo}–{hi})"
                )

        inside = int(mask.sum())
        outside = M - inside
        print(
            f"[SphereFilter] Inside: {inside}  |  Outside: {outside}  "
            f"(total: {M})"
        )
        return mask
