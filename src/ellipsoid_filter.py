import numpy as np


class EllipsoidFilter:
    """
    PCA-based ellipsoid coarse filter for background removal.

    Removes scene points that lie outside an ellipsoid fitted to camera
    positions.  Both camera centres and scene points must be supplied in
    the same PCA-aligned frame (computed externally).

    The filter only computes semi-axis radii from the camera extents and
    performs the inclusion test — it does **not** compute its own PCA.

    Algorithm (given PCA-frame inputs)
    -----------------------------------
    1. Compute semi-axis radii from camera extents (percentile or max),
       inflated by ``inflate_factor``.
    2. Evaluate the ellipsoid inclusion test  φ(p) ≤ 1.
    3. Return a boolean mask of kept points.

    Parameters
    ----------
    inflate_factor : float
        Multiplicative expansion applied to each semi-axis radius.
        Default 1.15  (15 % larger than the camera extent).
    use_percentile : bool
        If True, radii are derived from a percentile of absolute PCA
        coordinates (robust to outlier cameras).  If False, the maximum
        absolute coordinate is used instead.  Default True.
    percentile : float
        Percentile value when ``use_percentile`` is True.  Default 95.
    min_radius : float
        Minimum allowed semi-axis radius (avoids degenerate collapse for
        nearly coplanar / collinear camera arrangements).  Default 0.5.
    """

    def __init__(
        self,
        inflate_factor: float = 1.15,
        use_percentile: bool = True,
        percentile: float = 95.0,
        min_radius: float = 0.5,
    ):
        self.inflate_factor = float(inflate_factor)
        self.use_percentile = bool(use_percentile)
        self.percentile = float(percentile)
        self.min_radius = float(min_radius)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def filter(self, points_pca: np.ndarray, camera_centers_pca: np.ndarray, dim: int = 3):
        """
        Classify scene points as inside / outside the camera ellipsoid.

        Both inputs must already be in the **same PCA-aligned frame**
        (centred and rotated by the eigenvectors computed externally).

        Parameters
        ----------
        points_pca : (M, 3) float array
            Scene points in PCA frame.
        camera_centers_pca : (N, 3) float array
            Camera positions in PCA frame.
        dim : int
            If 2, the filter becomes a 2D elliptical cylinder, ignoring
            the 3rd PCA axis. Default is 3 (full ellipsoid).

        Returns
        -------
        mask : (M,) bool array
            True for every point inside (or on) the ellipsoid.
        info : dict
            Diagnostic information: ``radii``.
        """
        P = np.asarray(points_pca, dtype=np.float64)
        C = np.asarray(camera_centers_pca, dtype=np.float64)

        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError("points_pca must be (M, 3).")
        if C.ndim != 2 or C.shape[1] != 3:
            raise ValueError("camera_centers_pca must be (N, 3).")

        M = P.shape[0]

        if C.shape[0] < 2:
            print("[EllipsoidFilter] WARNING: fewer than 2 cameras — keeping all points.")
            return np.ones(M, dtype=bool), {}

        # ---- Compute semi-axis radii --------------------------------- #
        abs_C = np.abs(C)

        if self.use_percentile:
            raw_radii = np.percentile(abs_C, self.percentile, axis=0)  # (3,)
        else:
            raw_radii = abs_C.max(axis=0)  # (3,)

        # Inflate and clamp to minimum radius
        radii = np.maximum(raw_radii * self.inflate_factor, self.min_radius)
        a, b, c = radii

        if dim == 2:
            c = np.inf

        # ---- Ellipsoid inclusion test -------------------------------- #
        vals = (P[:, 0] ** 2) / (a * a) + \
               (P[:, 1] ** 2) / (b * b) + \
               (P[:, 2] ** 2) / (c * c)

        mask = vals <= 1.0

        # ---- Diagnostics -------------------------------------------- #
        info = {"radii": radii}

        inside_count = int(mask.sum())
        outside_count = M - inside_count
        print(
            f"[EllipsoidFilter] Radii: ({a:.3f}, {b:.3f}, {c:.3f})  |  "
            f"Inside: {inside_count}  |  Outside: {outside_count}"
        )

        return mask, info
