import numpy as np


class GroundFilter:
    """
    Robust ground filtering for 3D point clouds using a custom RANSAC
    implementation with geometric constraints.

    Algorithm Overview
    ------------------
    1. **Pre-processing & Normalization**: Candidate planes are normalized by
       their normal length, and the normal is forced to point upward.
    2. **RANSAC with Geometric Constraints**: An initial percentile-based
       censoring step selects candidate ground seeds. A RANSAC loop samples
       3-point subsets, fits planes, and rejects any plane whose normal
       deviates too far from vertical (slope test: |C̃| ≤ s).  The best
       plane is chosen by inlier count / lowest residual variance.
    3. **Point Classification via Signed Distance**: Instead of unsigned
       distance, the signed point-to-plane distance d_i = Ã·x + B̃·y +
       C̃·z + D̃ is used.  Points with d_i > g (safety margin) are kept as
       objects; points with d_i ≤ g are classified as ground.

    Parameters
    ----------
    ground_margin : float
        Safety margin *g* (metres).  Points with signed distance ≤ g above
        the fitted plane are classified as ground.  Default 0.25 m.
    slope_threshold : float
        Minimum absolute vertical component |C̃| of the normalised plane
        normal.  Planes with |C̃| ≤ s are rejected as too steep.
        Default 0.9  (≈ 25.8° maximum tilt).
    ransac_iterations : int
        Number of RANSAC iterations.  Default 1000.
    ransac_distance_threshold : float
        Maximum orthogonal distance (metres) for a point to be counted as
        an inlier during RANSAC.  Default 0.10 m.
    inlier_ratio_threshold : float
        Minimum fraction of seed points that must be inliers for a plane
        to be accepted.  Default 0.15.
    percentile : float
        Percentile used for initial censoring of candidate ground points.
        Default 5.0 (bottom 5 % of the vertical axis).
    vertical_axis : int
        Index of the vertical axis (0 = X, 1 = Y, 2 = Z).  Default 1.
    axis_points_up : bool
        If True the vertical axis grows upward (e.g. Z-up); if False it
        grows downward (e.g. raw COLMAP Y axis).  Default False.
    random_state : int or None
        Seed for the random number generator.  Default 42.
    """

    def __init__(
        self,
        ground_margin: float = 0.25,
        slope_threshold: float = 0.9,
        ransac_iterations: int = 1000,
        ransac_distance_threshold: float = 0.10,
        inlier_ratio_threshold: float = 0.15,
        percentile: float = 5.0,
        vertical_axis: int = 1,
        axis_points_up: bool = False,
        random_state: int = 42,
    ):
        self.ground_margin = float(ground_margin)
        self.slope_threshold = float(slope_threshold)
        self.ransac_iterations = int(ransac_iterations)
        self.ransac_distance_threshold = float(ransac_distance_threshold)
        self.inlier_ratio_threshold = float(inlier_ratio_threshold)
        self.percentile = float(percentile)
        self.vertical_axis = int(vertical_axis)
        self.axis_points_up = bool(axis_points_up)
        self.random_state = random_state

    # --------------------------------------------------------------------- #
    #  Internal helpers                                                      #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _plane_from_three_points(p1: np.ndarray,
                                 p2: np.ndarray,
                                 p3: np.ndarray):
        """
        Compute plane equation Ax + By + Cz + D = 0 from three 3-D points.

        Returns (A, B, C, D) or None if the points are (near-)collinear.
        """
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-12:
            return None  # degenerate (collinear) sample
        return (*normal, -normal.dot(p1))

    @staticmethod
    def _normalize_plane(A: float, B: float, C: float, D: float):
        """
        Normalize plane coefficients so that (A, B, C) becomes a unit normal.

        Step 1 of the algorithm:  divide by ‖n‖ = √(A² + B² + C²).
        """
        n = np.sqrt(A * A + B * B + C * C)
        if n < 1e-15:
            return A, B, C, D  # degenerate – shouldn't happen after the cross-product check
        return A / n, B / n, C / n, D / n

    def _ensure_upward_normal(self, A: float, B: float, C: float, D: float):
        """
        Ensure the plane normal points *upward* with respect to the
        configured vertical axis.

        If the component of the normal along the vertical axis is negative
        (or, when ``axis_points_up`` is False, positive – because ground is
        at *high* values of that axis), flip the whole plane.
        """
        # The component of the normal along the vertical axis
        vertical_component = [A, B, C][self.vertical_axis]

        if self.axis_points_up:
            # For an upward-pointing axis: normal should go in +axis direction
            # (i.e. vertical_component > 0).
            if vertical_component < 0:
                return -A, -B, -C, -D
        else:
            # For a downward-pointing axis (e.g. COLMAP Y): the ground is at
            # *high* values of the axis.  A normal pointing "away from ground"
            # (= upward in world) corresponds to the *negative* axis direction,
            # so we want vertical_component < 0.
            if vertical_component > 0:
                return -A, -B, -C, -D

        return A, B, C, D

    def _slope_is_valid(self, A: float, B: float, C: float) -> bool:
        """
        Slope validation: reject a plane whose normal is too far from
        vertical.

        The test checks |n_vertical| > slope_threshold.  For a perfectly
        horizontal ground plane n_vertical = ±1.
        """
        vertical_component = abs([A, B, C][self.vertical_axis])
        return vertical_component > self.slope_threshold

    # --------------------------------------------------------------------- #
    #  Public API                                                            #
    # --------------------------------------------------------------------- #

    def filter_ground(self, points: np.ndarray):
        """
        Classify a point cloud into *ground* and *non-ground* (object) points.

        Parameters
        ----------
        points : (N, 3) float array
            Input point cloud.

        Returns
        -------
        non_ground_points : (M, 3) float array
            Points classified as objects.
        ground_points : (K, 3) float array
            Points classified as ground.
        plane_params : tuple (A, B, C, D)
            Normalised plane coefficients of the best-fit ground plane.
        ground_mask : (N,) bool array
            True for every point classified as ground.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("Points must be (N, 3).")

        N = len(pts)

        # Trivial / degenerate cases
        if N < 3:
            return (
                pts.copy(),
                np.empty((0, 3), dtype=np.float64),
                (0.0, 0.0, 0.0, 0.0),
                np.zeros(N, dtype=bool),
            )

        # ----------------------------------------------------------------- #
        #  Step 2a – Initial censoring: select candidate ground seed points  #
        # ----------------------------------------------------------------- #
        vert_values = pts[:, self.vertical_axis]

        if self.axis_points_up:
            # Ground = lowest values
            thresh = np.percentile(vert_values, self.percentile)
            seed_mask = vert_values <= thresh
        else:
            # Ground = highest values (e.g. COLMAP Y, which grows downward)
            thresh = np.percentile(vert_values, 100.0 - self.percentile)
            seed_mask = vert_values >= thresh

        seed_points = pts[seed_mask]

        if len(seed_points) < 3:
            return (
                pts.copy(),
                np.empty((0, 3), dtype=np.float64),
                (0.0, 0.0, 0.0, 0.0),
                np.zeros(N, dtype=bool),
            )

        # ----------------------------------------------------------------- #
        #  Step 2b – RANSAC with geometric constraints                       #
        # ----------------------------------------------------------------- #
        rng = np.random.default_rng(self.random_state)
        n_seeds = len(seed_points)

        best_plane = None          # (A, B, C, D) normalised & upward
        best_inlier_count = 0
        best_variance = np.inf

        for _ in range(self.ransac_iterations):
            # Sample 3 distinct seed points
            idx = rng.choice(n_seeds, size=3, replace=False)
            p1, p2, p3 = seed_points[idx]

            # Fit plane through the 3 points
            result = self._plane_from_three_points(p1, p2, p3)
            if result is None:
                continue  # degenerate sample

            A, B, C, D = result

            # Step 1 – Normalise
            A, B, C, D = self._normalize_plane(A, B, C, D)

            # Step 1 – Ensure upward orientation
            A, B, C, D = self._ensure_upward_normal(A, B, C, D)

            # Step 2 – Slope validation:  reject if |C̃_vertical| ≤ s
            if not self._slope_is_valid(A, B, C):
                continue

            # Compute signed distances of *seed* points to this plane (vectorised)
            signed_dists = (
                A * seed_points[:, 0]
                + B * seed_points[:, 1]
                + C * seed_points[:, 2]
                + D
            )

            # Inliers: points within ±ransac_distance_threshold of the plane
            inlier_mask = np.abs(signed_dists) <= self.ransac_distance_threshold
            inlier_count = int(inlier_mask.sum())
            inlier_ratio = inlier_count / n_seeds

            # Quality gate: minimum inlier ratio
            if inlier_ratio < self.inlier_ratio_threshold:
                continue

            # Residual variance of inliers
            if inlier_count > 0:
                variance = float(np.var(signed_dists[inlier_mask]))
            else:
                variance = np.inf

            # Keep the best plane (most inliers; break ties with lowest variance)
            if (inlier_count > best_inlier_count) or (
                inlier_count == best_inlier_count and variance < best_variance
            ):
                best_inlier_count = inlier_count
                best_variance = variance
                best_plane = (A, B, C, D)

        # If RANSAC found no valid plane, return everything as non-ground
        if best_plane is None:
            print("[GroundFilter] WARNING: No valid ground plane found.")
            return (
                pts.copy(),
                np.empty((0, 3), dtype=np.float64),
                (0.0, 0.0, 0.0, 0.0),
                np.zeros(N, dtype=bool),
            )

        A, B, C, D = best_plane

        # ----- Optional refinement: refit on all inliers of the best plane ----- #
        # Compute signed distances for ALL points to the RANSAC plane
        all_signed = A * pts[:, 0] + B * pts[:, 1] + C * pts[:, 2] + D
        inlier_mask_all = np.abs(all_signed) <= self.ransac_distance_threshold
        inlier_pts = pts[inlier_mask_all]

        if len(inlier_pts) >= 3:
            # Least-squares plane refit on all inliers
            # Solve:  [x y z 1] · [A B C D]^T = 0  in least-squares sense
            # Equivalently, find n that minimises  ‖X n‖  with ‖n‖=1
            centroid = inlier_pts.mean(axis=0)
            centered = inlier_pts - centroid
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            normal = Vt[-1]  # last row = smallest singular value direction

            A_r, B_r, C_r = normal
            D_r = -normal.dot(centroid)

            # Normalise (should already be unit, but be safe)
            A_r, B_r, C_r, D_r = self._normalize_plane(A_r, B_r, C_r, D_r)
            A_r, B_r, C_r, D_r = self._ensure_upward_normal(A_r, B_r, C_r, D_r)

            if self._slope_is_valid(A_r, B_r, C_r):
                A, B, C, D = A_r, B_r, C_r, D_r

        # ----------------------------------------------------------------- #
        #  Step 3 – Point classification via signed distance                 #
        # ----------------------------------------------------------------- #
        signed_distances = A * pts[:, 0] + B * pts[:, 1] + C * pts[:, 2] + D

        # Classification rule:
        #   d_i > g   →  Object  (above ground by more than the margin)
        #   d_i ≤ g   →  Ground  (on or below the plane, within margin)
        ground_mask = signed_distances <= self.ground_margin

        non_ground_points = pts[~ground_mask]
        ground_points = pts[ground_mask]

        print(
            f"[GroundFilter] Plane: ({A:.4f}, {B:.4f}, {C:.4f}, {D:.4f})  |  "
            f"Ground: {ground_points.shape[0]}  |  Objects: {non_ground_points.shape[0]}"
        )

        return non_ground_points, ground_points, (A, B, C, D), ground_mask