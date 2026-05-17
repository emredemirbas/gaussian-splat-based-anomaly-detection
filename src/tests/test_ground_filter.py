"""
Unit tests for ground_filter module.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ground_filter import GroundFilter


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _make_flat_ground_cloud(n_ground=500, n_objects=100, ground_y=10.0, seed=42):
    """
    Synthetic cloud with a flat ground plane at y = ground_y (COLMAP Y-down)
    and objects floating above it (lower y values).
    """
    rng = np.random.default_rng(seed)

    # Ground points: spread in X-Z, y ≈ ground_y (high y = low in world)
    gx = rng.uniform(-5, 5, n_ground)
    gy = ground_y + rng.normal(0, 0.02, n_ground)
    gz = rng.uniform(-5, 5, n_ground)
    ground = np.column_stack([gx, gy, gz])

    # Object points: y << ground_y
    ox = rng.uniform(-3, 3, n_objects)
    oy = rng.uniform(ground_y - 8, ground_y - 2, n_objects)
    oz = rng.uniform(-3, 3, n_objects)
    objects = np.column_stack([ox, oy, oz])

    return np.vstack([ground, objects])


class TestGroundFilterInternals:

    def test_plane_from_three_points_nondegenerate(self):
        p1 = np.array([0, 0, 0], dtype=np.float64)
        p2 = np.array([1, 0, 0], dtype=np.float64)
        p3 = np.array([0, 1, 0], dtype=np.float64)
        result = GroundFilter._plane_from_three_points(p1, p2, p3)
        assert result is not None
        A, B, C, D = result
        # Normal should be along Z
        assert abs(C) > abs(A) and abs(C) > abs(B)

    def test_plane_from_three_collinear_returns_none(self):
        p1 = np.array([0, 0, 0], dtype=np.float64)
        p2 = np.array([1, 0, 0], dtype=np.float64)
        p3 = np.array([2, 0, 0], dtype=np.float64)
        assert GroundFilter._plane_from_three_points(p1, p2, p3) is None

    def test_normalize_plane(self):
        A, B, C, D = GroundFilter._normalize_plane(0, 0, 5, -10)
        assert abs(C - 1.0) < 1e-12
        assert abs(D - (-2.0)) < 1e-12

    def test_slope_is_valid(self):
        gf = GroundFilter(vertical_axis=1, slope_threshold=0.9)
        # Near-vertical normal along Y → valid
        assert gf._slope_is_valid(0.0, 0.99, 0.0) is True
        # Tilted normal → invalid
        assert gf._slope_is_valid(0.7, 0.7, 0.1) is False


class TestGroundFilterPipeline:

    def test_flat_ground_removal_y_down(self):
        """With a clear flat ground at high Y values, ground filter should remove it."""
        pts = _make_flat_ground_cloud(n_ground=500, n_objects=100)
        gf = GroundFilter(
            vertical_axis=1,
            axis_points_up=False,
            percentile=5.0,
            ground_margin=0.25,
            ransac_distance_threshold=0.10,
            slope_threshold=0.8,
            inlier_ratio_threshold=0.10,
            ransac_iterations=500,
        )
        non_ground, ground, plane, mask = gf.filter_ground(pts)

        # We should remove a substantial fraction of ground
        assert ground.shape[0] > 300, f"Expected to remove >300 ground pts, got {ground.shape[0]}"
        # And keep most object points
        assert non_ground.shape[0] >= 50, f"Expected to keep >=50 object pts, got {non_ground.shape[0]}"

    def test_mask_consistency(self):
        """ground_mask should partition points into ground + non_ground."""
        pts = _make_flat_ground_cloud()
        gf = GroundFilter(vertical_axis=1, axis_points_up=False)
        non_ground, ground, _, mask = gf.filter_ground(pts)
        assert np.sum(mask) == ground.shape[0]
        assert np.sum(~mask) == non_ground.shape[0]
        assert ground.shape[0] + non_ground.shape[0] == pts.shape[0]

    def test_too_few_points(self):
        """Fewer than 3 points → no ground removal."""
        pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        gf = GroundFilter()
        non_ground, ground, _, mask = gf.filter_ground(pts)
        assert non_ground.shape[0] == 2
        assert ground.shape[0] == 0

    def test_z_up_axis(self):
        """Ground filter should work with Z-up convention too."""
        rng = np.random.default_rng(99)
        n = 400
        # Ground at z ≈ 0, objects at z ≈ 5
        gx = rng.uniform(-5, 5, n)
        gy = rng.uniform(-5, 5, n)
        gz = rng.normal(0, 0.02, n)
        ground = np.column_stack([gx, gy, gz])

        ox = rng.uniform(-3, 3, 100)
        oy = rng.uniform(-3, 3, 100)
        oz = rng.uniform(3, 8, 100)
        objects = np.column_stack([ox, oy, oz])

        pts = np.vstack([ground, objects])
        gf = GroundFilter(
            vertical_axis=2,
            axis_points_up=True,
            percentile=5.0,
            ground_margin=0.25,
            ransac_iterations=500,
            slope_threshold=0.8,
            inlier_ratio_threshold=0.10,
        )
        _, ground_pts, _, _ = gf.filter_ground(pts)
        assert ground_pts.shape[0] > 200
