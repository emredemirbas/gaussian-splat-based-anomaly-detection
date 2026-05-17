"""
Unit tests for ellipsoid_filter module.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ellipsoid_filter import EllipsoidFilter


class TestEllipsoidFilter:

    @pytest.fixture
    def default_filter(self):
        return EllipsoidFilter(inflate_factor=1.0, use_percentile=False, min_radius=0.0)

    def test_points_inside_unit_sphere(self, default_filter):
        """Points at origin should be inside any camera-fitted ellipsoid."""
        cams = np.array([[1, 0, 0], [-1, 0, 0],
                         [0, 1, 0], [0, -1, 0],
                         [0, 0, 1], [0, 0, -1]], dtype=np.float64)
        pts = np.array([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=np.float64)
        mask, info = default_filter.filter(pts, cams)
        assert mask.all()

    def test_points_outside_are_rejected(self, default_filter):
        """Points far beyond camera extent should be rejected."""
        cams = np.array([[1, 0, 0], [-1, 0, 0],
                         [0, 1, 0], [0, -1, 0]], dtype=np.float64)
        pts = np.array([[100.0, 100.0, 100.0]], dtype=np.float64)
        mask, _ = default_filter.filter(pts, cams)
        assert not mask.any()

    def test_inflate_factor_expands_boundary(self):
        """Inflated filter should accept points the tight filter rejects."""
        cams = np.array([[1, 0, 0], [-1, 0, 0],
                         [0, 1, 0], [0, -1, 0],
                         [0, 0, 1], [0, 0, -1]], dtype=np.float64)

        # Point just outside the unit sphere
        pts = np.array([[1.05, 0, 0]], dtype=np.float64)

        tight = EllipsoidFilter(inflate_factor=1.0, use_percentile=False, min_radius=0.0)
        loose = EllipsoidFilter(inflate_factor=1.2, use_percentile=False, min_radius=0.0)

        mask_tight, _ = tight.filter(pts, cams)
        mask_loose, _ = loose.filter(pts, cams)

        assert not mask_tight[0]
        assert mask_loose[0]

    def test_min_radius_prevents_collapse(self):
        """min_radius should prevent degenerate axes from collapsing to zero."""
        # Cameras only along X axis → Y and Z extents are 0
        cams = np.array([[5, 0, 0], [-5, 0, 0]], dtype=np.float64)
        pts = np.array([[0, 0.3, 0]], dtype=np.float64)

        f = EllipsoidFilter(inflate_factor=1.0, use_percentile=False, min_radius=0.5)
        mask, info = f.filter(pts, cams)
        assert mask[0], "Point within min_radius should be kept"
        assert info["radii"][1] >= 0.5

    def test_output_shapes(self):
        cams = np.random.default_rng(0).standard_normal((20, 3))
        pts = np.random.default_rng(1).standard_normal((500, 3))
        f = EllipsoidFilter()
        mask, info = f.filter(pts, cams)
        assert mask.shape == (500,)
        assert mask.dtype == bool
        assert info["radii"].shape == (3,)

    def test_fewer_than_2_cameras_keeps_all(self):
        """Edge case: single camera → keep everything."""
        cams = np.array([[1, 2, 3]], dtype=np.float64)
        pts = np.random.default_rng(0).standard_normal((10, 3))
        f = EllipsoidFilter()
        mask, _ = f.filter(pts, cams)
        assert mask.all()
