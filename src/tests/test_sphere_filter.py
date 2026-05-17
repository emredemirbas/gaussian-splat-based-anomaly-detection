"""
Unit tests for sphere_filter module.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from sphere_filter import SphereFilter


class TestSphereFilter:
    """Tests for the KD-tree-based sphere filter."""

    def test_point_inside_single_sphere(self):
        """A point at the centre of a sphere must be kept."""
        centers = np.array([[0.0, 0.0, 0.0]])
        sf = SphereFilter(radius=1.0)
        sf.build(centers)

        queries = np.array([[0.0, 0.0, 0.0]])
        mask = sf.filter(queries)
        assert mask[0] is np.True_

    def test_point_outside_single_sphere(self):
        """A point well outside the sphere must be rejected."""
        centers = np.array([[0.0, 0.0, 0.0]])
        sf = SphereFilter(radius=1.0)
        sf.build(centers)

        queries = np.array([[10.0, 10.0, 10.0]])
        mask = sf.filter(queries)
        assert mask[0] is np.False_

    def test_point_on_boundary(self):
        """A point exactly at the boundary should be kept (<=)."""
        centers = np.array([[0.0, 0.0, 0.0]])
        sf = SphereFilter(radius=1.0)
        sf.build(centers)

        queries = np.array([[1.0, 0.0, 0.0]])
        mask = sf.filter(queries)
        assert mask[0] is np.True_

    def test_multiple_spheres(self):
        """Point near ANY sphere centre should be kept."""
        centers = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ])
        sf = SphereFilter(radius=1.0)
        sf.build(centers)

        queries = np.array([
            [0.5, 0.0, 0.0],    # inside sphere 0
            [10.5, 0.0, 0.0],   # inside sphere 1
            [5.0, 0.0, 0.0],    # not inside any sphere
            [20.9, 0.0, 0.0],   # inside sphere 2
        ])
        mask = sf.filter(queries)
        np.testing.assert_array_equal(mask, [True, True, False, True])

    def test_radius_scaling(self):
        """Larger radius should keep more points."""
        centers = np.array([[0.0, 0.0, 0.0]])

        queries = np.array([
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])

        sf_small = SphereFilter(radius=1.0)
        sf_small.build(centers)
        mask_small = sf_small.filter(queries)

        sf_large = SphereFilter(radius=5.0)
        sf_large.build(centers)
        mask_large = sf_large.filter(queries)

        assert mask_small.sum() < mask_large.sum()

    def test_build_validates_shape(self):
        """Build should reject non-(N,3) arrays."""
        sf = SphereFilter(radius=1.0)
        with pytest.raises(ValueError):
            sf.build(np.array([[0.0, 0.0]]))  # (1,2) instead of (1,3)

    def test_filter_before_build_raises(self):
        """Filtering without building the tree should raise."""
        sf = SphereFilter(radius=1.0)
        with pytest.raises(RuntimeError):
            sf.filter(np.array([[0.0, 0.0, 0.0]]))

    def test_empty_query(self):
        """Empty query array should return empty mask."""
        centers = np.array([[0.0, 0.0, 0.0]])
        sf = SphereFilter(radius=1.0)
        sf.build(centers)

        queries = np.empty((0, 3), dtype=np.float64)
        mask = sf.filter(queries)
        assert len(mask) == 0

    def test_many_centers_performance(self):
        """Smoke test with a realistic number of centres and queries."""
        rng = np.random.default_rng(42)
        centers = rng.standard_normal((10000, 3))
        queries = rng.standard_normal((50000, 3))

        sf = SphereFilter(radius=0.5)
        sf.build(centers)
        mask = sf.filter(queries)

        assert mask.shape == (50000,)
        # With 10K centres and radius 0.5 in unit-normal space,
        # some points should be inside and some outside
        assert 0 < mask.sum() < 50000

    def test_batch_consistency(self):
        """Different batch sizes must produce identical masks."""
        rng = np.random.default_rng(99)
        centers = rng.standard_normal((500, 3))
        queries = rng.standard_normal((10000, 3))

        sf = SphereFilter(radius=0.8)
        sf.build(centers)

        mask_small = sf.filter(queries, batch_size=1000)
        mask_large = sf.filter(queries, batch_size=100_000)

        np.testing.assert_array_equal(mask_small, mask_large)

