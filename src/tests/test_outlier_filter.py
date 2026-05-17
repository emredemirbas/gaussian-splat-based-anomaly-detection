"""
Unit tests for the StatisticalOutlierFilter module (locally-adaptive / LOF-like).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from outlier_filter import StatisticalOutlierFilter


class TestStatisticalOutlierFilter:
    """Tests for locally-adaptive Statistical Outlier Removal."""

    def test_obvious_outliers_removed(self):
        """Planted far-away outliers must be detected."""
        rng = np.random.default_rng(42)
        inliers = rng.normal(loc=0.0, scale=0.1, size=(200, 3))
        outliers = np.array([[50.0, 50.0, 50.0],
                             [-50.0, -50.0, -50.0]])
        cloud = np.vstack([inliers, outliers])

        sof = StatisticalOutlierFilter(k=20, threshold=2.0)
        mask = sof.filter(cloud)

        # Outliers are the last two rows
        assert mask[-1] == False
        assert mask[-2] == False
        # Most inliers should survive
        assert mask[:200].sum() > 180

    def test_multi_density_preserved(self):
        """Two clusters of different densities should both survive."""
        rng = np.random.default_rng(42)
        # Dense cluster
        dense = rng.normal(loc=0.0, scale=0.05, size=(300, 3))
        # Sparse cluster far away
        sparse = rng.normal(loc=[10.0, 10.0, 10.0], scale=0.5, size=(100, 3))
        cloud = np.vstack([dense, sparse])

        sof = StatisticalOutlierFilter(k=15, threshold=2.0)
        mask = sof.filter(cloud)

        # Both clusters should be mostly preserved
        dense_kept = mask[:300].sum()
        sparse_kept = mask[300:].sum()
        assert dense_kept > 270, f"Dense cluster lost too many: kept {dense_kept}/300"
        assert sparse_kept > 80, f"Sparse cluster lost too many: kept {sparse_kept}/100"

    def test_uniform_cluster_no_removal(self):
        """A tight uniform cluster should have nothing removed."""
        rng = np.random.default_rng(7)
        cloud = rng.normal(loc=0.0, scale=0.01, size=(500, 3))

        sof = StatisticalOutlierFilter(k=20, threshold=2.0)
        mask = sof.filter(cloud)

        assert mask.sum() >= 490

    def test_stricter_threshold_removes_more(self):
        """Lower threshold should remove more points."""
        rng = np.random.default_rng(99)
        # Add some moderate outliers
        cloud = rng.normal(loc=0.0, scale=1.0, size=(500, 3))
        outliers = rng.normal(loc=0.0, scale=5.0, size=(50, 3))
        cloud = np.vstack([cloud, outliers])

        sof_lenient = StatisticalOutlierFilter(k=20, threshold=2.0)
        sof_strict = StatisticalOutlierFilter(k=20, threshold=1.2)

        kept_lenient = sof_lenient.filter(cloud).sum()
        kept_strict = sof_strict.filter(cloud).sum()

        assert kept_strict < kept_lenient

    def test_invalid_shape_raises(self):
        """Non-(N,3) input should raise ValueError."""
        sof = StatisticalOutlierFilter(k=5)
        with pytest.raises(ValueError):
            sof.filter(np.array([[1.0, 2.0]]))

    def test_too_few_points_keeps_all(self):
        """When N <= k, all points should be kept (no crash)."""
        cloud = np.array([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0]])
        sof = StatisticalOutlierFilter(k=10, threshold=1.5)
        mask = sof.filter(cloud)
        assert mask.all()
        assert len(mask) == 2

    def test_invalid_k_raises(self):
        """k < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            StatisticalOutlierFilter(k=0)

    def test_batch_consistency(self):
        """Different batch sizes must produce identical masks."""
        rng = np.random.default_rng(123)
        cloud = rng.normal(size=(2000, 3))

        sof = StatisticalOutlierFilter(k=20, threshold=1.5)
        mask_small = sof.filter(cloud, batch_size=200)
        mask_large = sof.filter(cloud, batch_size=100_000)

        np.testing.assert_array_equal(mask_small, mask_large)
