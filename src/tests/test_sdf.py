"""
Unit tests for sdf module.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from sdf import TrajectorySDF


class TestTrajectorySDF2D:
    """Tests for 2D SDF fitting and prediction."""

    @pytest.fixture
    def circle_sdf(self):
        """SDF fitted to a circular trajectory in 2D."""
        t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        path = np.column_stack([np.cos(t), np.sin(t)]) * 5.0
        sdf = TrajectorySDF(sigma=5.0, step=0.3, reg=1e-5, dim=2)
        sdf.fit(path)
        return sdf

    def test_origin_is_inside(self, circle_sdf):
        """Center of a circular trajectory should be inside (sdf < 0)."""
        val = circle_sdf.predict(np.array([[0.0, 0.0]]))
        assert val[0] < 0, f"Expected negative SDF at origin, got {val[0]}"

    def test_far_point_is_outside(self, circle_sdf):
        """Point far from trajectory should be outside (sdf > 0)."""
        val = circle_sdf.predict(np.array([[100.0, 100.0]]))
        assert val[0] > 0, f"Expected positive SDF far away, got {val[0]}"

    def test_surface_is_near_zero(self, circle_sdf):
        """Points on the trajectory should have sdf ≈ 0."""
        on_path = np.array([[5.0, 0.0], [0.0, 5.0]])
        vals = circle_sdf.predict(on_path)
        np.testing.assert_allclose(vals, 0.0, atol=0.15)

    def test_batch_consistency(self, circle_sdf):
        """Different batch sizes should produce identical results."""
        pts = np.random.default_rng(0).standard_normal((200, 2))
        v1 = circle_sdf.predict(pts, batch_size=50)
        v2 = circle_sdf.predict(pts, batch_size=10000)
        np.testing.assert_allclose(v1, v2, atol=1e-10)


class TestTrajectorySDF3D:
    """Tests for 3D SDF fitting."""

    def test_3d_fit_predict(self):
        """Smoke test: 3D SDF should fit and predict without errors."""
        t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        # Use a helix so points span all 3 dimensions (prevents singular block matrix)
        path = np.column_stack([np.cos(t), np.sin(t), np.linspace(-1, 1, len(t))]) * 3.0
        sdf = TrajectorySDF(sigma=3.0, step=0.2, reg=1e-5, dim=3)
        sdf.fit(path)

        pts = np.random.default_rng(0).standard_normal((50, 3))
        vals = sdf.predict(pts)
        assert vals.shape == (50,)

    def test_predict_before_fit_raises(self):
        sdf = TrajectorySDF(dim=2)
        with pytest.raises(RuntimeError):
            sdf.predict(np.array([[0.0, 0.0]]))

    def test_invalid_dim_raises(self):
        with pytest.raises(ValueError):
            TrajectorySDF(dim=4)
