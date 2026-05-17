"""
Unit tests for math_utils module.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import math_utils


# ------------------------------------------------------------------ #
#  qvec2rotmat                                                        #
# ------------------------------------------------------------------ #

class TestQvec2Rotmat:
    """Tests for quaternion-to-rotation-matrix conversion."""

    def test_identity_quaternion(self):
        """Identity quaternion (1, 0, 0, 0) should produce the 3x3 identity."""
        R = math_utils.qvec2rotmat(np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_output_is_orthogonal(self):
        """R^T R should equal I for any unit quaternion."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        R = math_utils.qvec2rotmat(q)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)

    def test_determinant_is_one(self):
        """Rotation matrix determinant must be +1."""
        q = np.array([0.7071067811865476, 0.7071067811865476, 0.0, 0.0])
        R = math_utils.qvec2rotmat(q)
        assert abs(np.linalg.det(R) - 1.0) < 1e-6

    def test_180_degree_rotation_around_z(self):
        """qvec = (0, 0, 0, 1) → 180° about Z."""
        R = math_utils.qvec2rotmat(np.array([0.0, 0.0, 0.0, 1.0]))
        expected = np.diag([-1.0, -1.0, 1.0])
        np.testing.assert_allclose(R, expected, atol=1e-12)


# ------------------------------------------------------------------ #
#  smooth_path                                                        #
# ------------------------------------------------------------------ #

class TestSmoothPath:
    """Tests for trajectory smoothing."""

    def test_already_smooth(self):
        """A perfectly linear path should remain (almost) unchanged."""
        t = np.linspace(0, 10, 50)
        P = np.column_stack([t, 2 * t, -t])
        S = math_utils.smooth_path(P, lam=1.0)
        np.testing.assert_allclose(S, P, atol=1e-6)

    def test_output_shape_matches_input(self):
        P = np.random.default_rng(0).standard_normal((30, 3))
        S = math_utils.smooth_path(P, lam=10.0)
        assert S.shape == P.shape

    def test_lambda_zero_returns_copy(self):
        """lam <= 0 should return an unmodified copy."""
        P = np.random.default_rng(1).standard_normal((20, 3))
        S = math_utils.smooth_path(P, lam=0.0)
        np.testing.assert_array_equal(S, P)

    def test_too_few_points_returns_copy(self):
        """Fewer than 3 points → no smoothing possible."""
        P = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        S = math_utils.smooth_path(P, lam=100.0)
        np.testing.assert_array_equal(S, P)


# ------------------------------------------------------------------ #
#  PCA utilities                                                      #
# ------------------------------------------------------------------ #

class TestPCA:
    """Tests for compute_pca_frame, apply_pca_transform, and inverse."""

    @pytest.fixture
    def sample_points(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal((100, 3)) @ np.diag([10, 3, 0.1]) + [5, -2, 8]

    def test_eigenvalues_descending(self, sample_points):
        _, _, ev = math_utils.compute_pca_frame(sample_points)
        assert ev[0] >= ev[1] >= ev[2]

    def test_eigenvectors_orthonormal(self, sample_points):
        _, V, _ = math_utils.compute_pca_frame(sample_points)
        np.testing.assert_allclose(V.T @ V, np.eye(3), atol=1e-10)

    def test_round_trip(self, sample_points):
        """apply_pca → inverse_pca should recover original points."""
        mu, V, _ = math_utils.compute_pca_frame(sample_points)
        pca = math_utils.apply_pca_transform(sample_points, mu, V)
        recovered = math_utils.inverse_pca_transform(pca, mu, V)
        np.testing.assert_allclose(recovered, sample_points, atol=1e-10)

    def test_pca_centered(self, sample_points):
        """PCA-transformed points should have zero mean."""
        mu, V, _ = math_utils.compute_pca_frame(sample_points)
        pca = math_utils.apply_pca_transform(sample_points, mu, V)
        np.testing.assert_allclose(pca.mean(axis=0), 0, atol=1e-10)


# ------------------------------------------------------------------ #
#  is_planar_from_eigvals                                             #
# ------------------------------------------------------------------ #

class TestPlanarity:

    def test_clearly_planar(self):
        """λ3 ≪ λ2 → planar."""
        planar, ratio = math_utils.is_planar_from_eigvals(np.array([100, 50, 0.001]))
        assert planar is True
        assert ratio < 1e-2

    def test_clearly_not_planar(self):
        """All eigenvalues comparable → not planar."""
        planar, _ = math_utils.is_planar_from_eigvals(np.array([10, 8, 7]))
        assert planar is False

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            math_utils.is_planar_from_eigvals(np.array([1.0, 2.0]))
