"""
Tests for anomaly_detector module.
"""

import numpy as np
import pytest
from anomaly_detector import (
    RXDetector,
    RRXDetector,
    extract_features,
    estimate_sigma,
    threshold_scores,
)


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _make_background_with_anomalies(n_bg=500, n_anom=10, d=3, seed=42):
    """
    Create a simple test dataset:
      - background: N(0, I)
      - anomalies:  shifted to mean=5 in every dimension
    Returns X, true_labels (0=bg, 1=anom)
    """
    rng = np.random.RandomState(seed)
    bg = rng.randn(n_bg, d)
    anom = rng.randn(n_anom, d) + 5.0
    X = np.vstack([bg, anom])
    labels = np.zeros(n_bg + n_anom, dtype=int)
    labels[n_bg:] = 1
    return X, labels


def _make_gsplat_structured_array(n=100, seed=0):
    """Create a fake structured array mimicking 3DGS PLY fields."""
    rng = np.random.RandomState(seed)
    dtype = np.dtype([
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
    ])
    arr = np.zeros(n, dtype=dtype)
    for name in dtype.names:
        arr[name] = rng.randn(n).astype(np.float32)
    return arr


# --------------------------------------------------------------------------- #
#  extract_features                                                           #
# --------------------------------------------------------------------------- #

class TestExtractFeatures:
    def test_position_only(self):
        data = _make_gsplat_structured_array(50)
        X, names = extract_features(data, use_position=True, use_color=False)
        assert X.shape == (50, 3)
        assert names == ["x", "y", "z"]

    def test_position_and_color(self):
        data = _make_gsplat_structured_array(50)
        X, names = extract_features(data, use_position=True, use_color=True)
        assert X.shape == (50, 6)
        assert "f_dc_0" in names

    def test_all_features(self):
        data = _make_gsplat_structured_array(50)
        X, names = extract_features(data, use_position=True, use_color=True,
                                    use_opacity=True, use_scale=True)
        assert X.shape == (50, 10)  # 3 pos + 3 color + 1 opacity + 3 scale

    def test_rgb_fallback(self):
        """When f_dc_* not present but red/green/blue are, those are used."""
        dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"),
                          ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        arr = np.zeros(10, dtype=dtype)
        arr["red"] = 128
        X, names = extract_features(arr, use_position=True, use_color=True)
        assert X.shape == (10, 6)
        assert "red" in names

    def test_no_features_raises(self):
        dtype = np.dtype([("foo", "f4")])
        arr = np.zeros(5, dtype=dtype)
        with pytest.raises(ValueError, match="No features"):
            extract_features(arr, use_position=True, use_color=False)


# --------------------------------------------------------------------------- #
#  RXDetector                                                                 #
# --------------------------------------------------------------------------- #

class TestRXDetector:
    def test_fit_and_score_shapes(self):
        X = np.random.randn(100, 4)
        det = RXDetector()
        det.fit(X)
        scores = det.score(X)
        assert scores.shape == (100,)
        assert np.all(scores >= 0)

    def test_anomalies_score_higher(self):
        X, labels = _make_background_with_anomalies()
        det = RXDetector()
        scores = det.fit_score(X)
        bg_mean = scores[labels == 0].mean()
        anom_mean = scores[labels == 1].mean()
        assert anom_mean > bg_mean * 5, "Anomalies should score much higher"

    def test_fit_score_equals_separate(self):
        X = np.random.randn(50, 3)
        det = RXDetector()
        s1 = det.fit_score(X)
        det2 = RXDetector()
        det2.fit(X)
        s2 = det2.score(X)
        np.testing.assert_allclose(s1, s2)

    def test_unfitted_raises(self):
        det = RXDetector()
        with pytest.raises(RuntimeError, match="fit"):
            det.score(np.random.randn(10, 3))

    def test_identity_covariance(self):
        """For zero-mean, identity covariance data, RX score ≈ ||x||²."""
        rng = np.random.RandomState(123)
        X = rng.randn(10000, 3)
        det = RXDetector(reg=0.0)
        scores = det.fit_score(X)
        norms_sq = np.sum((X - X.mean(axis=0)) ** 2, axis=1)
        # Should correlate very strongly
        corr = np.corrcoef(scores, norms_sq)[0, 1]
        assert corr > 0.99


# --------------------------------------------------------------------------- #
#  RRXDetector                                                                #
# --------------------------------------------------------------------------- #

class TestRRXDetector:
    def test_fit_and_score_shapes(self):
        X = np.random.randn(200, 5)
        det = RRXDetector(n_features=50, sigma=1.0)
        det.fit(X)
        scores = det.score(X)
        assert scores.shape == (200,)
        assert np.all(scores >= 0)

    def test_anomalies_score_higher(self):
        X, labels = _make_background_with_anomalies(n_bg=500, n_anom=20)
        sigma = estimate_sigma(X)
        det = RRXDetector(n_features=200, sigma=sigma)
        scores = det.fit_score(X)
        bg_mean = scores[labels == 0].mean()
        anom_mean = scores[labels == 1].mean()
        assert anom_mean > bg_mean * 2, "RRX anomalies should score higher"

    def test_reproducibility(self):
        X = np.random.randn(100, 4)
        det1 = RRXDetector(n_features=100, sigma=1.0, random_state=0)
        s1 = det1.fit_score(X)
        det2 = RRXDetector(n_features=100, sigma=1.0, random_state=0)
        s2 = det2.fit_score(X)
        np.testing.assert_allclose(s1, s2)

    def test_different_seeds_differ(self):
        X = np.random.randn(100, 4)
        det1 = RRXDetector(n_features=100, sigma=1.0, random_state=0)
        s1 = det1.fit_score(X)
        det2 = RRXDetector(n_features=100, sigma=1.0, random_state=99)
        s2 = det2.fit_score(X)
        assert not np.allclose(s1, s2)

    def test_unfitted_raises(self):
        det = RRXDetector()
        with pytest.raises(RuntimeError, match="fit"):
            det.score(np.random.randn(10, 3))

    def test_nonlinear_cluster_detection(self):
        """
        RRX should handle non-Gaussian data (ring-shaped background)
        better than linear RX.
        """
        rng = np.random.RandomState(42)
        # Background: ring of radius 5
        theta = rng.uniform(0, 2 * np.pi, 400)
        r = 5.0 + rng.randn(400) * 0.3
        bg = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        # Anomaly: at center
        anom = rng.randn(15, 2) * 0.2
        X = np.vstack([bg, anom])
        labels = np.zeros(415, dtype=int)
        labels[400:] = 1

        sigma = estimate_sigma(X)
        det = RRXDetector(n_features=200, sigma=sigma)
        scores = det.fit_score(X)

        # Anomalies (center) should score higher
        bg_median = np.median(scores[labels == 0])
        anom_median = np.median(scores[labels == 1])
        assert anom_median > bg_median


# --------------------------------------------------------------------------- #
#  estimate_sigma                                                             #
# --------------------------------------------------------------------------- #

class TestEstimateSigma:
    def test_returns_positive(self):
        X = np.random.randn(100, 3)
        s = estimate_sigma(X)
        assert s > 0

    def test_scales_with_data(self):
        X1 = np.random.randn(200, 3)
        X2 = np.random.randn(200, 3) * 10
        s1 = estimate_sigma(X1)
        s2 = estimate_sigma(X2)
        assert s2 > s1 * 3  # X2 is 10x wider, sigma should reflect that


# --------------------------------------------------------------------------- #
#  threshold_scores                                                           #
# --------------------------------------------------------------------------- #

class TestThresholdScores:
    def test_basic(self):
        scores = np.arange(100, dtype=np.float64)
        mask = threshold_scores(scores, percentile=95)
        assert mask.sum() == 5  # values 95..99 exceed the 95th percentile

    def test_percentile_99(self):
        scores = np.arange(1000, dtype=np.float64)
        mask = threshold_scores(scores, percentile=99)
        # 99th percentile of [0..999] is 990.01, so 991..999 = 9 points
        assert 5 <= mask.sum() <= 15

    def test_all_same(self):
        scores = np.ones(50)
        mask = threshold_scores(scores, percentile=99)
        assert mask.sum() == 0  # nothing exceeds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
