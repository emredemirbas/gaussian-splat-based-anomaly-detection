"""
Anomaly detection for Gaussian Splat point clouds using RX-family detectors.

Implements:
    - Linear RX  (Global, Mahalanobis distance based)
    - RRX         (Randomized RX via Random Fourier Features, approximates KRX)

Reference:
    [1] Reed & Yu, "Adaptive multiple-band CFAR detection", IEEE TASSP, 1990.
    [2] Padrón Hidalgo et al., "Efficient Nonlinear RX Anomaly Detectors",
        IEEE GRSL, 2021.
    [3] Yego & Nar, "Ship Detection using Global-Local RX Detector", SIU 2025.
    [4] Yapıcı & Nar, "Randomized Low-Rank Nonlinear RX Detector", 2022.
"""

import numpy as np
from scipy import stats as sp_stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
#  Feature extraction from structured PLY arrays                              #
# --------------------------------------------------------------------------- #

def extract_features(gsplat_data: np.ndarray,
                     use_position: bool = True,
                     use_color: bool = True,
                     use_opacity: bool = False,
                     use_scale: bool = False,
                     ) -> np.ndarray:
    """
    Build a feature matrix X (n, d) from a structured PLY array.

    Standard 3DGS PLY fields:
        position  : x, y, z
        color SH  : f_dc_0, f_dc_1, f_dc_2  (DC spherical harmonic coeffs)
        opacity   : opacity  (logit space)
        scale     : scale_0, scale_1, scale_2  (log space)

    Parameters
    ----------
    gsplat_data : structured ndarray from PlyIO.read()
    use_position : include (x, y, z)
    use_color    : include DC color coefficients (f_dc_0/1/2) or (red, green, blue)
    use_opacity  : include opacity
    use_scale    : include scale (converted from log space)

    Returns
    -------
    X : (n, d) float64 feature matrix
    feature_names : list of str, names of the d features
    """
    cols = []
    names = []
    field_names = gsplat_data.dtype.names

    if use_position:
        for c in ("x", "y", "z"):
            if c in field_names:
                cols.append(gsplat_data[c].astype(np.float64))
                names.append(c)

    if use_color:
        # 3DGS format: DC spherical harmonics
        dc_cols = ("f_dc_0", "f_dc_1", "f_dc_2")
        rgb_cols = ("red", "green", "blue")
        if all(c in field_names for c in dc_cols):
            for c in dc_cols:
                cols.append(gsplat_data[c].astype(np.float64))
                names.append(c)
        elif all(c in field_names for c in rgb_cols):
            for c in rgb_cols:
                cols.append(gsplat_data[c].astype(np.float64))
                names.append(c)

    if use_opacity and "opacity" in field_names:
        cols.append(gsplat_data["opacity"].astype(np.float64))
        names.append("opacity")

    if use_scale:
        scale_cols = ("scale_0", "scale_1", "scale_2")
        if all(c in field_names for c in scale_cols):
            for c in scale_cols:
                # Convert from log-scale to linear
                cols.append(np.exp(gsplat_data[c].astype(np.float64)))
                names.append(c)

    if len(cols) == 0:
        raise ValueError("No features could be extracted. Check PLY field names.")

    X = np.column_stack(cols)
    return X, names


# --------------------------------------------------------------------------- #
#  Linear RX detector                                                         #
# --------------------------------------------------------------------------- #

class RXDetector:
    """
    Global Linear RX anomaly detector.

    D_RX(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)

    where μ is the mean and Σ is the covariance of the background.
    Assumes the background is Gaussian distributed.
    """

    def __init__(self, reg: float = 1e-6):
        """
        Parameters
        ----------
        reg : float
            Tikhonov regularisation added to the diagonal of Σ.
        """
        self.reg = float(reg)
        self.mu = None
        self.sigma_inv = None

    def fit(self, X: np.ndarray):
        """
        Estimate background statistics from X.

        Parameters
        ----------
        X : (n, d) data matrix
        """
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        self.mu = X.mean(axis=0)
        X_c = X - self.mu
        sigma = (X_c.T @ X_c) / n
        sigma += self.reg * np.eye(d, dtype=np.float64)
        self.sigma_inv = np.linalg.inv(sigma)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute RX anomaly scores (Mahalanobis distance squared).

        Parameters
        ----------
        X : (m, d) data matrix (can be the same as fit data)

        Returns
        -------
        scores : (m,) float64, higher = more anomalous
        """
        if self.sigma_inv is None:
            raise RuntimeError("Call fit() first.")
        X_c = np.asarray(X, dtype=np.float64) - self.mu
        # Vectorised Mahalanobis: score_i = x_c_i @ Σ⁻¹ @ x_c_i
        tmp = X_c @ self.sigma_inv          # (m, d)
        return np.einsum("ij,ij->i", tmp, X_c)

    def fit_score(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and return scores for the same X."""
        self.fit(X)
        return self.score(X)


# --------------------------------------------------------------------------- #
#  Randomized RX (RRX) detector via Random Fourier Features                   #
# --------------------------------------------------------------------------- #

class RRXDetector:
    """
    Randomized RX anomaly detector using Random Fourier Features (RFF).

    Approximates the Kernel RX (KRX) detector efficiently.

    Algorithm:
        1. Sample D random frequency vectors ω_i ~ N(0, γ²I)
           where γ = 1/σ for RBF kernel K(x,y) = exp(-||x-y||² / 2σ²)
        2. Map each point x ∈ ℝ^d → z(x) ∈ ℝ^{2D}:
             z(x) = (1/√D) [cos(ω₁ᵀx), sin(ω₁ᵀx), ..., cos(ω_Dᵀx), sin(ω_Dᵀx)]
        3. Center Z and compute covariance G = (1/n) ZᵀZ
        4. Score: D_RRX(x*) = z*ᵀ G⁻¹ z*

    Reference:
        Padrón Hidalgo et al., IEEE GRSL, 2021, Eq. (5).
        Yapıcı & Nar, MSc Thesis, AYBU, 2022, Eq. (2.12).
    """

    def __init__(self, n_features: int = 300, sigma: float = 1.0,
                 reg: float = 1e-6, random_state: int = 42):
        """
        Parameters
        ----------
        n_features : int
            Number of random Fourier basis functions D.
            The mapped space has dimension 2D.
        sigma : float
            RBF kernel bandwidth. Larger σ = wider kernel = smoother model.
            Rule of thumb: set σ ≈ median pairwise distance of the data.
        reg : float
            Regularisation constant for the covariance inversion.
        random_state : int
            Seed for reproducible random frequency sampling.
        """
        self.D = int(n_features)
        self.sigma = float(sigma)
        self.reg = float(reg)
        self.random_state = int(random_state)

        # Fitted state
        self.omega = None    # (D, d)  random frequencies
        self.z_mean = None   # (2D,)   mean of mapped features
        self.G_inv = None    # (2D, 2D) inverse covariance in RFF space

    def _build_rff(self, d: int):
        """Sample D random frequency vectors for RBF kernel approximation."""
        rng = np.random.RandomState(self.random_state)
        # For RBF kernel K(x,y) = exp(-||x-y||²/(2σ²)),
        # the spectral density is N(0, (1/σ²)I_d).
        gamma = 1.0 / self.sigma
        self.omega = rng.normal(0, gamma, size=(self.D, d))

    def _map(self, X: np.ndarray) -> np.ndarray:
        """
        Apply RFF mapping: x → z(x) ∈ ℝ^{2D}.

        z(x) = (1/√D) [cos(Ωx), sin(Ωx)]

        Parameters
        ----------
        X : (m, d) data matrix

        Returns
        -------
        Z : (m, 2D) mapped feature matrix
        """
        proj = X @ self.omega.T   # (m, D)
        scale = 1.0 / np.sqrt(self.D)
        return scale * np.hstack([np.cos(proj), np.sin(proj)])

    def fit(self, X: np.ndarray, batch_size: int = 100_000):
        """
        Fit the RRX detector on background data X.

        Parameters
        ----------
        X : (n, d) data matrix
        batch_size: int
            Batch size for computing features to avoid OOM
        """
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape

        # Step 1: Build random frequencies
        self._build_rff(d)

        # Step 2: Compute mean in batches
        dim_2D = 2 * self.D
        self.z_mean = np.zeros(dim_2D, dtype=np.float64)
        
        for i in range(0, n, batch_size):
            X_batch = X[i:i + batch_size]
            Z_batch = self._map(X_batch)
            self.z_mean += Z_batch.sum(axis=0)
        self.z_mean /= n

        # Step 3: Compute covariance and its inverse in batches
        G = np.zeros((dim_2D, dim_2D), dtype=np.float64)
        for i in range(0, n, batch_size):
            X_batch = X[i:i + batch_size]
            Z_batch = self._map(X_batch)
            Z_batch -= self.z_mean
            G += (Z_batch.T @ Z_batch)
        
        G /= n
        G += self.reg * np.eye(dim_2D, dtype=np.float64)
        self.G_inv = np.linalg.inv(G)

    def score(self, X: np.ndarray, batch_size: int = 100_000) -> np.ndarray:
        """
        Compute RRX anomaly scores for query points.

        Parameters
        ----------
        X : (m, d) data matrix
        batch_size: int
            Batch size for computing scores to avoid OOM

        Returns
        -------
        scores : (m,) float64, higher = more anomalous
        """
        if self.G_inv is None:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        m = X.shape[0]
        scores = np.zeros(m, dtype=np.float64)
        
        for i in range(0, m, batch_size):
            X_batch = X[i:i + batch_size]
            Z_batch = self._map(X_batch)
            Z_batch -= self.z_mean
            tmp = Z_batch @ self.G_inv
            scores[i:i + batch_size] = np.einsum("ij,ij->i", tmp, Z_batch)
            
        return scores

    def fit_score(self, X: np.ndarray, batch_size: int = 100_000) -> np.ndarray:
        """Fit on X and return scores for the same X."""
        self.fit(X, batch_size=batch_size)
        return self.score(X, batch_size=batch_size)


# --------------------------------------------------------------------------- #
#  Convenience: estimate RBF sigma from data                                  #
# --------------------------------------------------------------------------- #

def estimate_sigma(X: np.ndarray, n_samples: int = 2000) -> float:
    """
    Estimate a good σ for the RBF kernel using median heuristic.

    Samples a subset of pairwise distances and returns the median.
    This is a well-known heuristic in the kernel methods literature.

    Parameters
    ----------
    X : (n, d) data matrix
    n_samples : number of random pairs to sample

    Returns
    -------
    sigma : float
    """
    rng = np.random.RandomState(0)
    n = X.shape[0]
    idx_a = rng.randint(0, n, size=n_samples)
    idx_b = rng.randint(0, n, size=n_samples)
    dists = np.linalg.norm(X[idx_a] - X[idx_b], axis=1)
    median_dist = np.median(dists[dists > 0])
    return float(median_dist) if median_dist > 0 else 1.0


# --------------------------------------------------------------------------- #
#  Thresholding utilities                                                     #
# --------------------------------------------------------------------------- #

def threshold_scores(scores: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """
    Mark points above the given percentile as anomalies.

    Parameters
    ----------
    scores : (n,) anomaly scores
    percentile : percentile threshold (e.g. 99 means top 1% are anomalies)

    Returns
    -------
    is_anomaly : (n,) bool, True = anomaly
    """
    tau = np.percentile(scores, percentile)
    return scores > tau


# --------------------------------------------------------------------------- #
#  Candidate distributions for CFAR fitting (all defined on [0, ∞))           #
# --------------------------------------------------------------------------- #

_CFAR_CANDIDATES = {
    "gamma":       sp_stats.gamma,
    "lognorm":     sp_stats.lognorm,
    "chi2":        sp_stats.chi2,
    "weibull_min": sp_stats.weibull_min,
    "rayleigh":    sp_stats.rayleigh,
    "expon":       sp_stats.expon,
    "nakagami":    sp_stats.nakagami,
}


def cfar_threshold_scores(
    scores: np.ndarray,
    pfa: float = 0.01,
    n_bins: int = 100,
    candidates: list | None = None,
    plot: bool = True,
    save_path: str | None = None,
) -> tuple:
    """
    Data-driven CFAR thresholding for anomaly scores.

    Fits a family of candidate probability distributions to the score
    histogram via Maximum Likelihood Estimation (MLE), selects the
    best-fitting one using the Kolmogorov–Smirnov statistic, and
    computes the CFAR threshold as the inverse CDF at (1 − Pfa).

    Algorithm
    ---------
    1. Build a histogram of the anomaly scores with *n_bins* bins.
    2. For each candidate distribution, estimate its parameters from
       the scores via MLE (``scipy.stats.<dist>.fit()``).
    3. Evaluate goodness-of-fit with the KS test; select the
       distribution with the smallest KS statistic.
    4. Compute the threshold:  τ = F⁻¹(1 − Pfa)  where F is the CDF
       of the best-fitting distribution.
    5. Optionally plot the histogram, fitted PDF, and threshold.

    Parameters
    ----------
    scores : (n,) float array
        Anomaly scores (non-negative, e.g. from RRXDetector).
    pfa : float
        Desired probability of false alarm.  Default 0.01 (1 %).
    n_bins : int
        Number of histogram bins for the plot and analysis.
        Default 100.
    candidates : list of str or None
        Names of candidate distributions to try.  If None, all
        built-in candidates are used: gamma, lognorm, chi2,
        weibull_min, rayleigh, expon, nakagami.
    plot : bool
        If True, display the histogram with the fitted PDF and
        threshold line.  Default True.
    save_path : str or None
        If provided, save the plot to this file path (e.g.
        ``"output/cfar_plot.png"``).  The format is inferred from
        the extension.  Default None (no file saved).

    Returns
    -------
    is_anomaly : (n,) bool array
        True for every point whose score exceeds the CFAR threshold.
    threshold : float
        The computed CFAR threshold τ.
    fit_info : dict
        Diagnostic information::

            best_dist   – str, name of the selected distribution
            params      – tuple, MLE parameters of the best distribution
            ks_stat     – float, KS statistic of the best fit
            ks_pvalue   – float, KS test p-value of the best fit
            pfa         – float, the requested false alarm rate
            threshold   – float, same as the returned threshold
            all_fits    – dict mapping dist name →
                          (ks_stat, ks_pvalue, params)

    References
    ----------
    [1] Reed & Yu, "Adaptive multiple-band CFAR detection",
        IEEE TASSP, 1990.
    """
    scores = np.asarray(scores, dtype=np.float64)

    # Use only positive scores for fitting (avoids log(0) in lognorm etc.)
    positive_scores = scores[scores > 0]
    if len(positive_scores) == 0:
        raise ValueError(
            "All scores are zero or negative; cannot fit a distribution."
        )

    # ---- Determine candidate pool ---------------------------------------- #
    if candidates is None:
        dist_pool = _CFAR_CANDIDATES
    else:
        dist_pool = {}
        for name in candidates:
            if name not in _CFAR_CANDIDATES:
                raise ValueError(
                    f"Unknown distribution '{name}'. "
                    f"Available: {list(_CFAR_CANDIDATES.keys())}"
                )
            dist_pool[name] = _CFAR_CANDIDATES[name]

    # ---- Fit each candidate via MLE & evaluate with KS test -------------- #
    all_fits: dict = {}
    best_name: str | None = None
    best_ks: float = np.inf
    best_params: tuple | None = None

    for name, dist in dist_pool.items():
        try:
            params = dist.fit(positive_scores)
            ks_stat, ks_pval = sp_stats.kstest(
                positive_scores, dist.cdf, args=params,
            )
            all_fits[name] = (ks_stat, ks_pval, params)

            if ks_stat < best_ks:
                best_ks = ks_stat
                best_name = name
                best_params = params
        except Exception:
            # Some distributions may fail to converge; skip silently.
            continue

    if best_name is None:
        raise RuntimeError(
            "No candidate distribution could be fitted to the scores."
        )

    best_dist = _CFAR_CANDIDATES[best_name]
    ks_pval = all_fits[best_name][1]

    # ---- CFAR threshold: τ = F⁻¹(1 − Pfa) ------------------------------- #
    threshold = float(best_dist.ppf(1.0 - pfa, *best_params))

    # ---- Classify -------------------------------------------------------- #
    is_anomaly = scores > threshold

    # ---- Build diagnostic dict ------------------------------------------- #
    fit_info = {
        "best_dist":  best_name,
        "params":     best_params,
        "ks_stat":    best_ks,
        "ks_pvalue":  ks_pval,
        "pfa":        pfa,
        "threshold":  threshold,
        "all_fits":   all_fits,
    }

    # ---- Console summary ------------------------------------------------- #
    print(f"[CFAR] Best-fitting distribution: {best_name}")
    print(f"[CFAR] KS statistic: {best_ks:.6f}  (p-value: {ks_pval:.4e})")
    print(f"[CFAR] Parameters: {best_params}")
    print(f"[CFAR] Pfa = {pfa:.4f}  →  threshold τ = {threshold:.6f}")
    print(
        f"[CFAR] Anomalies: {int(is_anomaly.sum())} / {len(scores)} "
        f"({100.0 * is_anomaly.mean():.2f}%)"
    )

    if plot or save_path:
        _plot_cfar(
            scores, n_bins, best_name, best_dist, best_params,
            threshold, pfa, all_fits, save_path=save_path,
        )

    return is_anomaly, threshold, fit_info


def _plot_cfar(
    scores: np.ndarray,
    n_bins: int,
    best_name: str,
    best_dist,
    best_params: tuple,
    threshold: float,
    pfa: float,
    all_fits: dict,
    save_path: str | None = None,
) -> None:
    """
    Plot the score histogram, best-fitting PDF, and CFAR threshold.

    Uses a **log-scaled x-axis** with log-spaced histogram bins so that
    heavy-tailed score distributions (common for RX-family detectors)
    are clearly readable.
    """
    # ---- Prepare log-spaced bins ----------------------------------------- #
    positive = scores[scores > 0]
    if len(positive) == 0:
        return

    lo = positive.min()
    hi = scores.max()
    log_bins = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    # ---- Histogram (log-spaced bins) ------------------------------------- #
    ax.hist(
        positive, bins=log_bins, density=True, alpha=0.55,
        color="steelblue", edgecolor="white", linewidth=0.5,
        label="Score histogram",
    )

    # ---- Best-fitting PDF ------------------------------------------------ #
    x = np.logspace(np.log10(lo), np.log10(hi), 800)
    pdf = best_dist.pdf(x, *best_params)
    ks_stat = all_fits[best_name][0]
    ax.plot(
        x, pdf, "r-", linewidth=2.5,
        label=f"Best fit: {best_name}  (KS = {ks_stat:.4f})",
    )

    # ---- Runner-up PDFs (only reasonable fits) ----------------------------- #
    for name, (ks, _, params) in all_fits.items():
        if name == best_name:
            continue
        if ks > 0.5:
            continue  # skip terrible fits that distort the plot
        dist = _CFAR_CANDIDATES[name]
        try:
            ax.plot(
                x, dist.pdf(x, *params), "-",
                color="grey", alpha=0.35, linewidth=1.0,
                label=f"{name}  (KS = {ks:.4f})",
            )
        except Exception:
            pass

    # ---- Threshold line -------------------------------------------------- #
    ax.axvline(
        threshold, color="darkorange", linewidth=2.0, linestyle="--",
        label=f"CFAR threshold (Pfa={pfa}):  τ = {threshold:.2f}",
    )

    # ---- Shaded rejection region ----------------------------------------- #
    x_reject = x[x >= threshold]
    if len(x_reject) > 0:
        pdf_reject = best_dist.pdf(x_reject, *best_params)
        ax.fill_between(
            x_reject, pdf_reject, alpha=0.25, color="darkorange",
            label="Rejection region",
        )

    # ---- Log scale ------------------------------------------------------- #
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Clamp y-axis to the range of actual histogram bar heights
    # so that poorly-fitting PDFs don't stretch the plot
    hist_counts, _ = np.histogram(positive, bins=log_bins, density=True)
    nonzero = hist_counts[hist_counts > 0]
    if len(nonzero) > 0:
        y_lo = nonzero.min() * 0.3
        y_hi = nonzero.max() * 5.0
        ax.set_ylim(y_lo, y_hi)

    # ---- Summary text box ------------------------------------------------ #
    n_total = len(scores)
    n_anom = int((scores > threshold).sum())
    summary = (
        f"Distribution: {best_name}\n"
        f"KS statistic: {ks_stat:.4f}\n"
        f"Pfa: {pfa}\n"
        f"Threshold τ: {threshold:.2f}\n"
        f"Anomalies: {n_anom} / {n_total} ({100.0 * n_anom / n_total:.2f}%)"
    )
    ax.text(
        0.02, 0.97, summary, transform=ax.transAxes,
        fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
    )

    # ---- Cosmetics ------------------------------------------------------- #
    ax.set_xlabel("Anomaly Score (log scale)", fontsize=12)
    ax.set_ylabel("Density (log scale)", fontsize=12)
    ax.set_title("CFAR Anomaly Threshold Selection", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[CFAR] Plot saved to: {save_path}")

    plt.show()


# --------------------------------------------------------------------------- #
#  Clustering and Coloring                                                    #
# --------------------------------------------------------------------------- #

def cluster_and_color_anomalies(
    anomalies_data: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster anomalies based on position and color, and paint them.

    Parameters
    ----------
    anomalies_data : structured ndarray from PlyIO
    eps : float
        DBSCAN eps parameter.
    min_samples : int
        DBSCAN min_samples parameter.

    Returns
    -------
    colored_data : structured ndarray
        The same array but with colors updated based on cluster IDs.
    labels : (N,) int ndarray
        DBSCAN cluster labels for each point.  -1 = noise, >=0 = cluster.
    """
    if len(anomalies_data) == 0:
        return anomalies_data, np.array([], dtype=np.int64)

    # 1. Extract position and color features
    X_features, _ = extract_features(
        anomalies_data,
        use_position=True,
        use_color=True,
        use_opacity=False,
        use_scale=False
    )

    # 2. Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    # 3. Cluster using DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[Clustering] Found {n_clusters} clusters (eps={eps}, min_samples={min_samples})")

    # 4. Color assignment
    # Get a colormap (tab20 can handle up to 20 distinct colors, will cycle if more)
    cmap = cm.get_cmap("tab20", n_clusters if n_clusters > 0 else 1)
    
    # We will modify the data in place or return a copy (numpy structured arrays are mutable)
    colored_data = anomalies_data.copy()
    field_names = colored_data.dtype.names
    
    is_dc = all(c in field_names for c in ("f_dc_0", "f_dc_1", "f_dc_2"))
    is_rgb = all(c in field_names for c in ("red", "green", "blue"))
    
    C0 = 0.28209479177387814  # DC Spherical Harmonic constant

    for i in range(len(colored_data)):
        label = labels[i]
        if label == -1:
            # Noise points -> Dark Grey
            color_rgb = np.array([0.2, 0.2, 0.2])
        else:
            # Cluster points -> distinct color from colormap
            # label % cmap.N allows cycling if there are many clusters
            color_rgba = cmap(label % cmap.N)
            color_rgb = np.array(color_rgba[:3])
            
        if is_dc:
            # Convert [0, 1] RGB to SH DC
            f_dc = (color_rgb - 0.5) / C0
            colored_data["f_dc_0"][i] = f_dc[0]
            colored_data["f_dc_1"][i] = f_dc[1]
            colored_data["f_dc_2"][i] = f_dc[2]
        elif is_rgb:
            # Depending on format, RGB might be 0-255 or 0-1
            # Assuming 0-255 uint8 for standard PLY
            if colored_data.dtype["red"].kind in ('i', 'u'):
                color_val = (color_rgb * 255).astype(np.uint8)
            else:
                color_val = color_rgb
            colored_data["red"][i] = color_val[0]
            colored_data["green"][i] = color_val[1]
            colored_data["blue"][i] = color_val[2]

    return colored_data, labels

