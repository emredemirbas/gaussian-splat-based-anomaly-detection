import numpy as np

def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """
    Convert COLMAP quaternion (qw, qx, qy, qz) to a 3x3 rotation matrix.
    """
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w,     2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w,     1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w,     2 * y * z + 2 * x * w,   1 - 2 * x**2 - 2 * y**2]
    ], dtype=np.float64)

def smooth_path(positions: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """
    Smooth a 3D trajectory using second-order (curvature) regularization.

    Minimizes:
        ||X - P||^2 + lam * ||D X||^2
    where D is the second-difference operator (N-2 x N).

    Solves three independent linear systems for x,y,z.

    Parameters
    ----------
    positions : (N,3)
    lam : float

    Returns
    -------
    smoothed : (N,3)
    """
    P = np.asarray(positions, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("positions must be (N,3)")

    N = P.shape[0]
    if N < 3 or lam <= 0:
        return P.copy()

    D = np.zeros((N - 2, N), dtype=np.float64)
    for i in range(N - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0

    A = np.eye(N, dtype=np.float64) + lam * (D.T @ D)

    smoothed = np.empty_like(P)
    for k in range(3):
        smoothed[:, k] = np.linalg.solve(A, P[:, k])
    return smoothed

def compute_pca_frame(points: np.ndarray, eps: float = 1e-12):
    """
    Compute PCA frame from 3D points.

    Returns
    -------
    mu : (3,) mean
    V  : (3,3) eigenvectors sorted by descending eigenvalue (columns)
         Transform: (p - mu) @ V  -> PCA coordinates
    eigvals : (3,) sorted descending
    """
    X = np.asarray(points, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("points must be (N,3)")

    mu = X.mean(axis=0)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(X.shape[0], 1)

    eigvals, eigvecs = np.linalg.eigh(C)   # ascending
    order = np.argsort(eigvals)[::-1]      # descending
    eigvals = eigvals[order]
    V = eigvecs[:, order]                 # columns

    # Optional sign stabilization for reproducibility
    for i in range(3):
        col = V[:, i]
        j = np.argmax(np.abs(col))
        if col[j] < 0:
            V[:, i] = -col

    return mu, V, eigvals

def apply_pca_transform(points: np.ndarray, mu: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Apply PCA alignment transform: (p - mu) @ V
    """
    X = np.asarray(points, dtype=np.float64)
    return (X - mu) @ V

def inverse_pca_transform(points_pca: np.ndarray, mu: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Inverse of apply_pca_transform: p_world = p_pca @ V^T + mu

    Since V is orthonormal, V^T = V^{-1}.
    """
    X = np.asarray(points_pca, dtype=np.float64)
    return X @ V.T + mu

def is_planar_from_eigvals(eigvals: np.ndarray, tau: float = 1e-2, eps: float = 1e-12):
    """
    Decide if a trajectory is planar based on eigenvalue ratio.

    Let eigvals sorted descending: lambda1 >= lambda2 >= lambda3.
    Planar if lambda3 / (lambda2 + eps) < tau.

    Returns
    -------
    planar : bool
    ratio : float
    """
    ev = np.asarray(eigvals, dtype=np.float64).reshape(-1)
    if ev.size != 3:
        raise ValueError("eigvals must have length 3")
    ratio = float(ev[2] / (ev[1] + eps))
    return ratio < tau, ratio
