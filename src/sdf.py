import numpy as np
import matplotlib.pyplot as plt
class TrajectorySDF:
    """
    RBF-kernel SDF fitted to a trajectory, in either 2D or 3D.

    Construct constraints by shifting each trajectory point slightly toward the
    trajectory centroid (inside), keeping itself (surface), and shifting away (outside).

    Solve:
        (K + reg I) w = s
    with RBF kernel K and targets s: inside=-1, surface=0, outside=+1.
    """

    def __init__(self, sigma: float = 10.0, step: float = 0.2, reg: float = 1e-6, dim: int = 2):
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        self.sigma = float(sigma)
        self.step = float(step)
        self.reg = float(reg)
        self.dim = int(dim)

        self.path = None
        self.support_points = None
        self.weights = None

    def _enhance_points(self, points: np.ndarray) -> np.ndarray:
        P = np.asarray(points, dtype=np.float64)
        if P.ndim != 2 or P.shape[1] != self.dim:
            raise ValueError(f"points must be (N,{self.dim})")

        center = np.mean(P, axis=0, keepdims=True)  # (1,dim)
        dirs = center - P
        norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        U = dirs / norms

        inside = P + self.step * U
        surface = P
        outside = P - self.step * U
        return np.vstack([inside, surface, outside])

    def _kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        Xn = np.sum(X**2, axis=1, keepdims=True)
        Yn = np.sum(Y**2, axis=1, keepdims=True).T
        sqdist = Xn + Yn - 2.0 * (X @ Y.T)
        return np.exp(-sqdist / (2.0 * self.sigma**2))

    def fit(self, path_points: np.ndarray):
        P = np.asarray(path_points, dtype=np.float64)
        if P.ndim != 2 or P.shape[1] != self.dim:
            raise ValueError(f"path_points must be (N,{self.dim})")

        self.path = P
        self.support_points = self._enhance_points(P)

        N = P.shape[0]
        s = np.zeros(3 * N, dtype=np.float64)
        s[:N] = -1.0
        s[N:2*N] = 0.0
        s[2*N:] = 1.0

        K = self._kernel_matrix(self.support_points, self.support_points)
        A = K + self.reg * np.eye(K.shape[0], dtype=np.float64)
        self.weights = np.linalg.solve(A, s)

    def predict(self, query_points: np.ndarray, batch_size: int = 20000) -> np.ndarray:
        if self.support_points is None or self.weights is None:
            raise RuntimeError("SDF not fitted. Call fit() first.")

        Q = np.asarray(query_points, dtype=np.float64)
        if Q.ndim != 2 or Q.shape[1] != self.dim:
            raise ValueError(f"query_points must be (M,{self.dim})")

        out = np.empty(Q.shape[0], dtype=np.float64)
        for i in range(0, Q.shape[0], batch_size):
            qb = Q[i:i+batch_size]
            Kq = self._kernel_matrix(qb, self.support_points)
            
            out[i:i+batch_size] = (Kq @ self.weights)
            
        return out

    def visualize_2d(self, grid_res: int = 200, margin: float = 1.0):
        if self.dim != 2:
            raise ValueError("visualize_2d only available for dim==2")
        if self.path is None:
            raise RuntimeError("Fit the SDF first.")

        P = self.path
        xmin, ymin = P.min(axis=0) - margin
        xmax, ymax = P.max(axis=0) + margin

        xs = np.linspace(xmin, xmax, grid_res)
        ys = np.linspace(ymin, ymax, grid_res)
        Xg, Yg = np.meshgrid(xs, ys)
        grid = np.column_stack([Xg.ravel(), Yg.ravel()])

        vals = self.predict(grid, batch_size=50000).reshape(grid_res, grid_res)

        plt.figure()
        plt.imshow(vals, extent=[xmin, xmax, ymin, ymax], origin="lower", aspect="equal")
        plt.colorbar(label="SDF value")
        plt.contour(Xg, Yg, vals, levels=[0.0], colors="red")
        plt.plot(P[:, 0], P[:, 1], "w-", linewidth=2, label="trajectory")
        plt.legend()
        plt.title("2D Trajectory SDF (PCA-aligned space)")
        plt.show()
