import numpy as np
import io_utils
import math_utils
from ellipsoid_filter import EllipsoidFilter
from ground_filter import GroundFilter

data = io_utils.PlyIO.read('point_cloud.ply')
xyz_colmap = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)

cam_centers = io_utils.get_colmap_camera_centers('images.bin')
mu, V, eigvals = math_utils.compute_pca_frame(cam_centers)
planar, ratio = math_utils.is_planar_from_eigvals(eigvals, tau=0.01)
dim = 2 if planar else 3

cam_pca = math_utils.apply_pca_transform(cam_centers, mu, V)
xyz_pca = math_utils.apply_pca_transform(xyz_colmap, mu, V)

ellipsoid = EllipsoidFilter(min_radius=0.5, inflate_factor=1.15)
coarse_mask, info = ellipsoid.filter(xyz_pca, cam_pca, dim=dim)
xyz_colmap = xyz_colmap[coarse_mask]

gf = GroundFilter(
    ground_margin=0.25,
    vertical_axis=1,
    axis_points_up=False,
)
_, _, plane, _ = gf.filter_ground(xyz_colmap)

A, B, C, D = plane
dists = A * xyz_colmap[:,0] + B * xyz_colmap[:,1] + C * xyz_colmap[:,2] + D
print("\n--- DISTANCE DISTRIBUTION ---")
for margin in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
    ground = np.sum(dists <= margin)
    print(f"Margin {margin:.2f}m removes {ground} points")

