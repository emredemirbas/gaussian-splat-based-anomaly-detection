import numpy as np
import io_utils
import math_utils
from ellipsoid_filter import EllipsoidFilter

data = io_utils.PlyIO.read('point_cloud.ply')
xyz_colmap = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)

cam_centers = io_utils.get_colmap_camera_centers('images.bin')
mu, V, eigvals = math_utils.compute_pca_frame(cam_centers)
cam_pca = math_utils.apply_pca_transform(cam_centers, mu, V)
xyz_pca = math_utils.apply_pca_transform(xyz_colmap, mu, V)

print(f"Original Y range: min={xyz_colmap[:,1].min():.3f}, max={xyz_colmap[:,1].max():.3f}")

ellipsoid = EllipsoidFilter(min_radius=0.5, inflate_factor=1.15)
coarse_mask, info = ellipsoid.filter(xyz_pca, cam_pca)

surviving = xyz_colmap[coarse_mask]
print(f"Ellipsoid radii: {info['radii']}")
print(f"Surviving points: {len(surviving)}")
print(f"Surviving Y range: min={surviving[:,1].min():.3f}, max={surviving[:,1].max():.3f}")

