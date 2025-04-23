import numpy as np
from planning.misc.uncertainty_mapping import VoxelMap
import time
import open3d as o3d

points = np.load("./dual_grasp_pcd_mustard.npy")

voxel_size = 0.005

voxel_map = VoxelMap(points, voxel_size, visualize=True)

intrinsic_matrix = np.array([
    [525.0,   0.0, 319.5],
    [  0.0, 525.0, 239.5],
    [  0.0,   0.0,   1.0]
]) # Default Camera Intrinsics (640x480 resolution)

width_px = 640
height_px = 480

# Camera position (eye)
C = np.array([0.8, -0.00450857, 0.1432208])

# Target point the camera is looking at
target = np.array([0.43014115, -0.00450857, 0.1432208])

# Up vector (arbitrary, usually [0, 1, 0])
up = np.array([0.0, 0.0, 1.0])

# Forward vector (camera z-axis, points from camera to target)
z_axis = (target - C)
z_axis = z_axis / np.linalg.norm(z_axis)

# Right vector (camera x-axis)
x_axis = np.cross(z_axis, up)
x_axis = x_axis / np.linalg.norm(x_axis)

# Recomputed up vector (camera y-axis)
y_axis = np.cross(z_axis, x_axis)

# Rotation matrix
R = np.stack([x_axis, y_axis, z_axis], axis=0)
print(f"R: {R}")

# Extrinsic matrix [R | t]
extrinsic_matrix = np.vstack([np.hstack([R.T, C.reshape(3, 1)]), np.array([0, 0, 0, 1])])
print(extrinsic_matrix)
camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
camera_frame.transform(extrinsic_matrix)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points.T)
o3d.visualization.draw_geometries([point_cloud, camera_frame])

start = time.time()
voxel_map.update_with_observation(
    intrinsic_matrix, 
    extrinsic_matrix, 
    width_px, height_px, 
    occlusion_mask=np.array([[width_px//2 - 25 + j, i] for j in range(50) for i in range(height_px)]).T, 
    occlusion_indices=None,
    visualize=True,
    visualize_all=False
    )
print(f"update took {time.time() - start} seconds")

voxel_map.visualize()