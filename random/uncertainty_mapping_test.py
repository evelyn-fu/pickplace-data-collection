import numpy as np
from planning.misc.uncertainty_mapping import VoxelMap
import time
import open3d as o3d

from planning.grasp import GraspListener, GraspType
from pydrake.all import PointCloud, RigidTransform, RotationMatrix, RollPitchYaw

gripper_model_path = "file://./home/evelyn/src/pickplace_data_collection/models/schunk_wsg_50_welded_fingers_w_buffer.sdf"
grasp_node = GraspListener(gripper_length=0.16, gripper_model_path=gripper_model_path)

points = np.load("./dual_grasp_pcd_mustard.npy")

# TODO: Fill in points on bottom of the object

voxel_size = 0.005

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
# o3d.visualization.draw_geometries([point_cloud, camera_frame])

current_manipuland_pcd = PointCloud(points.shape[1])
current_manipuland_pcd.mutable_xyzs()[:] = points
current_manipuland_pcd.EstimateNormals(radius=0.1, num_closest=30)
current_manipuland_pcd.FlipNormalsTowardPoint(np.mean(points, axis=1))
current_manipuland_pcd.mutable_normals()[:] = -current_manipuland_pcd.mutable_normals()[:]

background_pts = np.load("./dual_grasp_background_pcd_mustard.npy")

pcd_with_background = PointCloud(background_pts.shape[1])
pcd_with_background.mutable_xyzs()[:] = background_pts
pcd_with_background.EstimateNormals(radius=0.1, num_closest=30)
pcd_with_background.FlipNormalsTowardPoint(np.mean(points, axis=1))
pcd_with_background.mutable_normals()[:] = -pcd_with_background.mutable_normals()[:]

platform_height = 0.065

grasp_node.compute_candidate_grasps(
    current_manipuland_pcd,
    pcd_with_background,
    candidate_num=1,
    num_samples=30,
    random_seed=np.random.randint(1000),
    grasp_type=GraspType.PAIR,
    split_ratio_threshold=0.15,
    ground_z=platform_height,
    is_manual=False,
    use_extra_buffer=False
)

grasp_pairs, _ = grasp_node.get_best_grasps()
X_WG1 = RigidTransform(grasp_pairs[0][0])
X_WG2 = RigidTransform(grasp_pairs[0][1])
_, _, _, indices_enclosed1 = grasp_node.check_nonempty(current_manipuland_pcd, X_WG1)
_, _, _, indices_enclosed2 = grasp_node.check_nonempty(current_manipuland_pcd, X_WG2)

X_WO = RigidTransform(RotationMatrix(), np.mean(points, axis=1)) # object in world frame
X_OG1 = X_WO.inverse() @ X_WG1 # grasp 1 in object frame
X_OG2 = X_WO.inverse() @ X_WG2 # grasp 2 in object frame

display_center = [0.5, 0.0, 0.5]

# Get transform of camera in display frames
X_cam = RigidTransform(RotationMatrix(RollPitchYaw(-110 / 180 * np.pi, 0, np.pi/2)), [1.2, 0.0, 0.54])
X_displays = [RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0, -np.pi/4 * i)), display_center) for i in range(8)]
X_display_cams = [X_display.inverse() @ X_cam for X_display in X_displays]

# Get transform of camera in object frames for each display location and grasp
X_pick1_cams = [X_OG1 @ X_display_cam for X_display_cam in X_display_cams]
X_pick2_cams = [X_OG2 @ X_display_cam for X_display_cam in X_display_cams]

# visualize grasp and camera frames
# Translate pcd to object frame
points_in_object_frame = X_WO.inverse() @ points
pcd_in_object_frame = o3d.geometry.PointCloud()
pcd_in_object_frame.points = o3d.utility.Vector3dVector(points_in_object_frame.T)
pcd_in_object_frame.paint_uniform_color([1.0, 0.0, 0.0])

gripper_points_in_object_frame1 = X_OG1 @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]) @ grasp_node.hand_collision_model.to_pcd().T
gripper_points_in_object_frame2 = X_OG2 @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]) @ grasp_node.hand_collision_model.to_pcd().T

gripper_pcd_in_object_frame1 = o3d.geometry.PointCloud()
gripper_pcd_in_object_frame1.points = o3d.utility.Vector3dVector(gripper_points_in_object_frame1.T)
gripper_pcd_in_object_frame1.paint_uniform_color([0.0, 1.0, 0.0])
gripper_pcd_in_object_frame2 = o3d.geometry.PointCloud()
gripper_pcd_in_object_frame2.points = o3d.utility.Vector3dVector(gripper_points_in_object_frame2.T)
gripper_pcd_in_object_frame2.paint_uniform_color([0.0, 1.0, 0.0])

camera_triads1 = []
camera_triads2 = []
for i in range(8):
    camera_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    camera_frame1.transform(X_pick1_cams[i].GetAsMatrix4())
    camera_triads1.append(camera_frame1)

    camera_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    camera_frame2.transform(X_pick2_cams[i].GetAsMatrix4())
    camera_triads2.append(camera_frame2)

origin_triad = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
origin_triad.transform(np.eye(4))

# visualize grasp 1 and camera frames
grasp1_geometries = [pcd_in_object_frame, gripper_pcd_in_object_frame1, origin_triad] + camera_triads1
o3d.visualization.draw_geometries(grasp1_geometries)

# visualize grasp 2 and camera frames
grasp2_geometries = [pcd_in_object_frame, gripper_pcd_in_object_frame2, origin_triad] + camera_triads2
o3d.visualization.draw_geometries(grasp2_geometries)


voxel_map = VoxelMap(points_in_object_frame, voxel_size, visualize=True)

start = time.time()
confidence_improvment1 = voxel_map.get_improvement_from_observations(
    intrinsic_matrix, 
    [X_pick1_cams[i].GetAsMatrix4() for i in range(8)], 
    width_px, height_px, 
    occlusion_indices=indices_enclosed1,
    # occlusion_mask=np.array([[width_px//2 - 25 + j, i] for j in range(50) for i in range(height_px)]).T, 
    visualize=True,
    visualize_all=False
)
print(f"improvement 1 check took {time.time() - start} seconds")
print("confidence improvment 1", confidence_improvment1)

start = time.time()
confidence_improvment2 = voxel_map.get_improvement_from_observations(
    intrinsic_matrix, 
    [X_pick2_cams[i].GetAsMatrix4() for i in range(8)], 
    width_px, height_px, 
    occlusion_indices=indices_enclosed2,
    # occlusion_mask=np.array([[width_px//2 - 25 + j, i] for j in range(50) for i in range(height_px)]).T, 
    visualize=True,
    visualize_all=False
)
print(f"improvement 2 check took {time.time() - start} seconds")
print("confidence improvment 2", confidence_improvment2)

start = time.time()
for i in range(8):
    voxel_map.update_with_observation(
        intrinsic_matrix, 
        X_pick1_cams[i].GetAsMatrix4(), 
        width_px, height_px, 
        occlusion_indices=indices_enclosed1,
        # occlusion_mask=np.array([[width_px//2 - 25 + j, i] for j in range(50) for i in range(height_px)]).T, 
        visualize=False,
        visualize_all=False
    )
print(f"update 1 took {time.time() - start} seconds")

# voxel_map.visualize()

start = time.time()
for i in range(8):
    voxel_map.update_with_observation(
        intrinsic_matrix, 
        X_pick2_cams[i].GetAsMatrix4(), 
        width_px, height_px, 
        occlusion_indices=indices_enclosed2,
        # occlusion_mask=np.array([[width_px//2 - 25 + j, i] for j in range(50) for i in range(height_px)]).T, 
        visualize=False,
        visualize_all=False
    )
print(f"update 2 took {time.time() - start} seconds")

voxel_map.visualize()

print("voxel map stats", voxel_map.stats())