from planning.grasp import GraspListener, GraspType
import numpy as np
import os
import copy
import open3d as o3d
from pydrake.all import (
    RigidTransform,
    RollPitchYaw,
    PointCloud
)


# Bin camera
# x_bin_rgb = RigidTransform(np.loadtxt("/home/real2sim/calibrations/bin_calibration_2_7_daniilidis.txt"))
# x_depth_rgb_bin = RigidTransform([[0.999968,    0.00149319,  0.00783427,    0.0147784],
#                                     [-0.00146555,   0.999993, -0.00353279, -4.93721e-05],
#                                     [-0.00783949, 0.00352119,    0.999963,  0.000204544],
#                                     [          0,          0,           0,            1]])
x_bin_camera = RigidTransform(RollPitchYaw(-164.69831287,  -35.83297034,  -99.44115857), [-0.0574518,  0.874365 ,  0.332985])

# bin_pcd_xyzs = np.load(os.path.abspath(os.path.join(__file__ ,"../../bin_contents.npy")))
bin_pcd_xyzs = np.load(os.path.abspath(os.path.join(__file__ ,"../../dual_grasp_pcd_bowl.npy")))
bin_pcd_normals = np.load(os.path.abspath(os.path.join(__file__ ,"../../dual_grasp_normals_bowl.npy")))
bin_pcd = PointCloud(bin_pcd_xyzs.shape[1], )
bin_pcd.mutable_xyzs()[:] = bin_pcd_xyzs
bin_pcd.EstimateNormals(radius=0.1, num_closest=30)
bin_pcd.mutable_normals()[:] = bin_pcd_normals
# bin_pcd.EstimateNormals(radius=0.1, num_closest=30)
# bin_pcd.FlipNormalsTowardPoint(x_bin_camera.translation())

# scene_pcd_xyzs = np.load(os.path.abspath(os.path.join(__file__ ,"../../bin_background.npy")))
scene_pcd_xyzs = np.load(os.path.abspath(os.path.join(__file__ ,"../../dual_grasp_background_bowl.npy")))
scene_pcd_normals = np.load(os.path.abspath(os.path.join(__file__ ,"../../dual_grasp_background_normals_bowl.npy")))
scene_pcd = PointCloud(scene_pcd_xyzs.shape[1])
scene_pcd.mutable_xyzs()[:] = scene_pcd_xyzs
scene_pcd.EstimateNormals(radius=0.1, num_closest=30)
scene_pcd.mutable_normals()[:] = scene_pcd_normals
# bin_pcd.EstimateNormals(radius=0.1, num_closest=30)
# bin_pcd.FlipNormalsTowardPoint(x_bin_camera.translation())

pcd = bin_pcd.xyzs().T # Shape (N,3)
pcd_normals = bin_pcd.normals().T # Shape (N,3)
o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
o3d_pcd.paint_uniform_color([0,0,1])
o3d_pcd.normals = o3d.utility.Vector3dVector(pcd_normals)
# o3d.visualization.draw_geometries([o3d_pcd], point_show_normal=True, window_name="merged pcd")

gripper_model_path = "file://./home/evelyn/sources/Real2SimObjectManipulation/models/schunk_wsg_50_welded_fingers_w_buffer.sdf"
grasp_node = GraspListener(gripper_model_path=gripper_model_path)

ONLINE_VOXEL_RADIUS = 0.005

# Planning first grasping trajectory
grasp_node.compute_candidate_grasps(
    bin_pcd,
    scene_pcd,
    candidate_num=1,
    num_samples=30,
    random_seed=np.random.randint(1000),
    grasp_type=GraspType.PAIR,
    split_ratio_threshold=0.15,
    # is_manual=True,
    use_extra_buffer=True
)

grasp_pairs, grasp_costs = grasp_node.get_best_grasps()
grasp_pairs = copy.deepcopy(grasp_pairs)
grasp_costs = copy.deepcopy(grasp_costs)
if len(grasp_pairs) < 10:
    print("not enough pairs, checking more points", len(grasp_pairs))
    grasp_node.compute_candidate_grasps(
        bin_pcd,
        scene_pcd,
        candidate_num=1,
        num_samples=30,
        random_seed=np.random.randint(1000),
        grasp_type=GraspType.PAIR,
        split_ratio_threshold=0.15,
        # is_manual=True,
        use_extra_buffer=True,
        # check_backwards=True
    )
    new_grasp_pairs, new_grasp_costs = grasp_node.get_best_grasps()
    print("num new grasps:", len(new_grasp_pairs))
    grasp_pairs += new_grasp_pairs
    grasp_costs += new_grasp_costs

    print(grasp_costs)
    new_sorted_inds = np.argsort(np.array(grasp_costs))
    sorted_grasp_pairs = [grasp_pairs[idx] for idx in new_sorted_inds]

print("grasp pairs:", len(sorted_grasp_pairs))
X_WG1, X_WG2 = sorted_grasp_pairs[0]

manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bin_pcd.xyzs().T))
manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

gripper_xyzs = grasp_node.hand_extra_buffer_collision_model.to_pcd()
gripper_cloud1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(ONLINE_VOXEL_RADIUS).transform(
    (X_WG1 @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4())
)
gripper_cloud1.paint_uniform_color([1.0, 0.0, 0.0])
gripper_cloud2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(ONLINE_VOXEL_RADIUS).transform(
    (X_WG2 @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4())
)
gripper_cloud2.paint_uniform_color([0.0, 1.0, 0.0])

frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(X_WG1)
frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(X_WG2)

viz_geoms = [manipuland_cloud, gripper_cloud1, frame1, gripper_cloud2, frame2]
o3d.visualization.draw_geometries(viz_geoms)