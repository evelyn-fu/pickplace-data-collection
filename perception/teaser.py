# from https://github.com/MIT-SPARK/TEASER-plusplus/blob/master/examples/teaser_python_fpfh_icp/example.py
#!/bin/python3

import argparse
import copy
from typing import Optional, Tuple

import numpy as np
import teaserpp_python
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from perception.icp import get_initial_alignment

def pcd2xyz(pcd):
    return np.asarray(pcd.points).T

def extract_fpfh(pcd, voxel_size):
  radius_normal = voxel_size * 2
  pcd.estimate_normals(
      o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

  radius_feature = voxel_size * 5
  fpfh = o3d.pipelines.registration.compute_fpfh_feature(
      pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
  return np.array(fpfh.data).T

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
  feat1tree = cKDTree(feat1)
  dists, nn_inds = feat1tree.query(feat0, k=knn, workers=-1)
  if return_distance:
    return nn_inds, dists
  else:
    return nn_inds

def find_correspondences(feats0, feats1, mutual_filter=True):
  nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
  corres01_idx0 = np.arange(len(nns01))
  corres01_idx1 = nns01

  if not mutual_filter:
    return corres01_idx0, corres01_idx1

  nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
  corres10_idx1 = np.arange(len(nns10))
  corres10_idx0 = nns10

  mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
  corres_idx0 = corres01_idx0[mutual_filter]
  corres_idx1 = corres01_idx1[mutual_filter]

  return corres_idx0, corres_idx1

def get_teaser_solver(noise_bound):
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.inlier_selection_mode = \
        teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    solver_params.rotation_tim_graph = \
        teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    solver_params.rotation_estimation_algorithm = \
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 10000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    return solver

def create_homogenous_transform(
    rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T

def icp(model: np.ndarray, scene: np.ndarray, voxel_size=0.005):
    model_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model.T))
    scene_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene.T))
    
    model_pcd.paint_uniform_color([0.0, 0.0, 1.0]) # show model_pcd in blue
    scene_pcd.paint_uniform_color([1.0, 0.0, 0.0]) # show scene_pcd in red

    # Get initial alignment
    init_transform = get_initial_alignment(model_pcd, scene_pcd)
    print(f"Initial transform: {init_transform}")
    # Try original orientation and 180° rotations around each axis
    rotations = [
        np.eye(4),  # Original
        create_homogenous_transform(R.from_rotvec([np.pi, 0, 0]).as_matrix(), np.zeros(3)),  # 180° around X
        create_homogenous_transform(R.from_rotvec([0, np.pi, 0]).as_matrix(), np.zeros(3)),  # 180° around Y
        create_homogenous_transform(R.from_rotvec([0, 0, np.pi]).as_matrix(), np.zeros(3))   # 180° around Z
    ]
    
    best_transform = None
    best_fitness = 0
    
    j = 0
    best_ind = 0
    for rot in rotations:
        # Combine initial alignment with current rotation
        test_transform = np.copy(init_transform)
        test_transform[:3, :3] = rot[:3, :3] @ init_transform[:3, :3]

        # Apply test transformation
        model_transformed = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model.T))
        model_transformed.transform(test_transform)

        model_xyz = pcd2xyz(model_transformed) # np array of size 3 by N
        scene_xyz = pcd2xyz(scene_pcd) # np array of size 3 by M

        # extract FPFH features
        model_feats = extract_fpfh(model_transformed,voxel_size)
        scene_feats = extract_fpfh(scene_pcd,voxel_size)

        # establish correspondences by nearest neighbour search in feature space
        corrs_model, corrs_scene = find_correspondences(
            model_feats, scene_feats, mutual_filter=True)
        model_corr = model_xyz[:,corrs_model] # np array of size 3 by num_corrs
        scene_corr = scene_xyz[:,corrs_scene] # np array of size 3 by num_corrs

        num_corrs = model_corr.shape[1]
        print(f'FPFH generates {num_corrs} putative correspondences.')

        # visualize the point clouds together with feature correspondences
        points = np.concatenate((model_corr.T,scene_corr.T),axis=0)
        lines = []
        for i in range(num_corrs):
            lines.append([i,i+num_corrs])
        colors = [[0, 1, 0] for i in range(len(lines))] # lines are shown in green
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_plotly([model_transformed,scene_pcd,line_set])

        # robust global registration using TEASER++
        NOISE_BOUND = voxel_size
        teaser_solver = get_teaser_solver(NOISE_BOUND)
        teaser_solver.solve(model_corr,scene_corr)
        solution = teaser_solver.getSolution()
        R_teaser = solution.rotation
        t_teaser = solution.translation
        T_teaser = create_homogenous_transform(R_teaser,t_teaser)

        # local refinement using ICP
        icp_sol = o3d.pipelines.registration.registration_icp(
            model_transformed, scene_pcd, NOISE_BOUND, T_teaser,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        print("ICP solution:", icp_sol.transformation)

        # Update best result if current result is better
        if icp_sol.fitness > best_fitness:
            best_fitness = icp_sol.fitness
            best_transform = icp_sol.transformation @ test_transform
            best_ind = j
        j += 1

    model_solution_final = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model.T))
    model_solution_final.transform(best_transform)
    model_solution_final.paint_uniform_color([0.0, 1.0, 0.0]) # show in green

    # o3d.visualization.draw_plotly([model_solution_final, scene_pcd])

    print("Transformed model center:", model_solution_final.get_center())
    print("Scene center:", scene_pcd.get_center())
    print(f"Best fitness: {best_fitness}")
    print(f"Best transform: {best_transform}")
    print(f"Best index: {best_ind}")
    
    return best_transform