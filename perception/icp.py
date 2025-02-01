import numpy as np
import open3d as o3d
import copy
from scipy.spatial.transform import Rotation as R

# Before RANSAC, add initial alignment
def get_initial_alignment(source, target):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    # Center both point clouds
    source_center = np.mean(np.asarray(source_copy.points), axis=0)
    target_center = np.mean(np.asarray(target_copy.points), axis=0)
    source_centered = source_copy.translate(-source_center)
    target_centered = target_copy.translate(-target_center)
    
    # Get principal axes
    source_cov = np.cov(np.asarray(source_centered.points).T)
    target_cov = np.cov(np.asarray(target_centered.points).T)
    source_eigvals, source_eigvecs = np.linalg.eigh(source_cov)
    target_eigvals, target_eigvecs = np.linalg.eigh(target_cov)
    
    # Sort by eigenvalues to match corresponding axes
    source_order = source_eigvals.argsort()[::-1]
    target_order = target_eigvals.argsort()[::-1]
    source_eigvecs = source_eigvecs[:, source_order]
    target_eigvecs = target_eigvecs[:, target_order]
    
    # Build rotation matrix
    R = target_eigvecs @ source_eigvecs.T
    
    # Ensure it's a valid rotation matrix (right-handed)
    if np.linalg.det(R) < 0:
        source_eigvecs[:, 2] *= -1
        R = target_eigvecs @ source_eigvecs.T
    
    # Build full transformation
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = target_center - source_center
    
    return T

# Try multiple initial alignments
def try_multiple_alignments(source, target, distance_threshold=0.05):
    # Get initial alignment
    init_transform = get_initial_alignment(source, target)
    
    # Try original orientation and 180° rotations around each axis
    rotations = [
        np.eye(4),  # Original
        R.from_rotvec([np.pi, 0, 0]).as_matrix(),  # 180° around X
        R.from_rotvec([0, np.pi, 0]).as_matrix(),  # 180° around Y
        R.from_rotvec([0, 0, np.pi]).as_matrix()   # 180° around Z
    ]
    
    best_transform = None
    best_fitness = 0
    
    for rot in rotations:
        # Combine initial alignment with current rotation
        test_transform = np.copy(init_transform)
        test_transform[:3, :3] = rot[:3, :3] @ init_transform[:3, :3]
        
        # Apply test transformation
        source_transformed = copy.deepcopy(source)
        source_transformed.transform(test_transform)
        
        # Compute FPFH features
        source_transformed.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        radius_feature = 0.3
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_transformed,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        
        # Run RANSAC
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_transformed, target, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            ransac_n=30,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(0.5),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 100)
        )
        
        # Refine with ICP
        icp_result = o3d.pipelines.registration.registration_icp(
            source_transformed, target, distance_threshold,
            result.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )
        
        # Update best result if current result is better
        if icp_result.fitness > best_fitness:
            best_fitness = icp_result.fitness
            best_result = icp_result
            best_transform = test_transform @ icp_result.transformation
    
    return best_transform, best_fitness