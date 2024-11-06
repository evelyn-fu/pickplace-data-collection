# Imports
import numpy as np
from pydrake.all import PointCloud, Rgba, RigidTransform, RotationMatrix, StartMeshcat
from scipy.spatial import KDTree

def least_squares_transform(scene, model) -> RigidTransform:
    """
    Calculates the least-squares best-fit transform that maps corresponding
    points scene to model.
    Args:
      scene: 3xN numpy array of corresponding points
      model: 3xM numpy array of corresponding points
    Returns:
      X_BA: A RigidTransform object that maps point_cloud_A on to point_cloud_B
            such that
                        X_BA.multiply(model) ~= scene,
    """

    # Ensure that the input arrays are 3xN
    assert scene.shape[0] == 3 and model.shape[0] == 3, "Scene and model must be 3xN arrays"

    s_bar = np.mean(scene, axis=1)
    m_bar = np.mean(model, axis=1)
    
    # 2. Center the points by subtracting the centroids
    s_centered = scene - s_bar[:, np.newaxis]
    m_centered = model - m_bar[:, np.newaxis]
    
    # 3. Compute the covariance matrix
    W = np.dot(s_centered, m_centered.T)
    
    U, S, V_T = np.linalg.svd(W)
    D = np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(U @ V_T)]])
    R = U @ D @ V_T
    p = s_bar - R @ m_bar
    
    # 7. Create the RigidTransform
    X_BA = RigidTransform(RotationMatrix(R), p)

    return X_BA

def nearest_neighbors(scene, model):
    """
    Find the nearest (Euclidean) neighbor in model for each
    point in scene
    Args:
        scene: 3xN numpy array of points
        model: 3xM numpy array of points
    Returns:
        distances: (N, ) numpy array of Euclidean distances from each point in
            scene to its nearest neighbor in model.
        indices: (N, ) numpy array of the indices in model of each
            scene point's nearest neighbor - these are the c_i's
    """
    kdtree = KDTree(model.T)

    distances, indices = kdtree.query(scene.T, k=1)

    return distances.flatten(), indices.flatten()

def icp(scene, model, max_iterations=20, tolerance=1e-3):
    """
    Perform ICP to return the correct relative transform between two set of points.
    Args:
        scene: 3xN numpy array of points
        model: 3xM numpy array of points
        max_iterations: max amount of iterations the algorithm can perform.
        tolerance: tolerance before the algorithm converges.
    Returns:
      X_BA: A RigidTransform object that maps point_cloud_A on to point_cloud_B
            such that
                        X_BA.multiply(model) ~= scene,
      mean_error: Mean of all pairwise distances.
      num_iters: Number of iterations it took the ICP to converge.
    """
    X_BA = RigidTransform()

    mean_error = 0
    num_iters = 0
    prev_error = 0

    while True:
        num_iters += 1  
          
        curr_model = X_BA @ model
        distances, indices = nearest_neighbors(scene, curr_model)
        mean_error = np.mean(distances)
        
        X_BA = least_squares_transform(scene, curr_model[...,indices]) @ X_BA
        if abs(mean_error - prev_error) < tolerance or num_iters >= max_iterations:
            break

        prev_error = mean_error

    return X_BA, mean_error, num_iters