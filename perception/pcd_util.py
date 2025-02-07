import numpy as np
from pydrake.perception import PointCloud
from scipy.spatial import cKDTree

def compute_principal_minor_components(pcd):
    cov = np.cov(pcd.T)
    eigval, eigvec = np.linalg.eig(cov)

    order = eigval.argsort()
    principal_component = eigvec[:, order[-1]]
    secondary_component = eigvec[:, order[1]]
    minor_component = eigvec[:, order[0]]

    return principal_component, secondary_component, minor_component

def crop_connected_points(cloud: PointCloud, center: np.ndarray, radius: float, voxel_radius: float = 0.01) -> PointCloud:
    """Crops a pointcloud to only include points that are continuously connected to the center point.
    
    Args:
        cloud: Drake PointCloud to crop
        center: (3,) array specifying center point to crop around
        radius: Maximum radius from center to consider points
        voxel_radius: Maximum distance between connected points (default 0.01m)
    
    Returns:
        Cropped PointCloud containing only connected points
    """
    # Convert to numpy array for processing
    xyz = np.asarray(cloud.xyzs())
    
    # Get points within initial radius
    dists = np.linalg.norm(xyz.T - center, axis=1)
    initial_mask = dists < radius
    
    if not np.any(initial_mask):
        # Return empty cloud if no points in radius
        return PointCloud(0)
        
    # Build KD tree for efficient nearest neighbor queries
    tree = cKDTree(xyz.T)
    
    # Start with point closest to center
    center_idx = np.argmin(dists)
    connected = {center_idx}
    to_check = {center_idx}
    
    # Iteratively find connected points
    while to_check:
        current = to_check.pop()
        
        # Find neighbors within voxel_radius * 1.1
        neighbors = tree.query_ball_point(xyz.T[current], voxel_radius * 5)
        
        # Add unvisited neighbors that are within radius
        for n in neighbors:
            if n not in connected and dists[n] < radius:
                connected.add(n)
                to_check.add(n)
    
    # Create mask for connected points
    connected_mask = np.zeros(len(xyz.T), dtype=bool)
    connected_mask[list(connected)] = True
    
    # Create new pointcloud with only connected points
    new_cloud = PointCloud(np.sum(connected_mask))
    new_cloud.mutable_xyzs()[:] = xyz[:, connected_mask]
        
    return new_cloud
