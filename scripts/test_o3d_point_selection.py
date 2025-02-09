import open3d as o3d
import numpy as np

def select_points_from_pcd(pcd):
    """
    Opens an interactive Open3D window for point picking on the given point cloud.
    
    Instructions:
      - Hold SHIFT and left-click on points to select them.
      - When done, press 'q' to exit the window.
      
    Args:
        pcd (o3d.geometry.PointCloud): The point cloud for selection.
    
    Returns:
        np.ndarray: An array of selected points of shape (N, 3).
    """
    # Print instructions.
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Manual grasp selector")
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    point_indices = np.asarray(vis.get_picked_points())
    points = np.asarray(pcd.points)[point_indices]
    return points

def compute_coordinate_frame_for_pair(p1, p2, pcd, k_neighbors=30):
    """
    Computes a coordinate frame for a pair of points.
    
    The frame is defined as:
      - Origin: the midpoint of p1 and p2.
      - y axis: unit vector from p1 to p2.
      - z axis: computed from the average (smoothed) normal of the point cloud at the midpoint,
                then reversed and made orthogonal to y.
      - x axis: chosen such that (x, y, z) form a right-handed coordinate system.
    
    The smooth normal is estimated by finding the k nearest neighbors to the midpoint
    in the point cloud and averaging their normals.
    
    Args:
        p1, p2 (array-like): The two 3D points.
        pcd (o3d.geometry.PointCloud): The full point cloud (its normals must be computed).
        k_neighbors (int): Number of nearest neighbors to use for smoothing.
    
    Returns:
        np.ndarray: A 4x4 homogeneous transformation matrix representing the coordinate frame.
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    midpoint = (p1 + p2) / 2.0

    # Compute y-axis: direction from p1 to p2 (normalized)
    y_axis = p2 - p1
    norm_y = np.linalg.norm(y_axis)
    if norm_y < 1e-6:
        raise ValueError("The two points are too close together.")
    y_axis = y_axis / norm_y

    # Build a KDTree for the point cloud to find nearest neighbors to the midpoint.
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    [ k, idx, _ ] = pcd_tree.search_knn_vector_3d(midpoint, k_neighbors)
    if k == 0:
        raise ValueError("No neighbors found near the midpoint.")
    
    normals = np.asarray(pcd.normals)
    # Average the normals of the k neighbors to obtain a smooth normal estimate.
    avg_normal = np.mean(normals[idx, :], axis=0)
    norm_avg = np.linalg.norm(avg_normal)
    if norm_avg < 1e-6:
        raise ValueError("Computed average normal is too small.")
    avg_normal = avg_normal / norm_avg

    # Define the z axis to be opposite to the averaged normal.
    z_raw = -avg_normal
    # Project z_raw onto the plane orthogonal to y_axis:
    z_axis = z_raw - np.dot(z_raw, y_axis) * y_axis
    norm_z = np.linalg.norm(z_axis)
    if norm_z < 1e-6:
        raise ValueError("Degenerate z-axis after orthogonalization.")
    z_axis = z_axis / norm_z

    # Compute x-axis such that the frame is right handed: x = y cross z.
    x_axis = np.cross(y_axis, z_axis)
    norm_x = np.linalg.norm(x_axis)
    if norm_x < 1e-6:
        raise ValueError("Degenerate x-axis computed.")
    x_axis = x_axis / norm_x

    # Assemble the rotation matrix using x, y, and z as its columns.
    R = np.column_stack((x_axis, y_axis, z_axis))

    # Create a 4x4 transformation matrix.
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = midpoint

    return T

def compute_coordinate_frames(selected_points, pcd, k_neighbors=30):
    """
    Splits the selected points into pairs and computes a coordinate frame for each pair.
    
    If an odd number of points is selected, the last point is disregarded (with a warning).
    
    Args:
        selected_points (np.ndarray): An (N, 3) array of points.
        pcd (o3d.geometry.PointCloud): The point cloud (with normals computed).
        k_neighbors (int): Number of neighbors to use for smoothing the normal estimate.
        
    Returns:
        list of np.ndarray: A list of 4x4 transformation matrices (one for each pair).
    """
    num_points = selected_points.shape[0]
    if num_points % 2 == 1:
        print("Warning: odd number of selected points; disregarding the last point.")
        num_points -= 1  # Drop the last point

    frames = []
    for i in range(0, num_points, 2):
        p1 = selected_points[i]
        p2 = selected_points[i + 1]
        T = compute_coordinate_frame_for_pair(p1, p2, pcd, k_neighbors)
        frames.append(T)
    return frames

# === Main Script ===
if __name__ == "__main__":
    # Load an example point cloud provided by Open3D.
    pcd_data = o3d.data.PCDPointCloud()
    pcd = o3d.io.read_point_cloud(pcd_data.path)

    # Compute normals if not already available.
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    # --- Step 1: Let the user select points ---
    selected_pts = select_points_from_pcd(pcd)
    print("Selected Points:")
    print(selected_pts)

    if selected_pts.shape[0] < 2:
        print("Not enough points were selected to form at least one pair.")
    else:
        # --- Step 2 & 3: Process pairs and compute coordinate frames ---
        frames = compute_coordinate_frames(selected_pts, pcd, k_neighbors=30)
        print("\nComputed Coordinate Frames (4x4 transformation matrices):")
        for i, T in enumerate(frames):
            print(f"\nFrame {i}:")
            print(T)

        # Optionally, visualize the point cloud and the coordinate frames.
        frame_meshes = []
        for T in frames:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            frame.transform(T)
            frame_meshes.append(frame)
        o3d.visualization.draw_geometries([pcd] + frame_meshes, point_show_normal=True)
