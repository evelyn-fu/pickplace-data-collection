import open3d as o3d
import numpy as np

def select_points_from_pcd(pcd):
    """
    Opens an interactive Open3D window for point picking on the given point cloud.
    
    Instructions:
      - Hold SHIFT and left-click on points to select them.
      - **IMPORTANT:** For each coordinate frame, select 3 points in this order:
            1. First point (p1) for the pair.
            2. Second point (p2) for the pair.
            3. A third point (p3) indicating the rough desired direction of the x axis.
      - When done, press 'q' to exit the window.
      
    Args:
        pcd (o3d.geometry.PointCloud): The point cloud for selection.
    
    Returns:
        np.ndarray: An array of selected points (shape: (N, 3)).
    """
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

def compute_coordinate_frame_for_triplet(p1, p2, p3, eps=1e-6):
    """
    Computes a coordinate frame based on three points:
    
      - **Origin:** Midpoint of p1 and p2.
      - **y axis:** Normalized vector from p1 to p2.
      - **x axis:** Obtained by taking the vector from the midpoint to p3,
                  projecting it onto the plane perpendicular to the y axis, and normalizing.
      - **z axis:** Computed via the cross product: x_axis × y_axis,
                  ensuring a right-handed coordinate frame.
    
    Args:
        p1, p2, p3 (array-like): Three 3D points.
        eps (float): A small value to check for degeneracies.
    
    Returns:
        np.ndarray: A 4x4 homogeneous transformation matrix representing the coordinate frame.
    """
    # Ensure inputs are NumPy arrays.
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    
    # Compute the midpoint between p1 and p2 (the frame origin).
    midpoint = (p1 + p2) / 2.0
    
    # Compute the y axis: from p1 to p2.
    y_axis = p2 - p1
    norm_y = np.linalg.norm(y_axis)
    if norm_y < eps:
        raise ValueError("p1 and p2 are too close together to define a y direction.")
    y_axis = y_axis / norm_y
    
    # Compute a rough x direction from p3.
    raw_x = p3 - midpoint
    # Project raw_x onto the plane perpendicular to y_axis.
    proj = np.dot(raw_x, y_axis) * y_axis
    x_proj = raw_x - proj
    norm_x = np.linalg.norm(x_proj)
    if norm_x < eps:
        raise ValueError("The third point does not provide a valid x direction.")
    x_axis = x_proj / norm_x
    
    # Compute z axis as the cross product to ensure a right-handed frame.
    # Recall: in a right-handed coordinate system, x_axis × y_axis = z_axis.
    z_axis = np.cross(x_axis, y_axis)
    norm_z = np.linalg.norm(z_axis)
    if norm_z < eps:
        raise ValueError("Computed z axis is degenerate.")
    z_axis = z_axis / norm_z

    # Assemble the rotation matrix with columns: [x_axis, y_axis, z_axis].
    R = np.column_stack((x_axis, y_axis, z_axis))
    
    # Form the 4x4 transformation matrix.
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = midpoint
    return T

def compute_coordinate_frames(selected_points):
    """
    Splits the selected points into groups of three and computes a coordinate frame for each group.
    
    If the total number of selected points is not a multiple of three, a warning is printed and the extra
    points are disregarded.
    
    Args:
        selected_points (np.ndarray): An (N, 3) array of selected points.
    
    Returns:
        list of np.ndarray: A list of 4x4 transformation matrices (one for each coordinate frame).
    """
    num_points = selected_points.shape[0]
    if num_points % 3 != 0:
        print("Warning: The number of selected points is not a multiple of 3. Extra points will be disregarded.")
        num_points = num_points - (num_points % 3)
    
    frames = []
    for i in range(0, num_points, 3):
        p1 = selected_points[i]
        p2 = selected_points[i + 1]
        p3 = selected_points[i + 2]
        T = compute_coordinate_frame_for_triplet(p1, p2, p3)
        frames.append(T)
    return frames

if __name__ == "__main__":
    # Load an example point cloud provided by Open3D.
    pcd_data = o3d.data.PCDPointCloud()
    pcd = o3d.io.read_point_cloud(pcd_data.path)

    # --- Step 1: Let the user select points ---
    selected_pts = select_points_from_pcd(pcd)
    print("Selected points:")
    print(selected_pts)

    if selected_pts.shape[0] < 3:
        print("Not enough points were selected to form at least one coordinate frame (need 3 points per frame).")
    else:
        # --- Step 2: Compute coordinate frames from the selected triplets ---
        frames = compute_coordinate_frames(selected_pts)
        print("\nComputed Coordinate Frames (4x4 transformation matrices):")
        for i, T in enumerate(frames):
            print(f"\nFrame {i}:")
            print(T)

        # --- Optional: Visualize the point cloud and the coordinate frames ---
        frame_meshes = []
        for T in frames:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            frame.transform(T)
            frame_meshes.append(frame)
        o3d.visualization.draw_geometries([pcd] + frame_meshes)
