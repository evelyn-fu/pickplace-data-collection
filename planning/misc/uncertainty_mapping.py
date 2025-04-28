import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial import cKDTree
from pydrake.all import (
    PointCloud,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    Concatenate
)

class VoxelMap:
    def __init__(self, points: np.ndarray, voxel_size: float, confidence_threshold: float = 0.9, visualize=False):
        """
        Initializes a voxel map with signed distance values and confidence.

        Args:
            points (np.ndarray): A 3xn array of points in 3D space.
            voxel_size (float): The size of each voxel in the map.
        """
        if points.shape[0] != 3:
            raise ValueError("Input points must be a 3xn array.")

        self.voxel_size = voxel_size
        self.confidences = np.zeros(points.shape[1])
        self.confidence_threshold = confidence_threshold

        # Compute the center of the points
        self.center = np.mean(points, axis=1)
        print(f"Voxel map center: {self.center}")

        # Convert points to voxel indices
        self.points = points

        o3d_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(self.points.T)
        )
        o3d_pcd.estimate_normals()
        o3d_pcd.orient_normals_towards_camera_location(o3d_pcd.get_center())
        o3d_pcd.normals = o3d.utility.Vector3dVector(-np.asarray(o3d_pcd.normals))
        self.normals = np.asarray(o3d_pcd.normals).T
        triangle_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            o3d_pcd
        )
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(triangle_mesh)

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(tensor_mesh)

        if visualize:
            # visualize scene
            o3d.visualization.draw_geometries([triangle_mesh, o3d_pcd], point_show_normal=True)

        print(f"Initialized voxel map with {self.points.shape[1]} voxels.")

    def update_with_observation(
            self, 
            intrinsic_matrix, 
            extrinsic_matrix, 
            width_px, 
            height_px, 
            occlusion_mask=None, 
            occlusion_indices=None,
            visualize=False,
            visualize_all=False):
        """
        Updates the voxel map with a new observation.

        Args:
            intrinsic_matrix (np.ndarray): The intrinsic matrix of the camera
            extrinsic_matrix (np.ndarray): The extrinsic matrix of the camera
            width_px (int): The width of the camera image in pixels
            height_px (int): The height of the camera image in pixels
            occlusion_mask (np.ndarray): A 2xn array indicating occluded (x, y) pixels, optional
            occlusion_indices (np.ndarray): A 1xn array indicating indices of occluded points in the voxel map, optional
            visualize (bool): Whether to visualize the full update at the end
            visualize_all (bool): Whether to visualize each raycast step
        """

        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=np.linalg.inv(extrinsic_matrix),
            width_px=width_px,
            height_px=height_px,
        )

        raycast = self.scene.cast_rays(rays)

        # Step 3: Extract valid hits
        # raycast["t_hit"] is shape [H, W], float tensor of distances or inf
        t_hit = raycast["t_hit"].reshape((-1,))
        mask_hit = t_hit.isfinite()
        occlusion_mask_flat = occlusion_mask[1, :] * width_px + occlusion_mask[0, :] if occlusion_mask is not None else None
        if occlusion_mask_flat is not None:
            mask_hit[occlusion_mask_flat] = False

        # Step 4: Get ray origins and directions (shape [H*W, 3])
        rays_flat = rays.reshape((-1, 6))
        ray_origins = rays_flat[:, 0:3]
        ray_dirs = rays_flat[:, 3:6]

        # Step 5: Compute hit points only where hits occurred
        hit_ray_origins = ray_origins[mask_hit]
        hit_ray_dirs = ray_dirs[mask_hit]
        hit_t = t_hit[mask_hit].reshape((-1, 1))
        hit_points = hit_ray_origins + hit_t * hit_ray_dirs

        # Convert to NumPy for further processing
        hit_points_np = hit_points.numpy()

        # (Optional) if occlusion_mask is used
        if occlusion_indices is None:
            occlusion_indices = []

        # Step 6: Iterate over the hit points (minimized)
        points = []
        hit_points_viz = set()
        lines = []

        tree = cKDTree(self.points.T)
        neighbors_list = tree.query_ball_point(hit_points_np, r=self.voxel_size)
        occlusion_mask = np.ones(self.points.shape[1], dtype=bool)
        occlusion_mask[occlusion_indices] = False
        # for idx, hit_point in enumerate(hit_points_np):
        for idx, neighbor_inds in enumerate(neighbors_list):
            within_radius_indices = np.array(neighbor_inds)
            mask = ~np.isin(within_radius_indices, occlusion_indices)
            visible_indices = within_radius_indices[mask]
            if len(visible_indices) == 0:
                continue

            ray_dir = hit_ray_dirs[idx].numpy()
            surface_normals = self.normals[:, visible_indices].T
            weights = -np.dot(surface_normals, ray_dir)
            self.confidences[visible_indices] += weights

            if visualize:
                points += [hit_ray_origins[idx].numpy(), (hit_ray_origins[idx] + hit_ray_dirs[idx] * hit_t[idx]).numpy()]
                lines.append([len(points) - 2, len(points) - 1])
                hit_points_viz.update(visible_indices)
            if visualize_all:
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(self.points.T)
                hit_point_cloud = o3d.geometry.PointCloud()
                hit_point_cloud.points = o3d.utility.Vector3dVector(self.points[:, visible_indices].T)
                hit_point_cloud.paint_uniform_color([1, 0, 0])  # Red color for hit points
                point_cloud.paint_uniform_color([0.2, 0.2, 0.2])  # Gray color for all points
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(np.array([hit_ray_origins[idx].numpy(), (hit_ray_origins[idx] + hit_ray_dirs[idx] * hit_t[idx]).numpy()]))
                line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
                o3d.visualization.draw_geometries([line_set, point_cloud, hit_point_cloud])
        
        self.confidences = np.clip(self.confidences, 0, 1)

        if visualize:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np.array(points))
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([0, 0, 1])
            hit_point_cloud = o3d.geometry.PointCloud()
            hit_point_cloud.points = o3d.utility.Vector3dVector(self.points[:, list(hit_points_viz)].T)
            hit_point_cloud.paint_uniform_color([1, 0, 0])  # Red color for hit points  
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(self.points.T)
            point_cloud.paint_uniform_color([0.2, 0.2, 0.2])  # Gray color for all points
            o3d.visualization.draw_geometries([line_set, point_cloud, hit_point_cloud])
    
    def get_improvement_from_observations(
            self, 
            intrinsic_matrix, 
            extrinsic_matrices, 
            width_px, 
            height_px, 
            occlusion_indices=None,
            visualize=False,
            visualize_all=False):
        """
        Calculates the improvement in confidence from new observations.

        Args:
            intrinsic_matrix (np.ndarray): The intrinsic matrix of the camera
            extrinsic_matrix (np.ndarray): The extrinsic matrix of the camera
            width_px (int): The width of the camera image in pixels
            height_px (int): The height of the camera image in pixels
            occlusion_mask (np.ndarray): A 2xn array indicating occluded (x, y) pixels, optional
            visualize (bool): Whether to visualize the full update at the end
            visualize_all (bool): Whether to visualize each raycast step
        
        Returns:
            confidence_improvement (float): The average improvement in confidence over all voxels.
        """

        # Initialize for visualization
        points = []
        hit_points_viz = set()
        lines = []

        new_confidences = self.confidences.copy()
        for extrinsic_matrix in extrinsic_matrices:
            rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                intrinsic_matrix=intrinsic_matrix,
                extrinsic_matrix=np.linalg.inv(extrinsic_matrix),
                width_px=width_px,
                height_px=height_px,
            )

            raycast = self.scene.cast_rays(rays)

            # Step 3: Extract valid hits
            # raycast["t_hit"] is shape [H, W], float tensor of distances or inf
            t_hit = raycast["t_hit"].reshape((-1,))
            mask_hit = t_hit.isfinite()

            # Step 4: Get ray origins and directions (shape [H*W, 3])
            rays_flat = rays.reshape((-1, 6))
            ray_origins = rays_flat[:, 0:3]
            ray_dirs = rays_flat[:, 3:6]

            # Step 5: Compute hit points only where hits occurred
            hit_ray_origins = ray_origins[mask_hit]
            hit_ray_dirs = ray_dirs[mask_hit]
            hit_t = t_hit[mask_hit].reshape((-1, 1))
            hit_points = hit_ray_origins + hit_t * hit_ray_dirs

            # Convert to NumPy for further processing
            hit_points_np = hit_points.numpy()

            # (Optional) if occlusion_mask is used
            if occlusion_indices is None:
                occlusion_indices = []

            # Step 6: Iterate over the hit points (minimized)

            tree = cKDTree(self.points.T)
            neighbors_list = tree.query_ball_point(hit_points_np, r=self.voxel_size)
            occlusion_mask = np.ones(self.points.shape[1], dtype=bool)
            occlusion_mask[occlusion_indices] = False
            # for idx, hit_point in enumerate(hit_points_np):
            for idx, neighbor_inds in enumerate(neighbors_list):
                within_radius_indices = np.array(neighbor_inds)
                mask = ~np.isin(within_radius_indices, occlusion_indices)
                visible_indices = within_radius_indices[mask]
                if len(visible_indices) == 0:
                    continue

                ray_dir = hit_ray_dirs[idx].numpy()
                surface_normals = self.normals[:, visible_indices].T
                weights = -np.dot(surface_normals, ray_dir)
                new_confidences[visible_indices] += weights

                if visualize:
                    points += [hit_ray_origins[idx].numpy(), (hit_ray_origins[idx] + hit_ray_dirs[idx] * hit_t[idx]).numpy()]
                    lines.append([len(points) - 2, len(points) - 1])
                    hit_points_viz.update(visible_indices)
                if visualize_all:
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(self.points.T)
                    hit_point_cloud = o3d.geometry.PointCloud()
                    hit_point_cloud.points = o3d.utility.Vector3dVector(self.points[:, visible_indices].T)
                    hit_point_cloud.paint_uniform_color([1, 0, 0])  # Red color for hit points
                    point_cloud.paint_uniform_color([0.2, 0.2, 0.2])  # Gray color for all points
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(np.array([hit_ray_origins[idx].numpy(), (hit_ray_origins[idx] + hit_ray_dirs[idx] * hit_t[idx]).numpy()]))
                    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
                    o3d.visualization.draw_geometries([line_set, point_cloud, hit_point_cloud])
        
        new_confidences = np.clip(new_confidences, 0, 1)
        confidence_improvement = np.mean(new_confidences) - np.mean(self.confidences)

        if visualize:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np.array(points))
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([0, 0, 1])
            hit_point_cloud = o3d.geometry.PointCloud()
            hit_point_cloud.points = o3d.utility.Vector3dVector(self.points[:, list(hit_points_viz)].T)
            hit_point_cloud.paint_uniform_color([1, 0, 0])  # Red color for hit points  
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(self.points.T)
            point_cloud.paint_uniform_color([0.2, 0.2, 0.2])  # Gray color for all points
            o3d.visualization.draw_geometries([line_set, point_cloud, hit_point_cloud])

        return confidence_improvement

    def visualize(self, X_cam=None):
        """
        Visualizes the voxel map using open3d
        """
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(self.points.T)

        base_color = np.array([[0.2, 0.2, 0.2]])
        target_color = np.array([[1, 0, 0]])
        colors = base_color + np.expand_dims(np.clip(self.confidences, 0, 1), axis=1) @ (target_color - base_color) # red more confident, gray less confident
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        geometries = [point_cloud]
        if X_cam is not None:
            observation_direction = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            observation_direction.transform(X_cam)
            geometries.append(observation_direction)

        o3d.visualization.draw_geometries(geometries)

    def fully_observed(self):
        """
        Returns True if the map is fully observed
        """
        return np.all(self.confidences > self.confidence_threshold)
    
    def stats(self):
        """
        Returns the coverage statistics of the voxel map
        """

        clipped_confidences = np.clip(self.confidences, 0, 1)

        # compute the number of voxels in the largest connected component of unobserved voxels
        unobserved_mask = clipped_confidences < self.confidence_threshold
        tree = cKDTree(self.points.T)

        # Find connected components of unobserved voxels
        visited = np.zeros(len(clipped_confidences), dtype=bool)
        largest_component_size = 0

        for i in range(len(clipped_confidences)):
            if not unobserved_mask[i] or visited[i]:
                continue

            # Perform a breadth-first search (BFS) to find connected components
            queue = [i]
            component_size = 0

            while queue:
                current = queue.pop(0)
                if visited[current]:
                    continue

                visited[current] = True
                component_size += 1

                # Find neighbors within voxel_size
                neighbors = tree.query_ball_point(self.points[:, current], r=self.voxel_size)
                for neighbor in neighbors:
                    if unobserved_mask[neighbor] and not visited[neighbor]:
                        queue.append(neighbor)

            largest_component_size = max(largest_component_size, component_size)

        print(f"Largest connected component of unobserved voxels: {largest_component_size}")

        return {
            "mean_confidence": np.mean(clipped_confidences),
            "max_confidence": np.max(clipped_confidences),
            "min_confidence": np.min(clipped_confidences),
            "percent_under_threshold": np.sum(clipped_confidences < self.confidence_threshold) / len(clipped_confidences),
            "percent_fully_observed": np.sum(clipped_confidences > self.confidence_threshold) / len(clipped_confidences),
            "largest_unobserved_component_percentage": largest_component_size / len(clipped_confidences),
        }