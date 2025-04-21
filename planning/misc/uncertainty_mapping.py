import numpy as np
import open3d as o3d
from tqdm import tqdm
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
    def __init__(self, points: np.ndarray, voxel_size: float, confidence_threshold: float = 0.9):
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
        
        self.pcd = PointCloud(self.points.shape[1])
        self.pcd.mutable_xyzs()[:] = self.points
        self.pcd = self.pcd.VoxelizedDownSample(voxel_size=voxel_size)
        self.pcd.EstimateNormals(radius=0.1, num_closest=30)
        self.pcd.FlipNormalsTowardPoint(self.center)
        self.pcd.mutable_normals()[:] = -self.pcd.normals() # flip away from center
        self.normals = self.pcd.normals()

        o3d_pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(self.points.T)
        )
        o3d_pcd.normals = o3d.utility.Vector3dVector(self.normals.T)
        triangle_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            o3d_pcd
        )
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(triangle_mesh)

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(tensor_mesh)

        print(f"Initialized voxel map with {self.points.shape[1]} voxels.")

    def update_with_observation(
            self, 
            intrinsic_matrix, 
            extrinsic_matrix, 
            width_px, 
            height_px, 
            voxel_size=0.005,
            occlusion_mask=None, 
            visualize=False,
            visualize_all=False):
        """
        Updates the voxel map with a new observation.

        Args:
            intrinsic_matrix (np.ndarray): The intrinsic matrix of the camera
            extrinsic_matrix (np.ndarray): The extrinsic matrix of the camera
            width_px (int): The width of the camera image in pixels
            height_px (int): The height of the camera image in pixels
            occlusion_mask (np.ndarray): A 1xn array indicating occluded pixels, optional
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

        # Step 4: Get ray origins and directions (shape [H*W, 3])
        rays_flat = rays.reshape((-1, 6))
        ray_origins = rays_flat[:, 0:3]
        ray_dirs = rays_flat[:, 3:6]

        # Step 5: Compute hit points only where hits occurred
        hit_ray_origins = ray_origins[mask_hit]
        hit_ray_dirs = ray_dirs[mask_hit]
        hit_t = t_hit[mask_hit].reshape((-1, 1))
        hit_points = hit_ray_origins + hit_t * hit_ray_dirs

        # (Optional) convert to NumPy for further processing
        hit_points_np = hit_points.numpy()

        # (Optional) if occlusion_mask is used
        if occlusion_mask is not None:
            occlusion_indices = np.where(occlusion_mask == 1)[0]
        else:
            occlusion_indices = []

        # Step 6: Iterate over the hit points (minimized)
        points = []
        hit_points = []
        lines = []
        for idx, hit_point in enumerate(hit_points_np):
            dists = np.linalg.norm(self.points - hit_point[:, None], axis=0)
            within_radius_indices = np.where(dists < voxel_size)[0]
            visible_indices = np.setdiff1d(within_radius_indices, occlusion_indices)
            if len(visible_indices) == 0:
                continue

            ray_dir = hit_ray_dirs[idx].numpy()
            surface_normals = self.normals[:, visible_indices].T
            weights = -np.dot(surface_normals, ray_dir)
            self.confidences[visible_indices] += weights

            if visualize:
                points += [hit_ray_origins[idx].numpy(), (hit_ray_origins[idx] + hit_ray_dirs[idx] * hit_t[idx]).numpy()]
                lines.append([len(points) - 2, len(points) - 1])
                hit_points.append(hit_point)
            if visualize_all:
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(self.points.T)
                hit_point_cloud = o3d.geometry.PointCloud()
                hit_point_cloud.points = o3d.utility.Vector3dVector([hit_point])
                hit_point_cloud.paint_uniform_color([1, 0, 0])  # Red color for hit points
                point_cloud.paint_uniform_color([0.2, 0.2, 0.2])  # Gray color for all points
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(np.array([hit_ray_origins[idx].numpy(), (hit_ray_origins[idx] + hit_ray_dirs[idx] * hit_t[idx]).numpy()]))
                line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
                o3d.visualization.draw_geometries([line_set, point_cloud, hit_point_cloud])
        
        if visualize:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np.array(points))
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([0, 0, 1])
            hit_point_cloud = o3d.geometry.PointCloud()
            hit_point_cloud.points = o3d.utility.Vector3dVector(np.array(hit_points))
            hit_point_cloud.paint_uniform_color([1, 0, 0])  # Red color for hit points
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(self.points.T)
            point_cloud.paint_uniform_color([0.2, 0.2, 0.2])  # Gray color for all points
            o3d.visualization.draw_geometries([line_set, point_cloud, hit_point_cloud])
    
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