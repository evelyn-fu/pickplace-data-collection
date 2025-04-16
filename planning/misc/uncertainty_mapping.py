import numpy as np
import open3d as o3d
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

        self.triangle_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(self.points.T)
            )
        )

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(self.triangle_mesh)

        print(f"Initialized voxel map with {len(self.voxel_map)} voxels.")

    def update_with_observation(self, intrinsic_matrix, extrinsic_matrix, width_px, height_px, occlusion_mask=None):
        """
        Updates the voxel map with a new observation.

        Args:
            intrinsic_matrix (np.ndarray): The intrinsic matrix of the camera
            extrinsic_matrix (np.ndarray): The extrinsic matrix of the camera in object frame
            width_px (int): The width of the camera image in pixels
            height_px (int): The height of the camera image in pixels
            occlusion_mask (np.ndarray): A 1xn array indicating occluded pixels, optional
        """

        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            width_px=width_px,
            height_px=height_px,
        )

        raycast = self.scene.cast_rays(rays)

        if occlusion_mask is not None:
            occlusion_indices = np.where(occlusion_mask == 1)[0]

        for i in range(rays.shape[0]):
            ray = rays[i]
            ray_dir = ray[3:6]
            ray_origin = ray[:3]
            if raycast["t_hit"][i] < np.inf:
                depth = raycast["t_hit"][i]
                hit_point = ray_origin + depth * ray_dir
                nearest_voxel_index = np.argmin(np.linalg.norm(hit_point - self.points))
                if nearest_voxel_index not in occlusion_indices:
                    # Update signed distance and confidence
                    surface_normal = self.normals[nearest_voxel_index]
                    weight = -np.dot(surface_normal, ray_dir)
                    self.confidences[nearest_voxel_index] += weight
    
    def visualize(self):
        """
        Visualizes the voxel map using open3d
        """
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(self.points.T)

        base_color = np.array([0.2, 0.2, 0.2])
        target_color = np.array([1, 0, 0])
        colors = base_color + np.clip(self.confidences, 0, 1) * (target_color - base_color) # red more confident, gray less confident
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([point_cloud])

    def fully_obserrve(self):
        """
        Returns True if the map is fully observed
        """
        return np.all(self.confidences > self.confidence_threshold)