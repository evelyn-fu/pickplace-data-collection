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
    def __init__(self, points: np.ndarray, voxel_size: float):
        """
        Initializes a voxel map with signed distance values and confidence.

        Args:
            points (np.ndarray): A 3xn array of points in 3D space.
            voxel_size (float): The size of each voxel in the map.
        """
        if points.shape[0] != 3:
            raise ValueError("Input points must be a 3xn array.")

        self.voxel_size = voxel_size
        self.voxel_map = {}

        # Compute the center of the points
        self.center = np.mean(points, axis=1)
        print(f"Voxel map center: {self.center}")

        # Convert points to voxel indices
        voxel_indices = np.floor(points / voxel_size).astype(int)

        # Initialize the voxel map with signed distance and confidence
        for idx in range(voxel_indices.shape[1]):
            voxel_key = tuple(voxel_indices[:, idx])
            if voxel_key not in self.voxel_map:
                self.voxel_map[voxel_key] = {"signed_distance": 0.0, "confidence": 0.0}

        self.pcd = PointCloud(points.shape[1])
        self.pcd.mutable_xyzs()[:] = points
        self.pcd = self.pcd.VoxelizedDownSample(voxel_size=voxel_size)
        self.pcd.EstimateNormals(radius=0.1, num_closest=30)
        self.pcd.FlipNormalsTowardPoint(self.center)
        self.pcd.mutable_normals()[:] = -self.pcd.normals() # flip away from center

        self.triangle_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(points.T)
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
            occlusion_mask (np.ndarray): A 3xn array indicating occluded pixels, optional
        """

        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix,
            width_px=width_px,
            height_px=height_px,
        )

        raycast = self.scene.cast_rays(rays)

        if occlusion_mask is not None:
            occlusion_set = np.floor(occlusion_mask / self.voxel_size).astype(int)

        for i in range(rays.shape[0]):
            ray = rays[i]
            ray_dir = ray[3:6]
            ray_origin = ray[:3]
            if raycast["t_hit"][i] < np.inf:
                depth = raycast["t_hit"][i]
                hit_point = ray_origin + depth * ray_dir
                voxel_index = np.floor(hit_point / self.voxel_size).astype(int)
                voxel_key = tuple(voxel_index)
                if voxel_key in self.voxel_map and voxel_key:
                    # Update signed distance and confidence
                    pass
