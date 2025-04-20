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

    def update_with_observation(self, intrinsic_matrix, extrinsic_matrix, width_px, height_px, occlusion_mask=None, visualize=False):
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

        if occlusion_mask is not None:
            occlusion_indices = np.where(occlusion_mask == 1)[0]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(self.points.T)

        if visualize:
            self.visualize(X_cam=extrinsic_matrix)
        
        points = []
        lines = []
        print(f"ray origin: {rays[0][0][:3].numpy()}")
        for idx in tqdm(range(rays.shape[0] * rays.shape[1]), desc="Casting rays"):
            i = idx // rays.shape[1]
            j = idx % rays.shape[1]
            ray = rays[i, j]
            ray_dir = ray[3:6]
            ray_origin = ray[:3]
            points += [ray_origin.numpy(), (ray_origin + ray_dir * 0.1).numpy()]
            lines.append([len(points) - 2, len(points) - 1])
            if raycast["t_hit"][i, j] < np.inf:
                print(f"Hit at pixel ({i}, {j})")
                depth = raycast["t_hit"][i, j]
                hit_point = ray_origin + depth * ray_dir
                nearest_voxel_index = np.argmin(np.linalg.norm(np.expand_dims(hit_point, axis=1) - self.points))
                if occlusion_mask is not None and nearest_voxel_index not in occlusion_indices:
                    # Update signed distance and confidence
                    surface_normal = self.normals[nearest_voxel_index]
                    weight = -np.dot(surface_normal, ray_dir)
                    self.confidences[nearest_voxel_index] += weight
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array(points))
        line_set.lines = o3d.utility.Vector2iVector(lines)
        o3d.visualization.draw_geometries([line_set, point_cloud])
    
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