# Mostly taken from https://github.com/nepfaff/rlg_panda_stack/blob/main/src/perception/grasp_node.py with ros stuff removed

import time
import open3d as o3d
import numpy as np
import threading
from typing import List
import planning.utils.utils as utils
import os
from planning.utils.geometry import to_rotation_matrices, se3_inverse_batch
from pydrake.all import (
    PointCloud,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
)
from manipulation.utils import ConfigureParser
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
from planning.misc.sdf_tools import SignedDensityField
import time
import torch

lock = threading.Lock()


class GraspListener():
    """The class responsible for computing and evaluation grasp candidates."""

    def __init__(self, hand_finger_path=None):
        if hand_finger_path == None:
            hand_finger_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'misc', 'hand_finger.sdf'))
        self.hand_collision_model = SignedDensityField.from_sdf(hand_finger_path)

        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0005)
        parser = Parser(self.plant)
        ConfigureParser(parser)
        parser.AddModelsFromUrl("package://manipulation/schunk_wsg_50_welded_fingers.sdf")
        self.plant.Finalize()

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()

        self.plant_context = self.plant.GetMyContextFromRoot(context)
        self.scene_graph_context = self.scene_graph.GetMyContextFromRoot(context)

    def check_collision(self, pcd, X_G, visualize=False):
        """Returns true if not in collision and false otherwise."""
        thre = 0.0
        sdf = self.compute_sdf(pcd, X_G, visualize)
        return sdf > thre

    def compute_darboux_frame(self, point, normal, pcd, kdtree, ball_radius=0.002, max_nn=50):
        """
        Given a index of the pointcloud, return a RigidTransform from world to the
        Darboux frame at that point.

        Y forward

        Args:
        - point (np.ndarray): point cloud point to compute the darboux frame for of shape (3,).
        - normal (np.ndarray): normla corresponding to `point` of shape (3,).
        - pcd (Drake PointCloud object): pointcloud of the object.
        - kdtree (scipy.spatial.KDTree object): kd tree to use for nn search.
        - ball_radius (float): ball_radius used for nearest-neighbors search.
        - max_nn (int): maximum number of points considered in nearest-neighbors search.
        """
        normals = pcd.normals()  # 3xN np array of normals

        distances, indices = kdtree.query(point, distance_upper_bound=ball_radius, k=max_nn)
        nn_indices = indices[np.isfinite(distances)]
        nn_normals = normals[:, nn_indices]

        M = nn_normals.dot(nn_normals.T)

        eigval, eigvec = np.linalg.eig(M)  # each column is eigenvector in decreasing eigenvalue order.

        order = eigval.argsort()

        nvec = eigvec[:, order[2]]
        tvec_major = eigvec[:, order[1]]
        tvec_minor = eigvec[:, order[0]]

        # If the z-direction is opposite from the normal vector, flip the x direction.
        if nvec.dot(normal) > 0:
            nvec = -nvec

        finger_tip_translation = np.array([0, 0, -0.06])
        # need the transform from finger tip to grasp
        R = np.vstack((tvec_major, tvec_minor, nvec))
        if np.linalg.det(R) < 0:
            tvec_major = -tvec_major
            R = np.vstack((tvec_major, tvec_minor, nvec))

        # make sure x axis facing upward
        if (R @ np.array([1, 0, 0]))[2] < 0:
            R = R @ utils.rotZ(np.pi)[:3, :3]

        R = RotationMatrix.ProjectToRotationMatrix(R)
        X_WF = RigidTransform(RotationMatrix(R), point + R @ finger_tip_translation)

        return X_WF

    def compute_darboux_frames(
        self, points, normals, pcd, kdtree, ball_radius=0.002, max_nn=50
    ) -> List[RigidTransform]:
        """
        TODO: This is not vectorized and hence slow. It would be better to provide a vectorized implementation.

        Args:
        - points (np.ndarray): the points to compute darboux frames for of shape (N,3).
        - normals (np.ndarray): the normals corresponding to `points` of shape (N,3).
        - pcd (Drake PointCloud object): pointcloud of the object.
        - kdtree (scipy.spatial.KDTree object): kd tree to use for nn search.
        - ball_radius (float): ball_radius used for nearest-neighbors search.
        - max_nn (int): maximum number of points considered in nearest-neighbors search.
        """
        frames = []
        for point, normal in zip(points, normals):
            frame = self.compute_darboux_frame(point, normal, pcd, kdtree, ball_radius, max_nn)
            frames.append(frame)
        return frames

    def compute_sdf(self, pcd, X_G, visualize=False):
        """Computes the signed distance from scratch via Drake query_object."""
        #print("start compute sdf")

        # not a free body set freebody pose will fail
        X_WGfix = RigidTransform(RotationMatrix(RollPitchYaw(np.pi/2, 0, np.pi/2)))
        self.plant.SetFreeBodyPose(self.plant_context, self.plant.GetBodyByName("body"), X_G.multiply(X_WGfix))
        query_object = self.scene_graph.get_query_output_port().Eval(self.scene_graph_context)
        pcd_sdf = np.inf

        for pt in pcd.xyzs().T:
            distances = query_object.ComputeSignedDistanceToPoint(pt)
            for body_index in range(len(distances)):
                distance = distances[body_index].distance
                if distance < pcd_sdf:
                    pcd_sdf = distance

        return pcd_sdf

    def compute_sdf_fast(self, pcd, X_G, visualize=False):
        """A lookup to the pre-computed sdf of the hand collision model."""
        pcd_W_np = pcd.xyzs()
        X_GW = X_G.inverse()
        pcd_G_np = X_GW.multiply(pcd_W_np)
        dist = self.hand_collision_model.get_distance(pcd_G_np.T)
        return dist.min()

    def check_collision_batch(self, pcd, X_G, visualize=False):
        """Returns true if not in collision and false otherwise."""
        thre = 0.0
        sdf = self.compute_sdf_batch(pcd, X_G, visualize)
        return sdf > thre

    def compute_sdf_batch(self, pcd_W_np: np.ndarray, X_Gs: np.ndarray, visualize=False):
        """A lookup to the pre-computed sdf of the hand collision model.
        parallelize over the transforms
        :param X_Gs: n x 4 x 4
        :param pcd_W_np: n x m x 3
        """
        pcd_G_np = X_Gs[:, :3, :3] @ pcd_W_np + X_Gs[:, :3, [3]]
        # get_distance only requires the last dimension to be 3: ... x 3 -> ... x 1
        dist = self.hand_collision_model.get_distance(pcd_G_np.transpose(0, 2, 1))
        return dist.reshape(len(dist), -1).min(axis=-1)

    def find_minimum_distance(self, pcd, X_WG):
        """
        By doing line search, compute the maximum allowable distance along the z axis before penetration.
        Return the maximum distance, as well as the new transform. Returns (np.nan, None) if nothing is returned after
        line search.

        NOTE: This does not consider the collision scene (e.g. table) but only the object point cloud.
        """
        num_z_samples = 10  # NOTE: The size of this affects the grasp computation time significantly
        z_grid = np.linspace(-0.05, 0.05, num_z_samples)
        signed_distance = -np.inf
        X_WGnew = RigidTransform()

        for z in z_grid:
            # Record the computed values using last z.
            last_signed_distance = signed_distance
            X_WGlast = X_WGnew

            # Compute new values.)
            X_WGnew = X_WG.multiply(RigidTransform([0.0, 0.0, z]))
            # print(z, X_WGnew)

            # manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
            # manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])
            # viz_geoms = [manipuland_cloud]
            # viz_geoms.append(self.make_gripper_line_set(X_WGnew.GetAsMatrix4(), [0.0, 1.0, 0.0]))

            signed_distance = self.compute_sdf(pcd, X_WGnew)

            # visualize 
            # o3d.visualization.draw_geometries(viz_geoms)

            # If the value crossed for the first time, return.
            thre = 0.0
            if (last_signed_distance > thre) and (signed_distance < thre):
                return last_signed_distance, X_WGlast

        # If nothing is returned after line search, discard the sample by sending None.
        return np.nan, None

    def find_minimum_distance_batch(self, pcds, X_WGs):
        """
        By doing line search, compute the maximum allowable distance along the z axis before penetration.
        Return the maximum distance, as well as the new transform. Returns (np.nan, None) if nothing is returned after
        line search.

        NOTE: This does not consider the collision scene (e.g. table) but only the object point cloud.
        """
        num_z_samples = 10  # NOTE: The size of this affects the grasp computation time significantly

        z_grid = np.linspace(-0.05, 0.05, num_z_samples)
        signed_distances = -np.inf * np.ones(len(X_WGs))

        # Mask is true for poses that finished updating
        finished_mask = np.zeros(len(X_WGs), dtype=bool)

        X_WGnew = np.tile(np.eye(4), (len(X_WGs), 1, 1))
        translation_transform = np.tile(np.eye(4), (len(X_WGs), 1, 1))
        for z in z_grid:
            # Record the computed values using last z.
            last_signed_distances = signed_distances
            X_WGlast = X_WGnew

            # Compute new values.
            translation_transform[:, 2, 3] = z
            # Only compute for not finished ones
            not_finished_mask = np.bitwise_not(finished_mask)
            X_WGnew[not_finished_mask] = X_WGs[not_finished_mask] @ translation_transform[not_finished_mask]
            signed_distances[not_finished_mask] = self.compute_sdf_batch(pcds, X_WGnew[not_finished_mask])

            # If the value crossed for the first time, finish updating
            thre = 0.0
            finished_mask = np.bitwise_or(
                finished_mask, np.bitwise_and(last_signed_distances > thre, signed_distances < thre)
            )

            # Check if all finished
            if np.all(finished_mask):
                break

        return last_signed_distances, X_WGlast

    def check_nonempty_batch(self, pcd_W_np, pcd_W_normals, X_WGs, visualize=False):
        """
        Check if the "closing region" of the gripper is nonempty by transforming the pointclouds to gripper coordinates.
        Assumes the panda gripper model.

        Args:
            - pcd (PointCloud object): pointcloud of the object shape.
            - X_WG (Drake RigidTransform): transform of the gripper of shape (M, 4, 4).

        Return:
            - is_nonempty (boolean): boolean set to True if there is a point within the cropped region. Shape (M,)
            - pcd_normals_G_np (np.array): pcd normals within the gripper closing region of shape (M, 3, N).
        """
        # Bounding box of the closing region written in the coordinate frame of the gripper body.
        # Do not modify
        crop_min = [-0.02, -0.053, 0.05]  # [-0.054, 0.036, -0.01]
        crop_max = [0.02, 0.053, 0.105]  # [0.054, 0.117, 0.01]

        # Transform the pointcloud to gripper frame.
        X_GWs = se3_inverse_batch(X_WGs)
        R = X_GWs[:, :3, :3]
        pcd_G_np = R @ (pcd_W_np) + X_GWs[:, :3, [3]]
        pcd_normals_G_np = R @ (pcd_W_normals)  # in the grasp frame

        # Check if there are any points within the cropped region.
        mask = (
            (crop_min[0] <= pcd_G_np[:, 0, :])
            * (pcd_G_np[:, 0, :] <= crop_max[0])
            * (crop_min[1] <= pcd_G_np[:, 1, :])
            * (pcd_G_np[:, 1, :] <= crop_max[1])
            * (crop_min[2] <= pcd_G_np[:, 2, :])
            * (pcd_G_np[:, 2, :] <= crop_max[2])
        )

        normal_masks = [np.nonzero(m)[0] for m in mask]
        is_nonempty = [m.any() for m in normal_masks]

        pcd_normals_G_np_selected = [
            normals[:, :, m].reshape(3, -1) for m in normal_masks for normals in [pcd_normals_G_np]
        ]

        return np.asarray(is_nonempty), pcd_normals_G_np_selected

    def check_nonempty(self, pcd, X_WG, visualize=False):
        """
        Check if the "closing region" of the gripper is nonempty by transforming the pointclouds to gripper coordinates.
        Assumes the panda gripper model.

        Args:
            - pcd (PointCloud object): pointcloud of the object.
            - X_WG (Drake RigidTransform): transform of the gripper.

        Return:
            - is_nonempty (boolean): boolean set to True if there is a point within the cropped region.
            - pcd_normals_G_np (np.array): pcd normals within the gripper closing region of shape (3, N).
        """
        pcd_W_np = pcd.xyzs()
        pcd_W_normals = pcd.normals()

        # Bounding box of the closing region written in the coordinate frame of the gripper body.
        # Do not modify
        crop_min = [-0.02, -0.053, 0.05]  # [-0.054, 0.036, -0.01]
        crop_max = [0.02, 0.053, 0.105]  # [0.054, 0.117, 0.01]

        # Transform the pointcloud to gripper frame.
        X_GW = X_WG.inverse()
        R = X_GW.GetAsMatrix4()[:3, :3]
        pcd_G_np = X_GW.multiply(pcd_W_np)
        pcd_normals_G_np = R @ (pcd_W_normals)  # in the grasp frame

        # Check if there are any points within the cropped region.
        mask = (
            (crop_min[0] <= pcd_G_np[0, :])
            * (pcd_G_np[0, :] <= crop_max[0])
            * (crop_min[1] <= pcd_G_np[1, :])
            * (pcd_G_np[1, :] <= crop_max[1])
            * (crop_min[2] <= pcd_G_np[2, :])
            * (pcd_G_np[2, :] <= crop_max[2])
        )
        indices = np.nonzero(mask)[0]

        is_nonempty = indices.any()

        # if visualize:
        #     meshcat.Delete()
        #     pcd_G = PointCloud(pcd)
        #     pcd_G.mutable_xyzs()[:] = pcd_G_np

        #     draw_grasp_candidate(RigidTransform())
        #     meshcat.SetObject("cloud", pcd_G)

        #     box_length = np.array(crop_max) - np.array(crop_min)
        #     box_center = (np.array(crop_max) + np.array(crop_min)) / 2.0
        #     meshcat.SetObject("closing_region", Box(box_length[0], box_length[1], box_length[2]), Rgba(1, 0, 0, 0.3))
        #     meshcat.SetTransform("closing_region", RigidTransform(box_center))

        return is_nonempty, pcd_normals_G_np[:, indices]

    def compute_costs(
            self, 
            X_WG: RigidTransform, 
            within_box_pt_normals: np.ndarray, 
            split_ratios: float, 
            major_split_axis: int,
            minor_split_axis: int,
            align_grasp_axis: List[float]=[0,0,1],
            align_secondary_axis: List[float]=[1,0,0],
            ) -> float:
        """
        Computes a grasp candidate cost based on a weighted sum of:
        - Antipodal (grasp normal) cost (prefer more antipodal)
        - Gripper axis alignment cost (prefer more aligned with given axes)
        - Vertical position cost (prefer higher grasps)

        :param X_WG: The grasp candidate to compute the cost for.
        :param within_box_pt_normals: Point cloud normals within the gripper closing region of shape (3, N).
        :param split_ratios: Array of [minor axis split ratio, major axis split ratio]. Values are in range [0,1] where
            higher indicates a more equal split along the principal object axis.
        """
        R = X_WG.GetAsMatrix4()[:3, :3]
        t = X_WG.GetAsMatrix4()[:3, 3]
        eff_vertical_vec = R.dot(np.array([0, 0, 1]))
        eff_horizontal_vec = R.dot(np.array([1, 0, 0]))

        antipodal_cost = -np.sum(
            within_box_pt_normals[1, :] ** 2
        )  # along the y axis of the gripper, larger good (antipodal metric)
        gripper_axis_alignment_cost = eff_vertical_vec @ align_grasp_axis  # want z axis of gripper to face towards desired axis, larger worse
        gripper_secondary_alignment_cost = np.abs(eff_horizontal_vec @ align_secondary_axis) # want x axis of gripper to face towards secondary axis (or 180 from)
        grasp_height_cost = -t[2]  # prefer higher position
        split_ratio_minor_axis_cost = -split_ratios[minor_split_axis]  # prefer higher split ratio
        split_ratio_major_axis_cost = -split_ratios[major_split_axis]
        cost = (
            antipodal_cost
            + 80.0 * gripper_axis_alignment_cost
            + 20.0 * gripper_secondary_alignment_cost
            # + 10.0 * grasp_height_cost
            + 1.0 * split_ratio_minor_axis_cost
            + 10.0 * split_ratio_major_axis_cost
        )
        return cost

    def compute_costs_batch(self, X_WGs: np.ndarray, within_box_pt_normals: np.ndarray, split_ratios: float, align_grasp_axis: List[float]=[0,0,1]) -> float:
        """
        Computes a grasp candidate cost based on a weighted sum of:
        - Antipodal (grasp normal) cost (prefer more antipodal)
        - Gripper vertical alignment cost (prefer more aligned with world z-axis)
        - Vertical position cost (prefer higher grasps)

        :param X_WG: The grasp candidate to compute the cost for.
        :param within_box_pt_normals: Point cloud normals within the gripper closing region of shape (3, N).
        :param split_ratios: Array of [minor axis split ratio, major axis split ratio]. Values are in range [0,1] where
            higher indicates a more equal split along the principal object axis.
        """
        R = X_WGs[:, :3, :3]
        t = X_WGs[:, :3, 3]

        eff_vertical_vec = R @ np.array([0, 0, 1])

        antipodal_cost = np.asarray([-np.sum(el[1, :] ** 2) for el in within_box_pt_normals])
        # along the y axis of the gripper, larger good (antipodal metric)
        gripper_vertical_alignment_cost = eff_vertical_vec @ align_grasp_axis  # want z axis of gripper to face down, larger worse
        grasp_height_cost = -t[:, 2]  # prefer higher position
        split_ratio_minor_axis_cost = -split_ratios[0]  # prefer higher split ratio
        split_ratio_major_axis_cost = -split_ratios[2]
        cost = (
            antipodal_cost
            + 100.0 * gripper_vertical_alignment_cost
            + 10.0 * grasp_height_cost
            + 1.0 * split_ratio_minor_axis_cost
            + 10.0 * split_ratio_major_axis_cost
        )
        return cost

    @staticmethod
    def compute_pcd_split_ratio_xy(pcd_points: np.ndarray) -> np.ndarray:
        """
        Computes the x and y split ratios for each point.
        The split ratio has range [0,1] where 1 is best (most equal split) and 0 is worst (most unequal split).

        :param pcd_points: Point cloud points of shape (N,3).
        :return: X and y split ratios for each point of shape (N,2).
        """
        # Min/max bounds for split ratio
        min_point_vals = np.min(pcd_points, axis=0)
        max_point_vals = np.max(pcd_points, axis=0)

        # Compute split ratio
        split = np.array([max_point_vals - pcd_points, pcd_points - min_point_vals])
        split_ratio = np.min(split, axis=0) / np.max(split, axis=0)
        return split_ratio[:, :2]

    @staticmethod
    def compute_pcd_split_ratio_major_axis(pcd_points: np.ndarray, viz_split_ratio_axes: bool = False) -> np.ndarray:
        """
        Computes the split ratio of the major point cloud axis.
        The split ratio has range [0,1] where 1 is best (most equal split) and 0 is worst (most unequal split).

        :param pcd_points: Point cloud points of shape (N,3).
        :param viz_major_axis: Whether to visualize the pcd with the principle and minor axes in open3d.
        :return: Minor and major point cloud split ratio for each point of shape (N,2) where the first entry is the
            minor axis.
        """
        cov = np.cov(pcd_points.T)
        eigval, eigvec = np.linalg.eig(cov)

        order = eigval.argsort()
        principal_component = eigvec[:, order[-1]]
        minor_component = eigvec[:, order[0]]

        # Rotate point cloud to align the principal component with the z-axis and the minor component with the x-axis
        z_axis, x_axis = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
        rot_principal_component_to_axes, _ = R.align_vectors(
            np.array([z_axis, x_axis]), np.stack([principal_component, minor_component])
        )
        pcd_points_axis_aligned = pcd_points @ rot_principal_component_to_axes.as_matrix().T

        if viz_split_ratio_axes:
            # Visualize the point cloud in blue, the principal axis in red, and the minor axis in green
            pcd_points_axis_aligned_normalized = pcd_points_axis_aligned - np.mean(pcd_points_axis_aligned, axis=0)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_points_axis_aligned_normalized))
            pcd.paint_uniform_color([0.0, 0.0, 1.0])
            principle_component_line = o3d.geometry.LineSet()
            principle_component_line.points = o3d.utility.Vector3dVector(
                np.array([[0.0, 0.0, -0.3], [0.0, 0.0, 0.3], [-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
            )
            principle_component_line.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 3]]))
            principle_component_line.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
            o3d.visualization.draw_geometries([pcd, principle_component_line, world_frame])

        # Min/max bounds for split ratio
        min_point_vals = np.min(pcd_points_axis_aligned, axis=0)
        max_point_vals = np.max(pcd_points_axis_aligned, axis=0)

        # Compute split ratio
        split = np.array([max_point_vals - pcd_points_axis_aligned, pcd_points_axis_aligned - min_point_vals])
        split_ratio = np.min(split, axis=0) / np.max(split, axis=0)

        return split_ratio[:, [0, 2]]

    @staticmethod
    def compute_pcd_split_ratio(pcd_points: np.ndarray, viz_split_ratio_axes: bool = False) -> np.ndarray:
        """
        Computes the split ratio of the major point cloud axis.
        The split ratio has range [0,1] where 1 is best (most equal split) and 0 is worst (most unequal split).

        :param pcd_points: Point cloud points of shape (N,3).
        :param viz_major_axis: Whether to visualize the pcd with the principle and minor axes in open3d.
        :return: Minor, secondary, and major point cloud split ratio for each point of shape (N,3) where the first entry is the
            minor axis, second is the secondary, and last is the major axis.
        """
        cov = np.cov(pcd_points.T)
        eigval, eigvec = np.linalg.eig(cov)

        order = eigval.argsort()
        principal_component = eigvec[:, order[-1]]
        secondary_component = eigvec[:, order[1]]
        minor_component = eigvec[:, order[0]]

        # Rotate point cloud to align the principal component with the z-axis and the minor component with the x-axis
        z_axis, x_axis = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
        rot_principal_component_to_axes, _ = R.align_vectors(
            np.array([z_axis, x_axis]), np.stack([principal_component, minor_component])
        )

        pcd_points_axis_aligned = pcd_points @ rot_principal_component_to_axes.as_matrix().T

        if viz_split_ratio_axes:
            # Visualize the point cloud in blue, the principal axis in red, and the minor axis in green
            pcd_points_axis_aligned_normalized = pcd_points_axis_aligned - np.mean(pcd_points_axis_aligned, axis=0)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_points_axis_aligned_normalized))
            pcd.paint_uniform_color([0.0, 0.0, 1.0])
            principle_component_line = o3d.geometry.LineSet()
            principle_component_line.points = o3d.utility.Vector3dVector(
                np.array([[0.0, 0.0, -0.3], [0.0, 0.0, 0.3], [-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
            )
            principle_component_line.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 3]]))
            principle_component_line.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
            o3d.visualization.draw_geometries([pcd, principle_component_line, world_frame])

        # Min/max bounds for split ratio
        min_point_vals = np.min(pcd_points_axis_aligned, axis=0)
        max_point_vals = np.max(pcd_points_axis_aligned, axis=0)

        # Compute split ratio
        split = np.array([max_point_vals - pcd_points_axis_aligned, pcd_points_axis_aligned - min_point_vals])
        split_ratio = np.min(split, axis=0) / np.max(split, axis=0)

        return split_ratio[:, [0, 1, 2]]


    @staticmethod
    def compute_pcd_point_locations(pcd_points: np.ndarray, viz_split_ratio_axes: bool = False) -> np.ndarray:
        """
        Computes the relative location of each point in pcd. Using (minor, secondary, major) axes as (x, y, z)
        The relative location has range [0,1] where 10 is the farthest point in the negative direction of the axis
        and 1 is the farthest point in the positive direction of the axis

        :param pcd_points: Point cloud points of shape (N,3).
        :param viz_major_axis: Whether to visualize the pcd with the principle and minor axes in open3d.
        :return: Minor, secondary, and major point cloud locations for each point of shape (N,3) where the first entry is the
            minor axis, second is the secondary, and last is the major axis.
        """
        cov = np.cov(pcd_points.T)
        eigval, eigvec = np.linalg.eig(cov)

        order = eigval.argsort()
        principal_component = eigvec[:, order[-1]]
        secondary_component = eigvec[:, order[1]]
        minor_component = eigvec[:, order[0]]

        # Rotate point cloud to align the principal component with the z-axis and the minor component with the x-axis
        z_axis, x_axis = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
        rot_principal_component_to_axes, _ = R.align_vectors(
            np.array([z_axis, x_axis]), np.stack([principal_component, minor_component])
        )

        pcd_points_axis_aligned = pcd_points @ rot_principal_component_to_axes.as_matrix().T

        if viz_split_ratio_axes:
            # Visualize the point cloud in blue, the principal axis in red, and the minor axis in green
            pcd_points_axis_aligned_normalized = pcd_points_axis_aligned - np.mean(pcd_points_axis_aligned, axis=0)
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_points_axis_aligned_normalized))
            pcd.paint_uniform_color([0.0, 0.0, 1.0])
            principle_component_line = o3d.geometry.LineSet()
            principle_component_line.points = o3d.utility.Vector3dVector(
                np.array([[0.0, 0.0, -0.3], [0.0, 0.0, 0.3], [-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
            )
            principle_component_line.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 3]]))
            principle_component_line.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
            o3d.visualization.draw_geometries([pcd, principle_component_line, world_frame])

        # Min/max bounds for location
        min_point_vals = np.min(pcd_points_axis_aligned, axis=0)
        max_point_vals = np.max(pcd_points_axis_aligned, axis=0)
        point_vals_range = max_point_vals - min_point_vals

        # Compute location
        location = (max_point_vals - pcd_points_axis_aligned) / point_vals_range

        return location

    @staticmethod
    def make_gripper_line_set(pose: np.ndarray, color=(1, 0, 0)):
        """
        Returns an Open3D LineSet for the panda gripper.
        :param pose: The homogenous gripper pose of shape (4,4).
        """
        hand_anchor_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, -0.00, 0.058],
                [0.0, -0.043, 0.058],
                [0.0, 0.043, 0.058],
                [0.0, -0.043, 0.098],
                [0.0, 0.043, 0.098],
            ]
        )
        line_index = [[0, 1], [1, 2], [1, 3], [3, 5], [2, 4]]

        hand_anchor_points[1] = (hand_anchor_points[2] + hand_anchor_points[3]) / 2.0

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(hand_anchor_points)
        line_set.lines = o3d.utility.Vector2iVector(line_index)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(line_index))])
        line_set.transform(pose)
        return line_set
    
    @staticmethod
    def make_triad_line_set(pose: np.ndarray, color=(1, 0, 0)):
        """
        Returns an Open3D LineSet for a triad.
        :param pose: The homogenous pose of shape (4,4).
        """
        hand_anchor_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        line_index = [[0, 1], [0, 2], [0, 3]]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(hand_anchor_points)
        line_set.lines = o3d.utility.Vector2iVector(line_index)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(line_index))])
        line_set.transform(pose)
        return line_set

    def compute_candidate_grasps(
        self, pcd: PointCloud, candidate_num=30, num_samples=20, random_seed=5, align_grasp_axis = [0, 0, 1], align_secondary_axis = [1, 0, 0], split_axis=2, minor_split_axis=0
    ):
        """
        Compute sorted candidate grasps.
        Based on "Grasp Pose Detection in Point Clouds": https://arxiv.org/abs/1706.09911.

        Args:
            - pcd (Drake PointCloud object): downsampled pointcloud of the object.
            - candidate_num (int): number of desired candidates. The samples with the lowest cost are returned.
            - num_samples (int): number of point cloud points to sample for grasp candidates.
            - random_seed (int): seed for rng, used for grading.
            - align_grasp_axis: axis for which we want to align grasps to
            - split axis: main axis we care about for split ratio where 2 is the principal axis, 1 is the secondary, and 0 is the minor
            - minor_split_axis: secondary axis we care about for split ratio

        Return:
            - candidate_lst (list of drake RigidTransforms): candidate list of
              grasps, sorted based on cost.
        """

        split_ratio_major_axis_threshold = 0.6  # Axis of biggest pcd variation
        split_ratio_minor_axis_threshold = 0.6  # Axis of smallest pcd variation

        # NOTE: All num_samples should be odd numbers
        y_min = -0.01
        y_max = 0.01
        num_y_samples = 3
        roll_min = -np.pi / 8
        roll_max = np.pi / 8
        num_roll_samples = 10
        # TODO: Look into exploiting Panda gripper symmetry (grasps rotated by n*pi should be equivalent)
        yaw_min = -np.pi / 2
        yaw_max = np.pi / 2
        num_yaw_samples = 11

        np.random.seed(random_seed)

        pcd_points = pcd.xyzs().T

        # Filter pcd based on split ratio
        split_ratios = self.compute_pcd_split_ratio(pcd_points, viz_split_ratio_axes=False)
        print(split_ratios[:, minor_split_axis])
        print(split_ratios[:, split_axis])
        mask = (split_ratios[:, minor_split_axis] > split_ratio_minor_axis_threshold) * (
            split_ratios[:, split_axis] > split_ratio_major_axis_threshold
        )
        split_ratio_filtered_points = pcd_points[mask]
        split_ratio_filtered_normals = pcd.normals()[:, mask].T

        # Kdtree based on unfiltered points for better normal queries
        kdtree = KDTree(pcd_points)

        # Sample random points to compute darboux frames for
        num_split_ratio_filtered_points = len(split_ratio_filtered_points)
        print("num_split_ratio_filtered_points", num_split_ratio_filtered_points)
        darboux_frame_sample_indices = np.random.choice(
            np.arange(num_split_ratio_filtered_points),
            int(min(num_samples, num_split_ratio_filtered_points)),
            replace=False,
        )

        print("len sample indicies", darboux_frame_sample_indices)

        # Compute darboux frames at samples
        X_WPs = self.compute_darboux_frames(
            split_ratio_filtered_points[darboux_frame_sample_indices],
            split_ratio_filtered_normals[darboux_frame_sample_indices],
            pcd,
            kdtree,
        )

        print("Num X_WPs", len(X_WPs))

        manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
        manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])
        viz_geoms = [manipuland_cloud]

        PARALLEL = False

        start_time = time.time()
        if not PARALLEL:

            # Local grid search around darboux frames
            candidate_lst: List[RigidTransform] = []
            candidate_costs: List[float] = []
            for X_WP, split_ratio in zip(X_WPs, split_ratios[darboux_frame_sample_indices]):

                # NOTE: The best variations to sample/ search over is situation/ grasp environment dependent (e.g. bin vs table)
                for y in np.linspace(y_min, y_max, num_y_samples):
                    for roll in np.linspace(roll_min, roll_max, num_roll_samples):
                        for yaw in np.linspace(yaw_min, yaw_max, num_yaw_samples):
                            # TODO: Explore whether it is faster to do this transform in numpy
                            X_PPnew = RigidTransform(RollPitchYaw(roll, 0.0, yaw), np.array([0, y, 0]))
                            X_WPnew = X_WP.multiply(X_PPnew)

                            # visualize
                            # manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
                            # manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])
                            # o3d.visualization.draw_geometries([manipuland_cloud, self.make_triad_line_set(X_WP.GetAsMatrix4(), [0.0, 1.0, 0.0])])
                            # print("darboux frame", X_WP)

                            # Compute a new transform that minimizes y-direction distance without penetration
                            distance, X_WPnew = self.find_minimum_distance(pcd, X_WPnew)
                            # If distance cannot be found, go over to the next iteration
                            if np.isnan(distance):
                                continue

                            # If the candidate has no collisions and the closing region is non
                            # empty, then append it to the list of candidates.
                            is_nonempty, within_box_pt_normals = self.check_nonempty(pcd, X_WPnew)
                            if is_nonempty:
                                candidate_lst.append(X_WPnew)
                                viz_geoms.append(self.make_gripper_line_set(X_WPnew.GetAsMatrix4(), [0.0, 1.0, 0.0]))
                                candidate_costs.append(
                                    self.compute_costs(
                                        X_WPnew, 
                                        within_box_pt_normals, 
                                        split_ratio, 
                                        split_axis, 
                                        minor_split_axis, 
                                        align_grasp_axis, 
                                        align_secondary_axis
                                    )
                                )
                            else:
                                continue
            o3d.visualization.draw_geometries(viz_geoms)
            print("sequential antipodal grasp time: {:.3f}".format(time.time() - start_time))

        else:

            y_split = np.linspace(y_min, y_max, num_y_samples)

            roll_split = np.linspace(roll_min, roll_max, num_roll_samples)
            pitch_split = np.zeros_like(roll_split)
            yaw_split = np.linspace(yaw_min, yaw_max, num_yaw_samples)
            # List of all possible roll, pitch, yaw combinations
            rot_list = np.stack(np.meshgrid(roll_split, pitch_split, yaw_split), axis=-1).reshape(-1, 3)
            batch_rot = to_rotation_matrices(torch.from_numpy(rot_list)).detach().numpy()

            pcd_W_np = pcd.xyzs()[np.newaxis, :]
            pcd_W_normals = pcd.normals()[np.newaxis, :]

            candidate_lst = []
            candidate_costs = []
            for X_WP, split_ratio in zip(X_WPs, split_ratios[darboux_frame_sample_indices]):
                batch_delta_transform = np.zeros((len(batch_rot), 4, 4))
                batch_delta_transform[:, :3, :3] = batch_rot
                batch_delta_transform[:, :3, 1] = y_split
                X_WP_batch = X_WP.GetAsMatrix4()[np.newaxis, :] @ batch_delta_transform

                distances, X_WPnew_batch = self.find_minimum_distance_batch(pcd_W_np, X_WP_batch)
                no_penetration_mask = distances > 0.0
                X_WPnew_batch_no_penetration = X_WPnew_batch[no_penetration_mask]
                if len(X_WPnew_batch_no_penetration) == 0:
                    continue

                cage_mask, within_box_pt_normals = self.check_nonempty_batch(
                    pcd_W_np, pcd_W_normals, X_WPnew_batch_no_penetration
                )
                X_WPnew_batch_nonempty = X_WPnew_batch_no_penetration[cage_mask]
                if len(X_WPnew_batch_nonempty) == 0:
                    continue

                grasp_costs = self.compute_costs_batch(X_WPnew_batch_no_penetration, within_box_pt_normals, split_ratio, align_grasp_axis)

                # Pick valid grasps and associated costs
                candidate_lst.append(X_WPnew_batch_nonempty)
                candidate_costs.append(grasp_costs[cage_mask])

            candidate_lst = np.concatenate(candidate_lst, axis=0)
            candidate_costs = np.concatenate(candidate_costs, axis=0)

            print("parallel antipodal grasp time: {:.3f}".format(time.time() - start_time))

        sorted_indices = np.argsort(candidate_costs)
        candidate_lst_sorted = [candidate_lst[idx] for idx in sorted_indices]

        self.grasp_candidates: List[np.ndarray] = candidate_lst_sorted[:candidate_num]

    def get_best_grasps(self, candidate_num=-1) -> List[np.ndarray]:
        """Returns a list of the `candidate_num` grasps with the lowest cost."""
        return self.grasp_candidates[:candidate_num]