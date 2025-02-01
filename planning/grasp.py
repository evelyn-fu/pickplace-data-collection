# Mostly taken from https://github.com/nepfaff/rlg_panda_stack/blob/main/src/perception/grasp_node.py with a lot of changes

import time
import open3d as o3d
import numpy as np
import threading
from typing import List, Tuple
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
import multiprocessing
from multiprocessing.pool import ThreadPool as Pool
import torch

lock = threading.Lock()

PROCESSES = multiprocessing.cpu_count()

class GraspListener():
    """The class responsible for computing and evaluation grasp candidates."""

    def __init__(self, hand_finger_path=None, gripper_model_path=None):
        if hand_finger_path == None:
            hand_finger_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'scenario_datas', 'gripper_sdf.pkl'))
        self.hand_collision_model = SignedDensityField.from_pkl(hand_finger_path)
        # self.hand_collision_model.visualize()

        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0005)
        parser = Parser(self.plant)
        ConfigureParser(parser)
        if gripper_model_path == None:
            gripper_model_path = "package://manipulation/schunk_wsg_50_welded_fingers.sdf"
        parser.AddModelsFromUrl(gripper_model_path)
        self.plant.Finalize()

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()

        self.plant_context = self.plant.GetMyContextFromRoot(context)
        self.scene_graph_context = self.scene_graph.GetMyContextFromRoot(context)

    def check_collision(self, pcd, X_G, visualize=False):
        """Returns true if not in collision and false otherwise."""
        thre = 0.0
        sdf = self.compute_sdf_fast(pcd, X_G, visualize)
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

        finger_tip_translation = np.array([0, 0, 0])
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
        X_GW = (X_G @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0])).inverse()
        pcd_G_np = X_GW.multiply(pcd_W_np)
        dist = self.hand_collision_model.get_distance(pcd_G_np.T)

        if visualize:
            manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_G_np.T))
            manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

            gripper_xyzs = self.hand_collision_model.to_pcd()
            gripper_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(0.005)
            gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])

            viz_geoms = [manipuland_cloud, gripper_cloud]
            o3d.visualization.draw_plotly(viz_geoms)

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

    def find_minimum_distance(self, pcd, X_WG, thre=0.0):
        """
        By doing line search, compute the maximum allowable distance along the z axis before penetration.
        Return the maximum distance, as well as the new transform. Returns (np.nan, None) if nothing is returned after
        line search.

        NOTE: This does not consider the collision scene (e.g. table) but only the object point cloud.
        """
        num_z_samples = 10  # NOTE: The size of this affects the grasp computation time significantly
        z_grid = np.linspace(-0.11, -0.01, num_z_samples)
        signed_distance = -np.inf
        X_WGnew = RigidTransform()

        X_WGlast = None
        last_signed_distance = np.nan
        for z in z_grid:
            # Compute new values.
            X_WGnew = X_WG.multiply(RigidTransform([0.0, 0.0, z]))
            # print(z, X_WGnew)

            # manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
            # manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])
            # viz_geoms = [manipuland_cloud]
            # viz_geoms.append(self.make_gripper_line_set(X_WGnew.GetAsMatrix4(), [0.0, 1.0, 0.0]))

            signed_distance = self.compute_sdf_fast(pcd, X_WGnew)

            # visualize 
            # o3d.visualization.draw_geometries(viz_geoms)

            # If the value crossed for the first time, return.
            if signed_distance < thre:
                if X_WGlast is None:
                    # input("no bueno, always crosses")
                    return last_signed_distance, X_WGlast

                # manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
                # manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

                # gripper_xyzs = self.hand_collision_model.to_pcd()
                # gripper_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(0.005).transform((X_WGlast @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0])).GetAsMatrix4())
                # gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])
                
                # gripper_next_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(0.005).transform((X_WGnew @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0])).GetAsMatrix4())
                # gripper_next_cloud.paint_uniform_color([0.0, 1.0, 0.0])

                # viz_geoms = [manipuland_cloud, gripper_cloud, gripper_next_cloud]
                # o3d.visualization.draw_plotly(viz_geoms)
                # input(f"Found grasp {last_signed_distance}, {signed_distance}, {X_WGlast}")
                return last_signed_distance, X_WGlast
            
            # Record the computed values using last z.
            last_signed_distance = signed_distance
            X_WGlast = X_WGnew

        # If nothing is returned after line search, discard the sample by sending None.
        # manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
        # manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])
        # viz_geoms = [manipuland_cloud]
        # viz_geoms.append(self.make_gripper_line_set(X_WG.GetAsMatrix4(), [0.0, 1.0, 0.0]))
        # o3d.visualization.draw_geometries(viz_geoms)
        # input("no bueno, never crosses")
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

        Args:
            - pcd (PointCloud object): pointcloud of the object.
            - X_WG (Drake RigidTransform): transform of the gripper.

        Return:
            - is_nonempty (boolean): boolean set to True if there is a point within the cropped region.
            - pcd_normals_G_np (np.array): pcd normals within the gripper closing region of shape (3, N).
            - proportion_enclosed: proportion of normals enclosed out of all possible points in PCD
        """
        pcd_W_np = pcd.xyzs()
        pcd_W_normals = pcd.normals()

        # Bounding box of the closing region written in the coordinate frame of the gripper body.
        # Do not modify
        crop_min = [-0.01, -0.053, 0.03]  # [-0.054, 0.036, -0.01]
        crop_max = [0.01, 0.053, 0.15]  # [0.054, 0.117, 0.01]

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

        if visualize:
            pcd_closing_region = pcd.xyzs().T[indices, :]
            pcd_closing_region_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_closing_region))
            pcd_closing_region_cloud.paint_uniform_color([1.0, 0.0, 0.0])
            manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
            manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])
            viz_geoms = [manipuland_cloud, pcd_closing_region_cloud, self.make_gripper_line_set(X_WG.GetAsMatrix4(), [0.0, 1.0, 0.0])]
            o3d.visualization.draw_geometries(viz_geoms)
        
        proportion_enclosed = len(indices) / (pcd_normals_G_np.shape[1])

        return is_nonempty, pcd_normals_G_np[:, indices], proportion_enclosed

    def compute_costs(
            self, 
            X_WG: RigidTransform, 
            within_box_pt_normals: np.ndarray,
            proportion_enclosed: float,
            split_ratios: np.ndarray,
            split_axes = np.ndarray,
            ) -> float:
        """
        Computes a grasp candidate cost based on a weighted sum of:
        - Antipodal (grasp normal) cost (prefer more antipodal)
        - Gripper axis alignment cost (prefer more aligned with given axes)
        - Split ratio cost (prefer to grasp near middles of object)

        :param X_WG: The grasp candidate to compute the cost for.
        :param within_box_pt_normals: Point cloud normals within the gripper closing region of shape (3, N).
        :param split_ratios: Array of [minor axis split ratio, major axis split ratio]. Values are in range [0,1] where
            higher indicates a more equal split along the principal object axis.
        """
        R = X_WG.GetAsMatrix4()[:3, :3]
        t = X_WG.GetAsMatrix4()[:3, 3]
        eff_vertical_vec = R.dot(np.array([0, 0, 1])) # vertical axis of gripper (parallel to fingers)
        eff_horizontal_vec = R.dot(np.array([0, 1, 0])) # horizontal axis of gripper (line connecting finger tips)
        
        antipodal_cost = -np.sum(
            within_box_pt_normals[1, :] ** 2
        ) / within_box_pt_normals.shape[1]  # along the horizontal axis of the gripper, larger good (antipodal metric)

        # want grasps to avoid alignment with long axes, smaller better
        gripper_vertical_axis_alignment_cost = np.abs(eff_vertical_vec @ split_axes)
        gripper_horizontal_axis_alignment_cost = -np.max(np.abs(eff_horizontal_vec @ split_axes))

        # consider lower two split ratio costs (grasp should be even along at least two axes)
        split_ratio_costs = -split_ratios
        split_ratio_costs_sorted = np.sort(split_ratio_costs)
        split_ratio_cost = split_ratio_costs_sorted[0] + split_ratio_costs_sorted[1]
        higher_up_cost = -min(t[2] - 0.15, 0) / 0.15

        cost = (
            60 * antipodal_cost
            + 10.0 * gripper_vertical_axis_alignment_cost[2]
            + 5.0 * gripper_vertical_axis_alignment_cost[1]
            + 20.0 * split_ratio_cost
            + 10.0 * higher_up_cost
            - 60.0 * proportion_enclosed
        )
        return cost
    def compute_costs_single(
            self, 
            X_WG: RigidTransform, 
            within_box_pt_normals: np.ndarray, 
            fraction_enclosed: float,
            split_ratios: np.ndarray, 
            major_split_axis: int,
            minor_split_axis: int,
            align_grasp_axis: List[float]=[0,0,1],
            align_minor_axis: List[float]=[1,0,0],
            ) -> float:
        R = X_WG.GetAsMatrix4()[:3, :3]
        t = X_WG.GetAsMatrix4()[:3, 3]
        eff_vertical_vec = R.dot(np.array([0, 0, 1])) # vertical axis of gripper (parallel to fingers)
        eff_horizontal_vec = R.dot(np.array([0, 1, 0])) # horizontal axis of gripper (line connecting finger tips)
        
        antipodal_cost = -np.sum(
            within_box_pt_normals[1, :] ** 2
        ) / within_box_pt_normals.shape[1]  # along the horizontal axis of the gripper, larger good (antipodal metric)

        gripper_axis_alignment_cost = -np.abs(eff_vertical_vec @ align_grasp_axis)  # want vertical axis of gripper to face towards desired axis
        gripper_minor_alignment_cost = -np.abs(eff_horizontal_vec @ align_minor_axis) # want horizontal axis of gripper to align with minor axis 
        split_ratio_minor_axis_cost = -split_ratios[minor_split_axis]  # prefer higher split ratio
        split_ratio_major_axis_cost = -split_ratios[major_split_axis]
        cost = (
            10.0 * antipodal_cost
            + 10.0 * gripper_axis_alignment_cost
            + 5.0 * gripper_minor_alignment_cost
            + 5.0 * split_ratio_minor_axis_cost
            + 5.0 * split_ratio_major_axis_cost
            + 5.0 * fraction_enclosed
        )
        return cost

    def compute_costs_batch(
            self, 
            X_WGs: RigidTransform, 
            within_box_pt_normals: np.ndarray,
            split_ratios: np.ndarray,
            split_axes = np.ndarray,
            ) -> float:
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

        eff_vertical_vec = R @ np.array([0, 0, 1]) # vertical axis of gripper (parallel to fingers)
        eff_horizontal_vec = R @ np.array([0, 1, 0]) # horizontal axis of gripper (line connecting finger tips)

        # along the y axis of the gripper, larger good (antipodal metric)
        antipodal_cost = np.asarray([-np.sum(el[1, :] ** 2) for el in within_box_pt_normals])
        # want grasps to avoid alignment with long axes, smaller better
        gripper_vertical_axis_alignment_cost = -np.max(np.abs(eff_vertical_vec @ split_axes)) 
        gripper_horizontal_axis_alignment_cost = -np.max(np.abs(eff_horizontal_vec @ split_axes))

        # consider lower two split ratio costs (grasp should be even along at least two axes)
        split_ratio_costs = -split_ratios
        split_ratio_costs_sorted = np.sort(split_ratio_costs)
        split_ratio_cost = split_ratio_costs_sorted[0] + split_ratio_costs_sorted[1]
        cost = (
            10.0 * antipodal_cost
            + 10.0 * gripper_vertical_axis_alignment_cost
            + 5.0 * gripper_horizontal_axis_alignment_cost
            + 5.0 * split_ratio_cost
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

        # upper bound on distance between points
        length = np.linalg.norm(max_point_vals - min_point_vals)

        # Compute split ratio
        split = np.array([max_point_vals - pcd_points_axis_aligned, pcd_points_axis_aligned - min_point_vals])
        split_ratio = np.min(split, axis=0) / np.max(split, axis=0)

        # return split axes
        axes = np.stack([principal_component, secondary_component, minor_component])

        return split_ratio[:, [0, 1, 2]], axes, length


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
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
            ]
        )
        line_index = [[0, 1], [0, 2], [0, 3]]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(hand_anchor_points)
        line_set.lines = o3d.utility.Vector2iVector(line_index)
        line_set.colors = o3d.utility.Vector3dVector([(1,0,0), (0,1,0), (0,0,1)])
        line_set.transform(pose)
        return line_set

    def compute_candidate_grasps(
        self, 
        pcd: PointCloud, 
        pcd_with_background: PointCloud, 
        candidate_num=30, 
        num_samples=20, 
        random_seed=5,
        pair=True,
        align_grasp_axis = [0, 0, 1], 
        align_minor_axis = [1, 0, 0], 
        split_axis=2, 
        minor_split_axis=0
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

        split_ratio_threshold = 0.6

        # NOTE: All num_samples should be odd numbers
        y_min = -0.01
        y_max = 0.01
        num_y_samples = 3
        roll_min = -np.pi / 2
        roll_max = np.pi / 2
        num_roll_samples = 7
        pitch_min = -np.pi / 4
        pitch_max = np.pi / 4
        num_pitch_samples = 5
        # TODO: Look into exploiting Panda gripper symmetry (grasps rotated by n*pi should be equivalent)
        yaw_min = -np.pi / 2
        yaw_max = np.pi / 2
        num_yaw_samples = 7

        np.random.seed(random_seed)

        pcd_points = pcd.xyzs().T

        # Filter pcd based on split ratio: must pass threshold for any 2/3 axes
        split_ratios, split_axes, length = self.compute_pcd_split_ratio(pcd_points, viz_split_ratio_axes=False)
        mask = np.any(split_ratios > split_ratio_threshold, axis=1)
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
        VISUALIZE = False
        VISUALIZE_EACH = False
        VISUALIZE_ALL = False

        start_time = time.time()
        if not PARALLEL:
            # Local grid search around darboux frames
            candidate_lst: List[np.ndarray] = []
            candidate_lst_by_grasp_origin_pt: List[np.ndarray] = []
            candidate_costs: List[float] = []
            for X_WP, split_ratio in zip(X_WPs, split_ratios[darboux_frame_sample_indices]):
                color = np.random.rand(3)
                color /= np.linalg.norm(color)
                color = tuple(color)# NOTE: The best variations to sample/ search over is situation/ grasp environment dependent (e.g. bin vs table)
                for y in np.linspace(y_min, y_max, num_y_samples):
                    for pitch in np.linspace(pitch_min, pitch_max, num_pitch_samples):
                        for roll in np.linspace(roll_min, roll_max, num_roll_samples):
                            for yaw in np.linspace(yaw_min, yaw_max, num_yaw_samples):
                                # TODO: Explore whether it is faster to do this transform in numpy
                                X_PPnew = RigidTransform(RollPitchYaw(roll, pitch, yaw), np.array([0, y, 0]))
                                X_WPnew = X_WP.multiply(X_PPnew)

                                if VISUALIZE:
                                    # visualize
                                    manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
                                    manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])
                                    o3d.visualization.draw_geometries([manipuland_cloud, self.make_triad_line_set(X_WP.GetAsMatrix4(), [0.0, 1.0, 0.0])])
                                    print("darboux frame", X_WP)

                
                                R_WPnew = X_WPnew.GetAsMatrix4()[:3, :3]
                                eff_vertical_vec = R_WPnew.dot(np.array([0, 0, 1])) # don't want it to face up
                                if eff_vertical_vec[2] > 0:
                                    continue
                                
                                # Compute a new transform that minimizes y-direction distance without penetration
                                # Use pcd with floor to avoid gripper smashing into the table
                                distance, X_WPnew = self.find_minimum_distance(pcd_with_background, X_WPnew)
                                # If distance cannot be found, go over to the next iteration
                                if np.isnan(distance):
                                    continue

                                # If the candidate has no collisions and the closing region is non
                                # empty, then append it to the list of candidates.
                                is_nonempty, within_box_pt_normals, proportion_enclosed = self.check_nonempty(pcd, X_WPnew)
                                if is_nonempty:
                                    candidate_lst.append(X_WPnew.GetAsMatrix4())
                                    candidate_lst_by_grasp_origin_pt.append(X_WP.GetAsMatrix4())
                                    viz_geoms.append(self.make_gripper_line_set(X_WPnew.GetAsMatrix4(), color))
                                    if pair:
                                        candidate_costs.append(
                                            self.compute_costs(
                                                X_WPnew, 
                                                within_box_pt_normals,
                                                proportion_enclosed, 
                                                split_ratio,
                                                split_axes
                                            )
                                        )
                                    else:
                                        candidate_costs.append(
                                            self.compute_costs_single(
                                                X_WPnew, 
                                                within_box_pt_normals, 
                                                proportion_enclosed,
                                                split_ratio, 
                                                split_axis, 
                                                minor_split_axis, 
                                                align_grasp_axis, 
                                                align_minor_axis
                                            )
                                        )
                                    if VISUALIZE_EACH:
                                        # print(candidate_costs[-1])
                                        manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
                                        manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

                                        gripper_xyzs = self.hand_collision_model.to_pcd()
                                        gripper_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(0.005).transform((X_WPnew @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0])).GetAsMatrix4())
                                        gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])

                                        viz_geoms = [manipuland_cloud, gripper_cloud]
                                        o3d.visualization.draw_plotly(viz_geoms)

            print("sequential antipodal grasp time: {:.3f}".format(time.time() - start_time))
            if VISUALIZE_ALL:
                o3d.visualization.draw_geometries(viz_geoms)

        else:
            y_split = np.linspace(y_min, y_max, num_y_samples)

            roll_split = np.linspace(roll_min, roll_max, num_roll_samples)
            pitch_split = np.linspace(pitch_min, pitch_max, num_pitch_samples)
            yaw_split = np.linspace(yaw_min, yaw_max, num_yaw_samples)
            # List of all possible roll, pitch, yaw combinations
            rot_list = np.stack(np.meshgrid(roll_split, pitch_split, yaw_split), axis=-1).reshape(-1, 3)
            batch_rot = to_rotation_matrices(torch.from_numpy(rot_list)).detach().numpy()

            pcd_W_np = pcd.xyzs()[np.newaxis, :]
            pcd_W_normals = pcd.normals()[np.newaxis, :]

            candidate_lst: List[np.ndarray] = []
            candidate_lst_by_grasp_origin_pt: List[np.ndarray] = []
            candidate_costs: List[float] = []
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

                grasp_costs = self.compute_costs_batch(X_WPnew_batch_no_penetration, within_box_pt_normals, split_ratio, split_axes)

                # Pick valid grasps and associated costs
                candidate_lst.append(X_WPnew_batch_nonempty)
                candidate_lst_by_grasp_origin_pt.append(X_WP_batch)
                candidate_costs.append(grasp_costs[cage_mask])

            candidate_lst = np.concatenate(candidate_lst, axis=0)
            candidate_costs = np.concatenate(candidate_costs, axis=0)

            print("parallel antipodal grasp time: {:.3f}".format(time.time() - start_time))
            if VISUALIZE_ALL:
                for X_G in candidate_lst:
                    viz_geoms.append(self.make_gripper_line_set(X_G, [0, 0, 1]))
                o3d.visualization.draw_geometries(viz_geoms)

        start = time.time()

        if pair:
            # Two grasp selection
            candidate_lst = np.array(candidate_lst)
            candidate_lst_by_grasp_origin_pt = np.array(candidate_lst_by_grasp_origin_pt)
            candidate_costs = np.array(candidate_costs)
            sorted_candidate_inds = np.argsort(candidate_costs)[:len(candidate_costs) // 2]
            candidates_filtered = candidate_lst[sorted_candidate_inds]
            candidates_grasp_origin_filtered = candidate_lst_by_grasp_origin_pt[sorted_candidate_inds]
            candidate_costs_filtered = candidate_costs[sorted_candidate_inds]
        
            # Extract translations from grasp origin pts (last column of each 4x4 matrix)
            translations = candidates_grasp_origin_filtered[:, :3, 3]
            
            # Compute pairwise translation differences
            translation_diffs = np.linalg.norm(translations[:, np.newaxis] - translations[np.newaxis, :], axis=-1)
            translation_cost = -translation_diffs / length
            
            # Extract rotations (top-left 3x3 part of each 4x4 matrix)
            rotations = candidates_filtered[:, :3, :3]
            
            # Compute pairwise rotational differences
            # R_relative = R2^T @ R1 (for each pair of rotations R1, R2)
            relative_rotations = np.einsum('nij,mjk->nmik', rotations, rotations.transpose(0, 2, 1))
            
            # Compute the angle of rotation from the relative rotation matrices
            # Trace(R_relative) = 1 + 2*cos(theta), where theta is the angle of rotation
            trace_relative_rotations = np.einsum('...ii', relative_rotations)
            rotation_diffs = np.arccos(np.clip((trace_relative_rotations - 1) / 2, -1.0, 1.0))  # Avoid precision errors
            rotation_cost = np.abs(np.pi/2 - rotation_diffs) / (np.pi/2)

            grasps_quality = candidate_costs_filtered[:, np.newaxis] + candidate_costs_filtered[np.newaxis, :]

            pair_costs = grasps_quality + 20 * translation_cost + 20 * rotation_cost
            pair_costs = np.triu(pair_costs, k=1) + np.tril(np.inf * np.ones_like(pair_costs)) # make lower + diagonal infinity to avoid double counting
            pair_costs = pair_costs.flatten()
            pairs = [(X_WG1, X_WG2) for X_WG1 in candidates_filtered for X_WG2 in candidates_filtered]

            sorted_pair_indices = np.argsort(pair_costs)
            pair_lst_sorted = [pairs[idx] for idx in sorted_pair_indices]

            print("pair selection time:", time.time()-start)
            # List of grasp pairs
            self.grasp_candidates: List[Tuple[np.ndarray]] = pair_lst_sorted
        else:
            sorted_indices = np.argsort(candidate_costs)
            candidate_lst_sorted = [candidate_lst[idx] for idx in sorted_indices]

            self.grasp_candidates: List[np.ndarray] = candidate_lst_sorted

    def get_best_grasps(self, candidate_num=-1) -> List[Tuple[np.ndarray]]:
        """Returns a list of the `candidate_num` grasps with the lowest cost."""
        if candidate_num == -1:
            return self.grasp_candidates
        return self.grasp_candidates[:candidate_num]