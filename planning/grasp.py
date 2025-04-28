# Mostly taken from https://github.com/nepfaff/rlg_panda_stack/blob/main/src/perception/grasp_node.py with a lot of changes
import open3d.visualization.gui as gui
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
    Concatenate
)
from manipulation.utils import ConfigureParser
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
from planning.misc.sdf_tools import SignedDensityField
import time
import multiprocessing
from multiprocessing.pool import ThreadPool as Pool
import torch
from enum import Enum
import math
import random
from perception.pcd_util import crop_connected_points

from .frame_placer_app import FramePlacerApp
lock = threading.Lock()

PROCESSES = multiprocessing.cpu_count()

class GraspType(Enum):
    PAIR = 1
    SIDE = 2
    TOP = 3
    STABLE = 4
    ADDITIONAL = 5

class GraspListener():
    """The class responsible for computing and evaluation grasp candidates."""

    def __init__(self, gripper_length=0.145, hand_finger_path=None, hand_finger_extra_buffer_path=None, gripper_model_path=None):
        if hand_finger_path == None:
            hand_finger_path = os.path.abspath(
                os.path.join(os.path.dirname( __file__ ), '..', 'scenario_datas', 'gripper_sdf.pkl'))
        if hand_finger_extra_buffer_path == None:
            hand_finger_extra_buffer_path = os.path.abspath(
                os.path.join(os.path.dirname( __file__ ), '..', 'scenario_datas', 'large_gripper_sdf.pkl'))
        print("Loading hand collision model from ", hand_finger_path)
        self.hand_collision_model = SignedDensityField.from_pkl(hand_finger_path)
        print("Loading extra buffer hand collision model from ", hand_finger_extra_buffer_path)
        self.hand_extra_buffer_collision_model = SignedDensityField.from_pkl(hand_finger_extra_buffer_path)
        # self.hand_collision_model.visualize()

        self.gripper_length = gripper_length

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

        self.voxel_map = None

        # Required for using gui. Should only ever call this once.
        gui.Application.instance.initialize()

    def set_voxel_map(self, voxel_map):
        """Set the voxel map for the grasp listener."""
        self.voxel_map = voxel_map

    def check_collision(self, pcd, X_G, use_extra_buffer=False, visualize=False):
        """Returns true if not in collision and false otherwise."""
        thre = 0.0
        sdf = self.compute_sdf_fast(pcd, X_G, use_extra_buffer, visualize)
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

        nvec = eigvec[:, order[2]] # normal vector
        tvec_major = eigvec[:, order[1]] # biggest curvature
        tvec_minor = eigvec[:, order[0]] # smallest curvature

        # If the z-direction is aligned with the normal vector, flip the z direction.
        # We want the frame's z-axis to point into the cloud.
        if nvec.dot(normal) > 0:
            nvec = -nvec

        # Frame z-axis pointing outwards from pcd, y axis pointing along pcd
        R = np.vstack((tvec_major, tvec_minor, nvec)).T
        # Ensure right-handed coordinate system
        if np.linalg.det(R) < 0:
            tvec_major = -tvec_major
            R = np.vstack((tvec_major, tvec_minor, nvec)).T

        R = RotationMatrix.ProjectToRotationMatrix(R)
        finger_tip_translation = np.array([0, 0, 0])
        # need the transform from finger tip to grasp
        X_WF = RigidTransform(RotationMatrix(R), point + R @ finger_tip_translation)

        return X_WF

    def compute_darboux_frames(
        self, points, normals, pcd, kdtree, ball_radius=0.002, max_nn=50, point_up=False,
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
            if point_up:
                # Construct a fake darboux frame whose z axis points down in the world frame.
                frame = RigidTransform(RotationMatrix(np.eye(3)) @ RotationMatrix.MakeXRotation(-np.pi), point)
            else:
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

    def compute_sdf_fast(self, pcd, X_G, use_extra_buffer=False, visualize=False):
        """A lookup to the pre-computed sdf of the hand collision model."""
        pcd_W_np = pcd.xyzs()
        # RollPitchYaw(np.pi/2, 0, np.pi/2) is world to wsg specific transform.
        # WSG y axis needs to be aligned with world -z.
        X_GW = (X_G @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0])).inverse()
        pcd_G_np = X_GW.multiply(pcd_W_np)
        if use_extra_buffer:
            dist = self.hand_extra_buffer_collision_model.get_distance(pcd_G_np.T)
        else:
            dist = self.hand_collision_model.get_distance(pcd_G_np.T)

        if visualize:
            manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_G_np.T))
            manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

            if use_extra_buffer:
                gripper_xyzs = self.hand_extra_buffer_collision_model.to_pcd()
            else:
                gripper_xyzs = self.hand_collision_model.to_pcd()
            gripper_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(0.005)
            gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])

            viz_geoms = [manipuland_cloud, gripper_cloud]
            o3d.visualization.draw_geometries(viz_geoms)

        return dist.min()

    def check_collision_batch(self, pcd, X_G, visualize=False):
        """Returns true if not in collision and false otherwise."""
        thre = 0.0
        sdf = self.compute_sdf_batch(pcd, X_G, visualize)
        return sdf > thre

    def compute_sdf_batch(self, pcd: np.ndarray, X_Gs: np.ndarray, use_extra_buffer=False, visualize=False):
        """A lookup to the pre-computed sdf of the hand collision model.
        parallelize over the transforms
        :param X_Gs: m x 4 x 4
        :param pcd_W_np: n x 3
        """
        R = RotationMatrix(RollPitchYaw(np.pi/2, 0, np.pi/2)).matrix()
        pcd_points = pcd.xyzs().T

        # Convert R to a 4x4 transformation matrix (embedding the rotation in a 4x4 matrix)
        R_4x4 = np.eye(4)
        R_4x4[:3, :3] = R
        R_4x4[3, 3] = 1

        # Perform the operation on all matrices in the batch
        X_GWs = np.linalg.inv(X_Gs @ R_4x4)
        pcd_G_np = (X_GWs[:, :3, :3] @ pcd_points.T) + X_GWs[:, :3, 3, None]

        # get_distance only requires the last dimension to be 3: ... x 3 -> ... x 1
        if use_extra_buffer:
            dist = self.hand_extra_buffer_collision_model.get_distance(pcd_G_np.transpose(0, 2, 1))
            gripper_xyzs = self.hand_extra_buffer_collision_model.to_pcd()
        else:
            dist = self.hand_collision_model.get_distance(pcd_G_np.transpose(0, 2, 1))
            gripper_xyzs = self.hand_extra_buffer_collision_model.to_pcd()
        
        if visualize:
            gripper_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(0.005)
            gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])
            for i in range(pcd_G_np.shape[0]):
                scene_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_G_np[i, :, :].T))
                o3d.visualization.draw_geometries([gripper_cloud, scene_cloud])

        return dist.reshape(len(dist), -1).min(axis=-1)

    def find_minimum_distance(self, pcd, X_WG, thre=0.0, min_range=-0.11, max_range=-0.01, num_samples=10, use_extra_buffer=False, backwards=False, viz=False):
        """
        By doing line search, compute the maximum allowable distance along the z axis before penetration.
        Return the maximum distance, as well as the new transform. Returns (np.nan, None) if nothing is returned after
        line search.

        NOTE: This does not consider the collision scene (e.g. table) but only the object point cloud.
        """
        if use_extra_buffer:
            max_range = min_range + (max_range - min_range) / 2
            num_samples = num_samples // 2
        z_grid = np.linspace(min_range, max_range, num_samples)
        signed_distance = -np.inf
        X_WGnew = RigidTransform()

        X_WGlast = None
        last_signed_distance = np.nan
        for z in z_grid:
            # Compute new values.
            X_WGnew = X_WG.multiply(RigidTransform([0.0, 0.0, z]))

            signed_distance = self.compute_sdf_fast(pcd, X_WGnew, use_extra_buffer=use_extra_buffer)

            # If the value crossed for the first time, return.
            if signed_distance < thre:
                if X_WGlast is None:
                    break

                if viz:
                    manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
                    manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])
    
                    gripper_xyzs = self.hand_collision_model.to_pcd()
                    if use_extra_buffer:
                        gripper_xyzs = self.hand_extra_buffer_collision_model.to_pcd()
                    gripper_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(
                        0.005).transform((X_WGlast @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0])).GetAsMatrix4())
                    gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])
                    
                    gripper_next_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(
                        0.005).transform((X_WGnew @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0])).GetAsMatrix4())
                    gripper_next_cloud.paint_uniform_color([0.0, 1.0, 0.0])

                    viz_geoms = [manipuland_cloud, gripper_cloud, gripper_next_cloud]
                    o3d.visualization.draw_plotly(viz_geoms)
                    # input(f"Found grasp {last_signed_distance}, {signed_distance}, {X_WGlast}")
                return last_signed_distance, X_WGlast
            
            # Record the computed values using last z.
            last_signed_distance = signed_distance
            X_WGlast = X_WGnew

        if backwards:
            z_grid = np.linspace(min_range, min_range + (max_range-min_range), num_samples)
            X_WGlast = None
            last_signed_distance = np.nan
            for z in z_grid:
                # Compute new values.
                X_WGnew = X_WG.multiply(RigidTransform([0.0, 0.0, z]))

                signed_distance = self.compute_sdf_fast(pcd, X_WGnew, use_extra_buffer=use_extra_buffer)

                # If the value crossed for the first time, return.
                if signed_distance > thre:
                    if viz:
                        manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
                        manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

                        gripper_xyzs = self.hand_collision_model.to_pcd()
                        if use_extra_buffer:
                            gripper_xyzs = self.hand_extra_buffer_collision_model.to_pcd()
                        gripper_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(
                            0.005).transform((X_WGlast @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0])).GetAsMatrix4())
                        gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])
                        
                        gripper_next_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(
                            0.005).transform((X_WGnew @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0])).GetAsMatrix4())
                        gripper_next_cloud.paint_uniform_color([0.0, 1.0, 0.0])

                        viz_geoms = [manipuland_cloud, gripper_cloud, gripper_next_cloud]
                        o3d.visualization.draw_plotly(viz_geoms)
                        # input(f"Found grasp going backwards {last_signed_distance}, {signed_distance}, {X_WGlast}")
                    return signed_distance, X_WGnew
                
                # Record the computed values using last z.
                last_signed_distance = signed_distance
                X_WGlast = X_WGnew

        # If nothing is returned after line search, discard the sample by sending None.
        return np.nan, None

    def find_minimum_distance_batch(self, pcd, X_WG, thre=0.0, min_range=-0.11, max_range=0.11, num_samples=20, use_extra_buffer=False, viz=False):
        """
        By doing line search, compute the maximum allowable distance along the z axis before penetration.
        Return the maximum distance, as well as the new transform. Returns (np.nan, None) if nothing is returned after
        line search.

        NOTE: This does not consider the collision scene (e.g. table) but only the object point cloud.
        """
        z_grid = np.linspace(min_range, max_range, num_samples)

        # Create a batch of translation vectors for each z value (translation only along the z-axis)
        translations = np.zeros((num_samples, 4, 4))
        translations[:, 3, 2] = z_grid  # Place each z value in the last column of the 4x4 matrix
        translations[:,:3, :3] = np.eye(3)
        translations[:, 3, 3] = 1

        # Apply the transformations in one batch
        X_WGs = X_WG.GetAsMatrix4() @ translations.transpose(0, 2, 1)  # Apply the transformation in batch
        dists = self.compute_sdf_batch(pcd, X_WGs, use_extra_buffer=use_extra_buffer) # n x num_samples x 3

        threshold_met = dists < thre
        valid_indices = np.where(threshold_met)[0]

        if len(valid_indices) > 0:
            first_index = valid_indices[0]
            return dists[first_index], RigidTransform(X_WGs[first_index])
        
        # If no valid index is found, return np.nan and None
        return np.nan, None

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
                The gripper frame has the y-axis connecting the two fingers and the z-axis pointing from the
                gripper body to the fingers.
            - proportion_enclosed: proportion of normals enclosed out of all possible points in PCD
            - indices_enclosed: indices of points in the closing region
        """
        pcd_W_np = pcd.xyzs()
        pcd_W_normals = pcd.normals()

        # Bounding box of the closing region written in the coordinate frame of the gripper body.
        # z-axis points from gripper body to fingers, y-axis is the grasping axis.
        # It is recommended to tune this while inspecting the crop_cloud with the visualize option.
        crop_min = [-0.0125, -0.053, 0.025]
        crop_max = [0.0125, 0.053, 0.025+0.10] # finray is ~11cm long

        # Transform the pointcloud to gripper frame.
        X_GW = X_WG.inverse()
        R_GW = X_GW.GetAsMatrix4()[:3, :3]
        pcd_G_np = X_GW.multiply(pcd_W_np)
        pcd_normals_G_np = R_GW @ pcd_W_normals  # in the grasp frame

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
            num_points = 1000  # Adjust for density
            points = np.random.uniform(low=crop_min, high=crop_max, size=(num_points, 3))
            crop_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)).transform(
                X_WG.GetAsMatrix4()
            )
            crop_pcd.paint_uniform_color([1.0, 0.5, 0.0])

            gripper_xyzs = self.hand_collision_model.to_pcd()
            gripper_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(
                0.005).transform((X_WG @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0])).GetAsMatrix4())
            gripper_cloud.paint_uniform_color([0.0, 1.0, 0.0])

            gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05).transform(
                (X_WG @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0])).GetAsMatrix4()
            )

            crop_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05).transform(
                X_WG.GetAsMatrix4()
            )
            
            pcd_closing_region = pcd.xyzs().T[indices, :]
            pcd_closing_region_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_closing_region))
            pcd_closing_region_normals = pcd_W_normals[:, indices].T
            pcd_closing_region_cloud.normals = o3d.utility.Vector3dVector(pcd_closing_region_normals)
            pcd_closing_region_cloud.paint_uniform_color([1.0, 0.0, 0.0])
            manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
            manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])
            viz_geoms = [
                crop_pcd,
                crop_frame,
                # gripper_frame,
                gripper_cloud,
                manipuland_cloud,
                pcd_closing_region_cloud,
                self.make_gripper_line_set(X_WG.GetAsMatrix4(), [0.0, 1.0, 0.0])
            ]
            o3d.visualization.draw_geometries(viz_geoms, point_show_normal=True)
        
        proportion_enclosed = len(indices) / (pcd_normals_G_np.shape[1])

        return is_nonempty, pcd_normals_G_np[:, indices], proportion_enclosed, indices

    def compute_costs(
            self, 
            X_WG: RigidTransform, 
            within_box_pt_normals: np.ndarray,
            split_axes: np.ndarray,
            center: np.ndarray,
            half_range: np.ndarray,
            ground_z: float,
            object_height: float,
            num_pcd_pts: float,
            proportion_enclosed_cost_thresh: float=0.1,
            proportion_good_enclosed_cost_thresh: float=0.01,
            proportion_enclosed_good_cost_thresh: float=0.1,
            ) -> tuple[float, dict]:
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
        grasp_center = (X_WG @ RigidTransform([0, 0.0, 3*self.gripper_length/4])).translation()

        rot = X_WG.GetAsMatrix4()[:3, :3]
        eff_vertical_vec = rot.dot(np.array([0, 0, 1])) # vertical axis of gripper (parallel to fingers)
        
        antipodal_within_grasp_cost = -np.sum(
            np.exp(within_box_pt_normals[1, :] ** 2)
        ) / within_box_pt_normals.shape[1] # along the horizontal axis of the gripper, larger good (antipodal metric)

        good_normals_within_grasp = np.sum(within_box_pt_normals[1, :] > 0.9)
        # antipodal_within_grasp_cost = -good_normals_within_grasp / within_box_pt_normals.shape[1]

        antipodal_cost = -np.sum(
            np.exp(within_box_pt_normals[1, :] ** 2)
        ) / num_pcd_pts # along the horizontal axis of the gripper, larger good (antipodal metric)

        # want grasps to avoid alignment with long axes, smaller better
        gripper_vertical_axis_alignment_cost = np.abs(eff_vertical_vec @ split_axes)
        min_high_enough = ground_z + 3*object_height/4
        higher_up_cost = -min(grasp_center[2] - min_high_enough, 0) / min_high_enough

        # split ratio cost along long axis at the grasp point, not origin point
        z_axis, x_axis = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
        rot_principal_component_to_axes, _ = R.align_vectors(
            np.array([z_axis, x_axis]), np.stack([split_axes[0], split_axes[2]])
        )

        point_axis_aligned = grasp_center @ rot_principal_component_to_axes.as_matrix().T

        decay_rate = 2.0
        normalized_distance = np.abs((point_axis_aligned - center) / half_range)
        normalized_distance = np.clip(normalized_distance, 0, 1)
        split_ratios = np.exp(-decay_rate * normalized_distance) - np.exp(-decay_rate)
        ranked_split_ratios = np.sort(split_ratios[0], axis=0)
        split_ratio_cost = -(ranked_split_ratios[1] + ranked_split_ratios[2])

        proportion_enclosed = within_box_pt_normals.shape[1]/num_pcd_pts
        proportion_enclosed_cost = 1e3 if proportion_enclosed < proportion_enclosed_cost_thresh else 0
        proportion_good_enclosed = good_normals_within_grasp/num_pcd_pts
        proportion_good_enclosed_cost = 1e3 if proportion_good_enclosed < proportion_good_enclosed_cost_thresh else 0
        proportion_enclosed_good = good_normals_within_grasp/within_box_pt_normals.shape[1]
        proportion_enclosed_good_cost = 1e3 if proportion_enclosed_good < proportion_enclosed_good_cost_thresh else 0

        cost_dict = {
            "antipodal_cost": 200.0 * antipodal_cost,
            "antipodal_within_grasp_cost": 50.0 * antipodal_within_grasp_cost,
            "gripper_vertical_axis_alignment_cost_principal": -10.0 * gripper_vertical_axis_alignment_cost[0],
            "gripper_vertical_axis_alignment_cost_secondary": -5.0 * gripper_vertical_axis_alignment_cost[1],
            "higher_up_cost": 50.0 * higher_up_cost,
            "proportion_enclosed": proportion_enclosed,
            "proportion_enclosed_cost": proportion_enclosed_cost,
            "proportion_good_enclosed": proportion_good_enclosed,
            "proportion_good_enclosed_cost": proportion_good_enclosed_cost,
            "proportion_enclosed_good": proportion_enclosed_good,
            "proportion_enclosed_good_cost": proportion_enclosed_good_cost,
            "num_pcd_pts": num_pcd_pts,
            "within_box_pt_normals.shape[1]": within_box_pt_normals.shape[1],
            "good_normals_within_grasp": good_normals_within_grasp,
            "split_ratio_cost": 100 * split_ratio_cost
        }

        considered_costs = [
            cost_dict["antipodal_cost"],
            cost_dict["antipodal_within_grasp_cost"],
            cost_dict["higher_up_cost"],
            cost_dict["split_ratio_cost"],
            cost_dict["proportion_enclosed_cost"],
            cost_dict["proportion_good_enclosed_cost"],
            cost_dict["proportion_enclosed_good_cost"],
        ]
        cost = sum(considered_costs)

        return cost, cost_dict
    
    def compute_costs_single(
            self, 
            X_WG: RigidTransform, 
            within_box_pt_normals: np.ndarray, 
            proportion_enclosed: float,
            split_ratios: np.ndarray, 
            major_split_axis: int,
            minor_split_axis: int,
            align_grasp_axis: List[float]=[0,0,1],
            align_minor_axis: List[float]=[1,0,0],
            ) -> tuple[float, dict]:
        R = X_WG.GetAsMatrix4()[:3, :3]
        t = X_WG.GetAsMatrix4()[:3, 3]
        eff_vertical_vec = R.dot(np.array([0, 0, 1])) # vertical axis of gripper (parallel to fingers)
        eff_horizontal_vec = R.dot(np.array([0, 1, 0])) # horizontal axis of gripper (line connecting finger tips)
        
        antipodal_cost = -np.sum(
            within_box_pt_normals[1, :] ** 2
        ) / within_box_pt_normals.shape[1] # along the horizontal axis of the gripper, larger good (antipodal metric)

        gripper_axis_alignment_cost = -np.abs(eff_vertical_vec @ align_grasp_axis)  # want vertical axis of gripper to face towards desired axis
        gripper_minor_alignment_cost = -np.abs(eff_horizontal_vec @ align_minor_axis) # want horizontal axis of gripper to align with minor axis 
        split_ratio_minor_axis_cost = -split_ratios[minor_split_axis]  # prefer higher split ratio
        split_ratio_major_axis_cost = -split_ratios[major_split_axis]

        proportion_enclosed_cost = -proportion_enclosed
        cost_dict = {
            "antipodal_cost": 100.0 * antipodal_cost,
            "gripper_axis_alignment_cost": 10.0 * gripper_axis_alignment_cost,
            "gripper_minor_alignment_cost": 5.0 * gripper_minor_alignment_cost,
            "split_ratio_minor_axis_cost": 5.0 * split_ratio_minor_axis_cost,
            "split_ratio_major_axis_cost": 5.0 * split_ratio_major_axis_cost,
            # "proportion_enclosed_cost": 100.0 * proportion_enclosed_cost 
        }

        cost = sum(cost_dict.values())
       
        return cost, cost_dict
    
    def compute_costs_top(
            self, 
            X_WG: RigidTransform, 
            within_box_pt_normals: np.ndarray, 
            split_ratios: float,
            split_axes: np.ndarray,
            proportion_enclosed: float,
            viz = False,
            pcd=None
        ) -> tuple[float, dict]:
        """
        Computes a grasp candidate cost based on a weighted sum of:
        - Antipodal (grasp normal) cost (prefer more antipodal)
        - Gripper vertical alignment cost (prefer more aligned with world z-axis)
        - Vertical position cost (prefer higher grasps)
        :param X_WG: The grasp candidate to compute the cost for.
        :param within_box_pt_normals: Point cloud normals within the gripper closing region of shape (3, N).
        :param split_ratios: Array of [x split ratio, y split ratio]. Values are in range [0,1] where
            higher indicates a more equal split along the principal object axis.
        """
        rot = X_WG.GetAsMatrix4()[:3, :3]
        t = X_WG.GetAsMatrix4()[:3, 3]
        eff_vertical_vec = rot.dot(np.array([0, 0, 1]))
        eff_x_vec = rot[:, 0]
        eff_y_vec = rot.dot(np.array([0, 1, 0]))
        gripper_x_axis_alignment_cost = (eff_x_vec @ split_axes[0].T)

        # Rotate point cloud to align the principal component with the z-axis and the minor component with the x-axis
        z_axis, y_axis = [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]
        rot_principal_component_to_axes, _ = R.align_vectors(
            np.array([z_axis, y_axis]), np.stack([split_axes[0], split_axes[1]])
        )
        if viz:
            rot = RotationMatrix(rot_principal_component_to_axes.as_matrix().T)
            T = np.eye(4)
            T[:3,:3] = rot.matrix()
            T[:3, 3] = t - [0, 0, 0.03]
            pca = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            pca.transform(T)
            grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05).transform(
                X_WG.GetAsMatrix4()
            )
            pcd_viz = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
            print(gripper_x_axis_alignment_cost)
            o3d.visualization.draw_geometries([pcd_viz, grasp_frame, pca])

        antipodal_cost = -np.sum(
            within_box_pt_normals[1, :] ** 2
        )  # along the y axis of the gripper, larger good (antipodal metric)
        gripper_vertical_alignment_cost = eff_vertical_vec[2]  # want z axis of gripper to face down, larger worse
        gripper_x_alignment_cost = np.abs(eff_x_vec[0])
        gripper_y_alignment_cost = np.abs(eff_y_vec[1])
        grasp_height_cost = -t[2]  # prefer higher position
        split_ratio_minor_axis_cost = -split_ratios[0]  # prefer higher split ratio
        split_ratio_major_axis_cost = -split_ratios[1]
        
        proportion_enclosed_cost = -proportion_enclosed
        cost_dict = {
            # Antipodal doesn't make sense for partial point clouds
            "antipodal_cost": 0 * antipodal_cost,
            "gripper_vertical_alignment_cost": 50.0 * gripper_vertical_alignment_cost,
            # Alignment scores are in range [-1,0] where -1 indicates perfect alignment with one of the axes. 
            # "xy_alignment_cost": -50.0 * max(gripper_x_alignment_cost, gripper_y_alignment_cost),
            "x_principal_alignment_cost": -100.0 * gripper_x_axis_alignment_cost,
            "grasp_height_cost": grasp_height_cost,
            # Split ratio doesn't make sense on partial point clouds
            "split_ratio_minor_axis_cost": 0*split_ratio_minor_axis_cost, # This is world x-axis
            "split_ratio_major_axis_cost": 0*split_ratio_major_axis_cost,
            # proportion_enclosed is the propertion of total pcd points that are within the fingers
            "proportion_enclosed_cost": 200.0 * proportion_enclosed_cost
        }
        cost = sum(cost_dict.values())

        return cost, cost_dict
    
    def compute_costs_stable(
            self, 
            X_WG: RigidTransform, 
            within_box_pt_normals: np.ndarray, 
            split_ratios: float,
            proportion_enclosed: float,
            num_pcd_pts: float,
        ) -> tuple[float, dict]:
        """
        Computes a grasp candidate cost based on a weighted sum of:
        - Antipodal (grasp normal) cost (prefer more antipodal)
        - Vertical position cost (prefer higher grasps)
        :param X_WG: The grasp candidate to compute the cost for.
        :param within_box_pt_normals: Point cloud normals within the gripper closing region of shape (3, N).
        :param split_ratios: Array of [minor axis split ratio, secondary axis split ratio, major axis split ratio].
            Values are in range [0,1] where higher indicates a more equal split along the principal object axis.
        """
        R = X_WG.GetAsMatrix4()[:3, :3]
        t = X_WG.GetAsMatrix4()[:3, 3]
        eff_vertical_vec = R.dot(np.array([0, 0, 1])) # vertical axis of gripper (parallel to fingers)
        
        antipodal_within_grasp_cost = -np.sum(
            np.exp(within_box_pt_normals[1, :] ** 2)
        ) / within_box_pt_normals.shape[1] # along the horizontal axis of the gripper, larger good (antipodal metric)

        good_normals_within_grasp = np.sum(within_box_pt_normals[1, :] > 0.9)
        # antipodal_within_grasp_cost = -good_normals_within_grasp / within_box_pt_normals.shape[1]

        antipodal_cost = -np.sum(
            np.exp(within_box_pt_normals[1, :] ** 2)
        ) / num_pcd_pts # along the horizontal axis of the gripper, larger good (antipodal metric)

        proportion_enclosed = within_box_pt_normals.shape[1]/num_pcd_pts
        proportion_enclosed_cost = 1e3 if proportion_enclosed < 0.05 else 0
        proportion_good_enclosed = good_normals_within_grasp/num_pcd_pts
        proportion_good_enclosed_cost = 1e3 if proportion_good_enclosed < 0.01 else 0
        proportion_enclosed_good = good_normals_within_grasp/within_box_pt_normals.shape[1]
        proportion_enclosed_good_cost = 1e3 if proportion_enclosed_good < 0.1 else 0

        # Prefer alignment to one of the axes. This is a decent heuristic for standing objects.
        eff_x_vec = R.dot(np.array([1, 0, 0]))
        eff_y_vec = R.dot(np.array([0, 1, 0]))
        eff_z_vec = R.dot(np.array([0, 0, 1]))

        # Vertical grasps
        gripper_z_to_world_z_alignment_score = np.abs(eff_z_vec[2])
        # Right angled grasps
        gripper_x_to_world_z_alignment_score = np.abs(np.dot(eff_x_vec, np.array([0, 0, 1])))
      
        # Prefer side grasps
        axis_alignment_cost = -max(gripper_z_to_world_z_alignment_score, 2*gripper_x_to_world_z_alignment_score)
        
        split_ratio_minor_axis_cost = -split_ratios[0]
        split_ratio_major_axis_cost = -split_ratios[2]

        cost_dict = {
            "antipodal_cost": 50.0 * antipodal_cost,
            "antipodal_within_grasp_cost": 100.0 * antipodal_within_grasp_cost,
            "proportion_enclosed": proportion_enclosed,
            "proportion_enclosed_cost": proportion_enclosed_cost,
            "proportion_good_enclosed": proportion_good_enclosed,
            "proportion_good_enclosed_cost": proportion_good_enclosed_cost,
            "proportion_enclosed_good": proportion_enclosed_good,
            "proportion_enclosed_good_cost": proportion_enclosed_good_cost,
            "num_pcd_pts": num_pcd_pts,
            "within_box_pt_normals.shape[1]": within_box_pt_normals.shape[1],
            "good_normals_within_grasp": good_normals_within_grasp,
            "split_ratio_minor_axis_cost": 50 * split_ratio_minor_axis_cost,
            "split_ratio_major_axis_cost": 100 * split_ratio_major_axis_cost,
            "axis_alignment_cost": 100 * axis_alignment_cost,
        }

        considered_costs = [
            cost_dict["antipodal_cost"],
            cost_dict["antipodal_within_grasp_cost"],
            cost_dict["proportion_enclosed_cost"],
            cost_dict["proportion_good_enclosed_cost"],
            cost_dict["proportion_enclosed_good_cost"],
            cost_dict["axis_alignment_cost"],
        ]
        cost = sum(considered_costs)

        return cost, cost_dict
    
    def compute_costs_voxel_map_coverage(
            self,
            X_WG: RigidTransform,
            intrinsic_matrix: np.ndarray,
            width_px: int, 
            height_px: int,
            indices_enclosed: np.ndarray,
        ) -> tuple[float, dict]:
        # TODO: Implement this function similar to in uncertainty_mapping_test.py for 8 camera views for the grasp
        pass


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
    def compute_pcd_split_ratio_cropped(
        pcd_points: np.ndarray,
        radius: float = 0.1,
        voxel_radius: float = 0.005,
        viz=False
    ) -> np.ndarray:
        """
        Computes the x and y split ratios for each point.
        
        The split ratio measures how close a point is to the center of the bounding box along the x and y axes.
        A value of 1 means the point is exactly at the center, while a value of 0 means it is at one of the edges.
        
        The split ratio for a given axis is computed as:

            split_ratio = 1 - abs( (2 * (p - c)) / (max - min) )

        where:
            - p: The coordinate of the point along the given axis (x or y).
            - c: The center of the axis, computed as (max + min) / 2.
            - max: The maximum coordinate value along the axis.
            - min: The minimum coordinate value along the axis.

        This formula normalizes the distance from the center such that:
            - Points **at the center** of the bounding box get `split_ratio = 1`.
            - Points **at the min/max boundary** get `split_ratio = 0`.
            - Points in between receive values in `(0,1]`, depending on their proximity to the center.

        Edge cases:
        - If max == min (i.e., all points have the same value along an axis), the split ratio is set to
            1 to avoid division by zero.

        :param pcd_points: Point cloud points of shape (N,3), where N is the number of points.
        :return: X and Y split ratios for each point, returned as an array of shape (N,2).
        """
        num_points = pcd_points.shape[0]
        accounted_for = np.zeros(num_points, dtype=bool)  # Track accounted points
        split_ratios = np.full((num_points, 3), np.nan)  # Store split ratios per point
        split_axes = np.full((num_points, 3, 3), np.nan)  # Store split axes per point
        lengths = np.full((num_points), np.nan)  # Store lengths per point
        centers = np.full((num_points, 3), np.nan)  # Store center of cluster per point
        half_ranges = np.full((num_points, 3), np.nan)  # Store half ranges of cluster per point
        
        while not np.all(accounted_for):
            # Select an unaccounted point as center
            unaccounted_indices = np.where(~accounted_for)[0]
            center_idx = random.choice(unaccounted_indices)
            center = pcd_points[center_idx, :3]  # Assuming XYZ format
            
            # Crop clustered subset around the center and get mask
            subset, subset_mask = crop_connected_points(pcd_points, center, radius, voxel_radius)
            
            # Mark these points as accounted for
            accounted_for[subset_mask] = True
            
            cov = np.cov(subset.T)
            eigval, eigvec = np.linalg.eig(cov)

            order = eigval.argsort()
            minor_component = eigvec[:, order[0]]
            secondary_component = eigvec[:, order[1]]
            principal_component = eigvec[:, order[2]]

            # Rotate point cloud to align the principal component with the z-axis and the minor component with the x-axis
            z_axis, x_axis = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
            rot_principal_component_to_axes, _ = R.align_vectors(
                np.array([z_axis, x_axis]), np.stack([principal_component, minor_component])
            )
            if viz:
                subset_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(subset))
                subset_pcd.paint_uniform_color([0.7, 0.7, 0.7]) # Gray
                center_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([center]))
                center_pcd.paint_uniform_color([1.0, 0.0, 0.0]) # Red

                rot = RotationMatrix(rot_principal_component_to_axes.as_matrix().T)
                T = np.eye(4)
                T[:3,:3] = rot.matrix()
                T[:3, 3] = center
                pca = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                pca.transform(T)
                
                o3d.visualization.draw_geometries([subset_pcd, center_pcd, pca])

            pcd_points_axis_aligned = pcd_points @ rot_principal_component_to_axes.as_matrix().T

            # Min/max bounds for split ratio
            min_point_vals = np.min(pcd_points_axis_aligned[subset_mask], axis=0)
            max_point_vals = np.max(pcd_points_axis_aligned[subset_mask], axis=0)
            center = (max_point_vals + min_point_vals) / 2
            half_range = (max_point_vals - min_point_vals) / 2
            half_range[np.where(half_range == 0)] = 1 # avoid div by 0

            decay_rate = 2.0
            normalized_distance = np.abs((pcd_points_axis_aligned - center) / half_range)
            normalized_distance = np.clip(normalized_distance, 0, 1)
            split_ratio = np.exp(-decay_rate * normalized_distance) - np.exp(-decay_rate)
            split_ratios[subset_mask] = split_ratio[subset_mask]

            # return split axes
            axes = np.stack([principal_component, secondary_component, minor_component])
            split_axes[subset_mask, :, :] = axes
            length = np.linalg.norm(max_point_vals - min_point_vals)
            lengths[subset_mask] = length
            centers[subset_mask, :] = center
            half_ranges[subset_mask, :] = half_range
        
        return split_ratios, split_axes, lengths, centers, half_ranges

    @staticmethod
    def compute_pcd_split_ratio(pcd_points: np.ndarray, viz_split_ratio_axes: bool = False) -> np.ndarray:
        """
        Computes the split ratios for each pcd point.
        The split ratio has range [0,1] where 1 is best (most equal split) and 0 is worst (most unequal split).

        See `compute_pcd_split_ratio_xy` for more details.

        :param pcd_points: Point cloud points of shape (N,3).
        :param viz_major_axis: Whether to visualize the pcd with the principle and minor axes in open3d.
        :return: Minor, secondary, and major point cloud split ratio for each point of shape (N,3) where the first entry is the
            minor axis, second is the secondary, and last is the major axis.
        """
        cov = np.cov(pcd_points.T)
        eigval, eigvec = np.linalg.eig(cov)

        order = eigval.argsort()
        minor_component = eigvec[:, order[0]]
        secondary_component = eigvec[:, order[1]]
        principal_component = eigvec[:, order[2]]

        # Rotate point cloud to align the principal component with the z-axis and the minor component with the x-axis
        z_axis, x_axis = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
        rot_principal_component_to_axes, _ = R.align_vectors(
            np.array([z_axis, x_axis]), np.stack([principal_component, minor_component])
        )

        pcd_points_axis_aligned = pcd_points @ rot_principal_component_to_axes.as_matrix().T

        # Min/max bounds for split ratio
        min_point_vals = np.min(pcd_points_axis_aligned, axis=0)
        max_point_vals = np.max(pcd_points_axis_aligned, axis=0)
        center = (max_point_vals + min_point_vals) / 2
        half_range = (max_point_vals - min_point_vals) / 2

        # Compute split ratio
        # split_ratio = 1 - np.abs((pcd_points_axis_aligned - center) / half_range)

        # Compute exponential decay split ratio.
        decay_rate = 2.0
        normalized_distance = np.abs((pcd_points_axis_aligned - center) / half_range)
        normalized_distance = np.clip(normalized_distance, 0, 1)
        split_ratio = np.exp(-decay_rate * normalized_distance) - np.exp(-decay_rate)

        # return split axes
        axes = np.stack([principal_component, secondary_component, minor_component])
        length = np.linalg.norm(max_point_vals - min_point_vals)

        if viz_split_ratio_axes:
            # Visualize the point cloud in blue, the principal axis in red, and the minor axis in green
            mean_val = np.mean(pcd_points_axis_aligned, axis=0)
            pcd_points_axis_aligned_normalized = pcd_points_axis_aligned - mean_val
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_points_axis_aligned_normalized))
            pcd.paint_uniform_color([0.7, 0.7, 0.7]) # Gray
            principle_component_line = o3d.geometry.LineSet()
            principle_component_line.points = o3d.utility.Vector3dVector(
                np.array([[0.0, 0.0, -0.3], [0.0, 0.0, 0.3], [-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
            )
            principle_component_line.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 3]]))
            principle_component_line.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

            num_samples = 10
            sampled_indices = np.random.choice(len(pcd_points), num_samples, replace=False)

            sampled_points = pcd_points_axis_aligned_normalized[sampled_indices]
            sampled_ratios = split_ratio[sampled_indices]
            colors = np.random.rand(num_samples, 3)
            
            sampled_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sampled_points))
            sampled_pcd.colors = o3d.utility.Vector3dVector(colors)

            print("Min point vals:", min_point_vals)
            print("Max point vals:", max_point_vals)
            print("Center:", center)
            
            print("Split ratios of points in order [minor, middle, major]:")
            for i, (ratio, color) in enumerate(zip(sampled_ratios, colors)):
                print(f"\033[38;2;{int(color[0]*255)};{int(color[1]*255)};{int(color[2]*255)}m● Point {i}: {ratio}\033[0m"
                      f"    ->   Coordinates: {sampled_points[i]+mean_val}"
                )
            
            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
            o3d.visualization.draw_plotly([pcd, sampled_pcd, principle_component_line, world_frame])

        return split_ratio[:, [0, 1, 2]], axes, length, center, half_range
    
    @staticmethod
    def compute_pcd_split_ratio_at_point(pcd_points: np.ndarray, point: np.ndarray, viz_split_ratio_axes: bool = False) -> np.ndarray:
        """
        Computes the split ratios for each pcd point.
        The split ratio has range [0,1] where 1 is best (most equal split) and 0 is worst (most unequal split).

        See `compute_pcd_split_ratio_xy` for more details.

        :param pcd_points: Point cloud points of shape (N,3).
        :param viz_major_axis: Whether to visualize the pcd with the principle and minor axes in open3d.
        :return: Minor, secondary, and major point cloud split ratio for each point of shape (N,3) where the first entry is the
            minor axis, second is the secondary, and last is the major axis.
        """
        cov = np.cov(pcd_points.T)
        eigval, eigvec = np.linalg.eig(cov)

        order = eigval.argsort()
        minor_component = eigvec[:, order[0]]
        secondary_component = eigvec[:, order[1]]
        principal_component = eigvec[:, order[2]]

        # Rotate point cloud to align the principal component with the z-axis and the minor component with the x-axis
        z_axis, x_axis = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
        rot_principal_component_to_axes, _ = R.align_vectors(
            np.array([z_axis, x_axis]), np.stack([principal_component, minor_component])
        )

        pcd_points_axis_aligned = pcd_points @ rot_principal_component_to_axes.as_matrix().T
        point_axis_aligned = point @ rot_principal_component_to_axes.as_matrix().T


        # Min/max bounds for split ratio
        min_point_vals = np.min(pcd_points_axis_aligned, axis=0)
        max_point_vals = np.max(pcd_points_axis_aligned, axis=0)
        center = (max_point_vals + min_point_vals) / 2
        half_range = (max_point_vals - min_point_vals) / 2

        # Compute split ratio
        # split_ratio = 1 - np.abs((pcd_points_axis_aligned - center) / half_range)

        # Compute exponential decay split ratio.
        decay_rate = 2.0
        normalized_distance = np.abs((point_axis_aligned - center) / half_range)
        normalized_distance = np.clip(normalized_distance, 0, 1)
        split_ratio = np.exp(-decay_rate * normalized_distance) - np.exp(-decay_rate)

        # return split axes
        axes = np.stack([principal_component, secondary_component, minor_component])
        length = np.linalg.norm(max_point_vals - min_point_vals)

        if viz_split_ratio_axes:
            # Visualize the point cloud in blue, the principal axis in red, and the minor axis in green
            mean_val = np.mean(pcd_points_axis_aligned, axis=0)
            pcd_points_axis_aligned_normalized = pcd_points_axis_aligned - mean_val
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_points_axis_aligned_normalized))
            pcd.paint_uniform_color([0.7, 0.7, 0.7]) # Gray
            principle_component_line = o3d.geometry.LineSet()
            principle_component_line.points = o3d.utility.Vector3dVector(
                np.array([[0.0, 0.0, -0.3], [0.0, 0.0, 0.3], [-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
            )
            principle_component_line.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 3]]))
            principle_component_line.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

            num_samples = 10
            sampled_indices = np.random.choice(len(pcd_points), num_samples, replace=False)

            sampled_points = pcd_points_axis_aligned_normalized[sampled_indices]
            sampled_ratios = split_ratio[sampled_indices]
            colors = np.random.rand(num_samples, 3)
            
            sampled_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sampled_points))
            sampled_pcd.colors = o3d.utility.Vector3dVector(colors)

            print("Min point vals:", min_point_vals)
            print("Max point vals:", max_point_vals)
            print("Center:", center)
            
            print("Split ratios of points in order [minor, middle, major]:")
            for i, (ratio, color) in enumerate(zip(sampled_ratios, colors)):
                print(f"\033[38;2;{int(color[0]*255)};{int(color[1]*255)};{int(color[2]*255)}m● Point {i}: {ratio}\033[0m"
                      f"    ->   Coordinates: {sampled_points[i]+mean_val}"
                )
            
            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
            o3d.visualization.draw_plotly([pcd, sampled_pcd, principle_component_line, world_frame])

        return split_ratio, axes, length, center, half_range


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
            o3d.visualization.draw_plotly([pcd, principle_component_line, world_frame])

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
        grasp_type: GraspType = GraspType.PAIR,
        align_grasp_axis = [0, 0, 1], 
        align_minor_axis = [1, 0, 0], 
        split_axis=2, 
        minor_split_axis=0,
        # Note that GPD paper does 10 z translations and 7 roll rotations and nothing else
        y_min = -0.01,
        y_max = 0.01,
        num_y_samples = 3,
        x_min = -0.01,
        x_max = 0.01,
        num_x_samples = 3,
        roll_min = -np.pi / 2,
        roll_max = np.pi / 2,
        num_roll_samples = 5,
        pitch_min = -np.pi / 4,
        pitch_max = np.pi / 4,
        num_pitch_samples = 5,
        # TODO: Look into exploiting Panda gripper symmetry (grasps rotated by n*pi should be equivalent)
        yaw_min = -np.pi / 4,
        yaw_max = np.pi / 4,
        num_yaw_samples = 3,
        split_ratio_threshold = 0.5,
        point_up=False,
        is_manual=False,
        voxel_radius=0.005,
        ground_z = 0.06,
        use_extra_buffer=False,
        check_backwards=False,
        proportion_enclosed_cost_thresh: float=0.1,
        proportion_good_enclosed_cost_thresh: float=0.01,
        proportion_enclosed_good_cost_thresh: float=0.1,
        intrinsic_matrix=None,
        width_px=None, 
        height_px=None,
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
        
        merged_pcd = Concatenate([pcd, pcd_with_background])

        if is_manual:
            # Prompt user to manually indicate the grasp(s) instead of using antipodal grasping.

            manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
            
            print(
                "Manually select grasps using the GUI. The y-axis connects the two finers and "
                "the z-axis points from the hand to the fingers."
                "You need to select at least two frames for pair grasps and at least one frame "
                "otherwise. Selecting multiple frames is also okay and leads to multiple grasp "
                "candidates where the first one is attempted first."
            )

            def get_frames():
                # NOTE: This will fail with `GLFW Error: The GLFW library is not initialized` if
                # any other non-plotly o3d visualization is called between two instantiations of
                # the gui. The recommendation is to not use manual grasping with other o3d
                # visualizations.
                app = FramePlacerApp(manipuland_cloud)
                gui.Application.instance.run()
                frames = app.frames
                origins = app.frame_origins
                buffers = app.buffers
                del app
                print(f"Got {len(frames)} frames")
                if grasp_type == GraspType.PAIR and len(frames) < 2:
                    print("Need at least 2 frames for pair grasps. Retrying.")
                    return get_frames()
                elif len(frames) < 1:
                    print("Need at least 1 frame. Retrying.")
                    return get_frames()
                return frames, origins, buffers
            
            num_repeats = 1 # Increase for grasp cost debugging. Keep at 1 otherwise.
            for _ in range(num_repeats):
                X_WGs, origins, buffers = get_frames()
                if len(X_WGs) == 0:
                    continue

                # Move up to prevent collisions.
                X_WGs_collision_free = []
                for i, X_WG in enumerate(X_WGs):
                    buffer = buffers[i]
                    transformed_pcd_xyzs = merged_pcd.xyzs() - (buffer * X_WG[:3, 2])[:, np.newaxis]
                    transformed_pcd = PointCloud(transformed_pcd_xyzs.shape[1])
                    transformed_pcd.mutable_xyzs()[:] = transformed_pcd_xyzs
                    
                    distance, X_WPnew = self.find_minimum_distance(
                        transformed_pcd, 
                        RigidTransform(X_WG), 
                        # min_range=-0.11 - buffer,
                        # max_range=-0.01 - buffer,
                        use_extra_buffer=use_extra_buffer,
                        viz=False
                    )
                    if np.isnan(distance):
                        print(f"Frame {i} in collision")
                        continue
                    X_WGs_collision_free.append(X_WPnew.GetAsMatrix4())

                    debug = False
                    if debug:
                        # Calculate scores (for debugging)
                        pcd_points = pcd.xyzs().T
                        split_ratio, split_axes, length, center, half_range = self.compute_pcd_split_ratio_at_point(pcd_points, origins[i])
                        is_nonempty, within_box_pt_normals, proportion_enclosed, _ = self.check_nonempty(pcd, X_WPnew)
                        cost, cost_dict = self.compute_costs(
                                        X_WPnew, 
                                        within_box_pt_normals,
                                        split_axes,
                                        center,
                                        half_range,
                                        ground_z,
                                        np.max(pcd_points[:, 2]) - np.min(pcd_points[:, 2]),
                                        pcd_points.shape[0]
                                    )
                        manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
                        manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

                        gripper_xyzs = self.hand_collision_model.to_pcd()
                        # RollPitchYaw(np.pi/2, 0, np.pi/2) is world to wsg specific transform.
                        # WSG y axis needs to be aligned with world -z.
                        gripper_cloud = o3d.geometry.PointCloud(
                            o3d.utility.Vector3dVector(gripper_xyzs)
                        ).voxel_down_sample(0.005).transform(
                            (X_WPnew @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0])
                        ).GetAsMatrix4())
                        gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])

                        viz_geoms = [manipuland_cloud, gripper_cloud]
                        o3d.visualization.draw_plotly(viz_geoms)
                if input == "C":
                    break
            
            if grasp_type == GraspType.PAIR:
                self.grasp_candidates: List[Tuple[np.ndarray]] = [
                    (X_WGs_collision_free[i], X_WGs_collision_free[i+1]) for i in range(len(X_WGs_collision_free)-1)
                ]
            else:
                self.grasp_candidates: List[np.ndarray] = X_WGs_collision_free
            self.sorted_costs = [0.0 for _ in range(len(self.grasp_candidates))]
            return


        # # NOTE: All num_samples should be odd numbers
        # y_min = -0.01
        # y_max = 0.01
        # num_y_samples = 3
        # roll_min = -np.pi / 2
        # roll_max = np.pi / 2
        # num_roll_samples = 7
        # pitch_min = -np.pi / 4
        # pitch_max = np.pi / 4
        # num_pitch_samples = 5
        # # TODO: Look into exploiting Panda gripper symmetry (grasps rotated by n*pi should be equivalent)
        # yaw_min = -np.pi / 2
        # yaw_max = np.pi / 2
        # num_yaw_samples = 7

        PARALLEL = False # There are bugs in the parallel implementation => Don't use!
        VISUALIZE_CLUSTERS = False
        VISUALIZE_FILTERED_CLOUDS = False
        VIZUALIZE_SPLIT_RATIOS_AT_ORIGIN = False
        VISUALIZE = False
        VISUALIZE_EACH = False
        VISUALIZE_ALL = False # Heat map of good to bad grasps but too messy for fine detail
        VISUALIZE_ORIG = False
        VISUALIZE_SORTED_WITH_COSTS = False

        np.random.seed(random_seed)

        downsampled_pcd = pcd.VoxelizedDownSample(voxel_size=0.02)
        downsampled_pcd_points = downsampled_pcd.xyzs().T
        pcd_points = pcd.xyzs().T

        flattened_cloud_xyzs = np.copy(pcd.xyzs())
        flattened_cloud_xyzs[2,:] = np.clip(
            flattened_cloud_xyzs[2,:],
            np.max(flattened_cloud_xyzs[2,:]) - (np.max(flattened_cloud_xyzs[2,:]) - np.min(flattened_cloud_xyzs[2,:])) / 10.0,
            np.max(flattened_cloud_xyzs[2,:])
        )
        flattened_cloud = PointCloud(flattened_cloud_xyzs.shape[1])
        flattened_cloud.mutable_xyzs()[:] = flattened_cloud_xyzs
        flattened_cloud = flattened_cloud.VoxelizedDownSample(voxel_size=0.005)
        flattened_cloud.EstimateNormals(radius=0.1, num_closest=30)
        flattened_cloud.FlipNormalsTowardPoint([-0.0338161, 0.62563, 0.360087])
        pcd_flattened_points = flattened_cloud.xyzs().T

        # Filter pcd based on split ratio
        if grasp_type == GraspType.TOP:
            split_ratios, all_split_axes, lengths, all_centers, all_half_ranges = self.compute_pcd_split_ratio_cropped(
                pcd_flattened_points, 
                viz=VISUALIZE_CLUSTERS
            )
            length = np.max(lengths)

            # Allow points where at least 2 out of 3 split ratios exceed the threshold
            mask = np.sum(split_ratios > split_ratio_threshold, axis=1) >= 2
            split_ratio_filtered_points = pcd_flattened_points[mask]

            # make sure we filter out all information we use about the filtered points
            split_ratios = split_ratios[mask]
            all_split_axes = all_split_axes[mask]
            all_centers = all_centers[mask]
            all_half_ranges = all_half_ranges[mask]

            split_ratio_filtered_normals = flattened_cloud.normals()[:, mask].T
        else:
            split_ratios, split_axes, length, center, half_range = self.compute_pcd_split_ratio(downsampled_pcd_points)
            all_split_axes = np.tile(split_axes, (split_ratios.shape[0], 1, 1))
            all_centers = np.tile(center, (split_ratios.shape[0], 1, 1))
            all_half_ranges = np.tile(half_range, (split_ratios.shape[0], 1, 1))

            # Allow points where at least 2 out of 3 split ratios exceed the threshold
            mask = np.sum(split_ratios > split_ratio_threshold, axis=1) >= 2
            split_ratio_filtered_points = downsampled_pcd_points[mask]

            # make sure we filter out all information we use about the filtered points
            split_ratios = split_ratios[mask]
            all_split_axes = all_split_axes[mask]
            all_centers = all_centers[mask]
            all_half_ranges = all_half_ranges[mask]

            split_ratio_filtered_normals = downsampled_pcd.normals()[:, mask].T

        if VISUALIZE_FILTERED_CLOUDS:
            manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(downsampled_pcd_points))
            manipuland_cloud.paint_uniform_color([0.7, 0.7, 0.7])
            filtered_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(split_ratio_filtered_points))
            filtered_cloud.paint_uniform_color([1,0,0])

            o3d.visualization.draw_geometries([
                manipuland_cloud, filtered_cloud
            ], window_name="points after split ratio filtering")

        # Kdtree based on unfiltered points for better normal queries
        if grasp_type == GraspType.TOP: 
            kdtree = KDTree(pcd_flattened_points)
        else:
            kdtree = KDTree(downsampled_pcd_points)

        # Sample random points to compute darboux frames for
        num_split_ratio_filtered_points = len(split_ratio_filtered_points)
        print("num_split_ratio_filtered_points", num_split_ratio_filtered_points)

        if grasp_type != GraspType.TOP:
            # filter out pts whose x is past 3/4 of obj
            x_center = (np.min(pcd_points, axis=0)[0] + np.max(pcd_points, axis=0)[0]) / 2
            x_width = np.max(pcd_points, axis=0)[0] - np.min(pcd_points, axis=0)[0]
            x_thresh = x_center + x_width / 4
            mask = split_ratio_filtered_points[:, 0] < x_thresh
            filtered_points = split_ratio_filtered_points[mask, :]
            filtered_normals = split_ratio_filtered_normals[mask, :]

            # make sure we filter out all information we use about the filtered points
            split_ratios = split_ratios[mask]
            all_split_axes = all_split_axes[mask]
            all_centers = all_centers[mask]
            all_half_ranges = all_half_ranges[mask]

            if VISUALIZE_FILTERED_CLOUDS:
                manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(downsampled_pcd_points))
                manipuland_cloud.paint_uniform_color([0.7, 0.7, 0.7])
                filtered_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(filtered_points))
                filtered_cloud.paint_uniform_color([1,0,0])

                o3d.visualization.draw_geometries([
                    manipuland_cloud, filtered_cloud
                ], window_name="points after x filtering")

            num_filtered_points = len(filtered_points)
            print("num_x_filtered_points", num_filtered_points)
        else:
            filtered_points = split_ratio_filtered_points
            filtered_normals = split_ratio_filtered_normals
            num_filtered_points = num_split_ratio_filtered_points

        darboux_frame_sample_indices = np.random.choice(
            np.arange(num_filtered_points),
            int(min(num_samples, num_filtered_points)),
            replace=False,
        )

        print("len sample indicies", darboux_frame_sample_indices)

        if VISUALIZE_FILTERED_CLOUDS:
            manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
            manipuland_cloud.paint_uniform_color([0.7, 0.7, 0.7])
            filtered_cloud = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(filtered_points[darboux_frame_sample_indices]))
            filtered_cloud.paint_uniform_color([1,0,0])

            o3d.visualization.draw_geometries([
                manipuland_cloud, filtered_cloud
            ], window_name="sampled points for grasp computation")

        # Compute darboux frames at samples
        X_WPs = self.compute_darboux_frames(
            points=filtered_points[darboux_frame_sample_indices],
            normals=filtered_normals[darboux_frame_sample_indices],
            pcd=downsampled_pcd,
            kdtree=kdtree,
            point_up=point_up
        )

        print("Num X_WPs", len(X_WPs))

        manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
        manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])
        viz_geoms = [manipuland_cloud]

        if VISUALIZE_ORIG:
            from pydrake.all import StartMeshcat
            from manipulation.meshcat_utils import AddMeshcatTriad
            meshcat = StartMeshcat()
            meshcat.SetObject("cloud", pcd)
            for i, X_WP in enumerate(X_WPs):
                AddMeshcatTriad(meshcat, f"frame{i}", length=0.025, radius=0.001, X_PT=X_WP)

            geoms = [manipuland_cloud]
            for X_WP in X_WPs:
                color = np.random.rand(3)
                color /= np.linalg.norm(color)
                color = tuple(color)
                geoms.append(self.make_gripper_line_set(X_WP.GetAsMatrix4(), color))
            o3d.visualization.draw_plotly(geoms)

            for X_WP in X_WPs:
                gripper_xyzs = self.hand_collision_model.to_pcd()
                # RollPitchYaw(np.pi/2, 0, np.pi/2) is world to wsg specific transform.
                # WSG y axis needs to be aligned with world -z.
                gripper_cloud = o3d.geometry.PointCloud(
                    o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(0.005).transform(
                        (X_WP @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0])).GetAsMatrix4())
                gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])

                o3d.visualization.draw_plotly([
                    manipuland_cloud,
                    gripper_cloud,
                    self.make_gripper_line_set(X_WP.GetAsMatrix4(), color),
                    self.make_triad_line_set(X_WP.GetAsMatrix4(), [0.0, 1.0, 0.0])
                ])

        start_time = time.time()
        if not PARALLEL:
            # Local grid search around darboux frames
            candidate_lst: List[np.ndarray] = []
            candidate_lst_by_grasp_origin_pt: List[np.ndarray] = []
            candidate_costs: List[float] = []
            candidiate_cost_dicts: list[dict] = []
            for X_WP, split_ratio, split_axes, center, half_range in zip(
                X_WPs, 
                split_ratios[darboux_frame_sample_indices], 
                all_split_axes[darboux_frame_sample_indices], 
                all_centers[darboux_frame_sample_indices], 
                all_half_ranges[darboux_frame_sample_indices],
            ):
                if VIZUALIZE_SPLIT_RATIOS_AT_ORIGIN:
                    # Rotate point cloud to align the principal component with the z-axis and the minor component with the x-axis
                    z_axis, x_axis = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
                    rot_principal_component_to_axes, _ = R.align_vectors(
                        np.array([z_axis, x_axis]), np.stack([split_axes[0], split_axes[2]])
                    )
                    viz_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_points))
                    viz_pcd.paint_uniform_color([0.7, 0.7, 0.7]) # Gray

                    rot = RotationMatrix(rot_principal_component_to_axes.as_matrix().T)
                    T = np.eye(4)
                    T[:3,:3] = rot.matrix()
                    T[:3, 3] = X_WP.translation()
                    pca = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    pca.transform(T)
                    
                    print(X_WP.translation())   
                    print(split_axes)
                    o3d.visualization.draw_geometries([viz_pcd, pca], window_name="split_ratio_at_origin")

                color = np.random.rand(3)
                color /= np.linalg.norm(color)
                color = tuple(color)# NOTE: The best variations to sample/ search over is situation/ grasp environment dependent (e.g. bin vs table)
                X_WPnew_best = None
                best_cost = np.inf
                best_cost_dict = None
                for x in np.linspace(x_min, x_max, num_x_samples):
                    for y in np.linspace(y_min, y_max, num_y_samples):
                        for pitch in np.linspace(pitch_min, pitch_max, num_pitch_samples):
                            for roll in np.linspace(roll_min, roll_max, num_roll_samples):
                                for yaw in np.linspace(yaw_min, yaw_max, num_yaw_samples):
                                    # TODO: Explore whether it is faster to do this transform in numpy
                                    X_PPnew = RigidTransform(RollPitchYaw(roll, pitch, yaw), np.array([x, y, 0]))
                                    X_WPnew = X_WP.multiply(X_PPnew)
                                    if X_WPnew.GetAsMatrix4()[0, 2] < 0.:
                                        # avoid grasps from far side, reach around not feasible
                                        continue 
                                    if X_WPnew.GetAsMatrix4()[0, 2] > 0.94: # 40 degree cone
                                        # avoid grasps from too straight forward, not feasible
                                        continue 

                                    if VISUALIZE:
                                        # visualize
                                        manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
                                        manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

                                        gripper_xyzs = self.hand_collision_model.to_pcd()
                                        # RollPitchYaw(np.pi/2, 0, np.pi/2) is world to wsg specific transform.
                                        # WSG y axis needs to be aligned with world -z.
                                        gripper_cloud = o3d.geometry.PointCloud(
                                            o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(0.005).transform(
                                                (X_WPnew @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0])).GetAsMatrix4())
                                        gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])

                                        o3d.visualization.draw_plotly([
                                            manipuland_cloud,
                                            gripper_cloud,
                                            self.make_triad_line_set(X_WPnew.GetAsMatrix4(), [0.0, 1.0, 0.0])
                                        ])
                                        print("darboux frame", X_WPnew)

                    
                                    R_WPnew = X_WPnew.GetAsMatrix4()[:3, :3]
                                    eff_vertical_vec = R_WPnew.dot(np.array([0, 0, 1])) # don't want it to face up
                                    if eff_vertical_vec[2] > 0:
                                        continue
                                    
                                    # Compute a new transform that minimizes z-direction distance without penetration
                                    # Use pcd with floor to avoid gripper smashing into the table
                                    if point_up:
                                        # Need to increase sampling range
                                        distance, X_WPnew = self.find_minimum_distance(
                                            merged_pcd,
                                            X_WPnew,
                                            # min_range=-0.1,
                                            # max_range=0.4,
                                            # num_samples=100,
                                            use_extra_buffer=use_extra_buffer,
                                        )
                                    else:
                                        distance, X_WPnew = self.find_minimum_distance(
                                            merged_pcd, 
                                            X_WPnew, 
                                            use_extra_buffer=use_extra_buffer,
                                            backwards=check_backwards
                                        )
                                    # If distance cannot be found, go over to the next iteration
                                    if np.isnan(distance):
                                        continue

                                    # If the candidate has no collisions and the closing region is non
                                    # empty, then append it to the list of candidates.
                                    is_nonempty, within_box_pt_normals, proportion_enclosed, indices_enclosed = self.check_nonempty(pcd, X_WPnew)
                                    if is_nonempty:
                                        if grasp_type == GraspType.PAIR:
                                            cost, cost_dict = self.compute_costs(
                                                    X_WPnew, 
                                                    within_box_pt_normals,
                                                    split_axes,
                                                    center,
                                                    half_range,
                                                    ground_z,
                                                    np.max(pcd_points[:, 2]) - np.min(pcd_points[:, 2]),
                                                    pcd_points.shape[0],
                                                    proportion_enclosed_cost_thresh,
                                                    proportion_good_enclosed_cost_thresh,
                                                    proportion_enclosed_good_cost_thresh,
                                                )
                                        elif grasp_type == GraspType.SIDE:
                                            cost, cost_dict = self.compute_costs_single(
                                                    X_WPnew, 
                                                    within_box_pt_normals, 
                                                    proportion_enclosed,
                                                    split_ratio, 
                                                    split_axis, 
                                                    minor_split_axis, 
                                                    align_grasp_axis, 
                                                    align_minor_axis
                                                )
                                        elif grasp_type == GraspType.TOP:
                                            _, _, proportion_enclosed, _ = self.check_nonempty(flattened_cloud, X_WPnew)
                                            cost, cost_dict = self.compute_costs_top(
                                                    X_WPnew, 
                                                    within_box_pt_normals, 
                                                    split_ratio,
                                                    split_axes,
                                                    proportion_enclosed,
                                                    pcd=flattened_cloud
                                                )
                                        elif grasp_type == GraspType.STABLE:
                                            cost, cost_dict = self.compute_costs_stable(
                                                    X_WPnew, 
                                                    within_box_pt_normals, 
                                                    split_ratio,
                                                    proportion_enclosed,
                                                    pcd_points.shape[0]
                                                )
                                        elif grasp_type == GraspType.ADDITIONAL:
                                            if intrinsic_matrix is None or width_px is None or height_px is None:
                                                raise ValueError("Intrinsic matrix and image dimensions must be provided for GraspType.ADDITIONAL")
                                            cost, cost_dict = self.compute_costs_voxel_map_coverage(
                                                    X_WPnew, 
                                                    intrinsic_matrix,
                                                    width_px, 
                                                    height_px,
                                                    indices_enclosed,
                                                )
                                        
                                        if cost < best_cost:
                                            best_cost = cost
                                            best_cost_dict = cost_dict
                                            X_WPnew_best = X_WPnew
                if X_WPnew_best is not None:
                    candidate_costs.append(
                        best_cost
                    )
                    candidiate_cost_dicts.append(best_cost_dict)
                    candidate_lst.append(X_WPnew_best.GetAsMatrix4())
                    candidate_lst_by_grasp_origin_pt.append(X_WP.GetAsMatrix4())
                    viz_geoms.append(self.make_gripper_line_set(X_WPnew_best.GetAsMatrix4(), color))

                    if VISUALIZE_EACH:
                        print(candidate_costs[-1])
                        print(candidiate_cost_dicts[-1])
                        manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
                        manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])
                        manipuland_top_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(flattened_cloud.xyzs().T))
                        manipuland_top_cloud.paint_uniform_color([0.0, 1.0, 0.0])

                        gripper_xyzs = self.hand_collision_model.to_pcd()
                        # RollPitchYaw(np.pi/2, 0, np.pi/2) is world to wsg specific transform.
                        # WSG y axis needs to be aligned with world -z.
                        gripper_cloud = o3d.geometry.PointCloud(
                            o3d.utility.Vector3dVector(gripper_xyzs)
                        ).voxel_down_sample(0.005).transform(
                            (X_WPnew_best @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0])
                        ).GetAsMatrix4())
                        gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])


                        # visualize axes, principal axis is z axis (blue), minor axis is x axis (red)
                        z_axis, x_axis = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
                        rot_principal_component_to_axes, _ = R.align_vectors(
                            np.array([z_axis, x_axis]), np.stack([split_axes[0], split_axes[2]])
                        )
                        rot = RotationMatrix(rot_principal_component_to_axes.as_matrix().T)

                        T = np.eye(4)
                        T[:3,:3] = rot.matrix()
                        T[:3, 3] = X_WP.translation()
                        pca = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                        pca.transform(T)

                        viz_geoms = [manipuland_cloud, manipuland_top_cloud, gripper_cloud, pca]
                        o3d.visualization.draw_geometries(viz_geoms)

            print("sequential antipodal grasp time: {:.3f}".format(time.time() - start_time))
            if VISUALIZE_ALL:
                o3d.visualization.draw_plotly(viz_geoms)

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
                import matplotlib.pyplot as plt
                sorted_indices = np.argsort(candidate_costs)
                sorted_candidates = [candidate_lst[i] for i in sorted_indices]
                sorted_costs = [candidate_costs[i] for i in sorted_indices]

                # Normalize costs to [0,1] range for interpolation
                min_cost = min(sorted_costs)
                max_cost = max(sorted_costs)
                cost_range = max_cost - min_cost
                normalized_costs = [(c - min_cost) / cost_range if cost_range > 0 else 0.5 for c in sorted_costs]

                viz_geoms = [manipuland_cloud]
                for X_G, norm_cost in zip(sorted_candidates, normalized_costs):
                    color = plt.cm.jet(norm_cost)[:3]  # Use jet colormap (RGB values)
                    viz_geoms.append(self.make_gripper_line_set(X_G, color))
                o3d.visualization.draw_plotly(viz_geoms, window_name="all grasps (red is high cost, blue is low cost)")

        start = time.time()

        if grasp_type == GraspType.PAIR:
            # Two grasp selection
            candidate_lst = np.array(candidate_lst)
            candidate_lst_by_grasp_origin_pt = np.array(candidate_lst_by_grasp_origin_pt)
            candidate_costs = np.array(candidate_costs)
            sorted_candidate_inds = np.argsort(candidate_costs)[:len(candidate_costs)]
            
            candidates_filtered = candidate_lst[sorted_candidate_inds]
            candidates_grasp_origin_filtered = candidate_lst_by_grasp_origin_pt[sorted_candidate_inds]
            candidate_costs_filtered = candidate_costs[sorted_candidate_inds]
            candidate_cost_dicts_filtered = [candidiate_cost_dicts[i] for i in sorted_candidate_inds]
        
            # Extract translations from grasp origin pts (last column of each 4x4 matrix)
            translations = candidates_grasp_origin_filtered[:, :3, 3]
            
            # Compute pairwise translation differences
            translation_diffs = np.linalg.norm(translations[:, np.newaxis] - translations[np.newaxis, :], axis=-1)
            translation_cost = -np.exp(translation_diffs / length)
            
            # Extract rotations (top-left 3x3 part of each 4x4 matrix)
            rotations = candidates_filtered[:, :3, :3]
            print(rotations.shape)
            z_axes = rotations[:, :, 2]
            print(z_axes.shape)
            
            # Compute pairwise rotational differences
            # R_relative = R2^T @ R1 (for each pair of rotations R1, R2)
            relative_rotations = np.einsum('nij,mjk->nmik', rotations, rotations.transpose(0, 2, 1))
            
            # Compute the angle of rotation from the relative rotation matrices
            # Trace(R_relative) = 1 + 2*cos(theta), where theta is the angle of rotation
            trace_relative_rotations = np.einsum('...ii', relative_rotations)
            rotation_diffs = np.arccos(np.clip((trace_relative_rotations - 1) / 2, -1.0, 1.0))  # Avoid precision errors
            rotation_cost = np.exp(np.abs(np.pi / 2 - rotation_diffs))
            # rotation_cost = np.maximum(np.exp(-rotation_diffs/np.pi), np.exp(-(np.pi - rotation_diffs)/np.pi))

            z_axis_dot_product = z_axes @ z_axes.T
            z_axis_diffs = np.clip(z_axis_dot_product, -1.0, 1.0)
            z_rotation_cost = (np.exp(np.abs(z_axis_diffs)) - 1) / (np.exp(1) - 1)

            grasps_quality = candidate_costs_filtered[:, np.newaxis] + candidate_costs_filtered[np.newaxis, :]
            best_quality = np.abs(np.min(grasps_quality))
            print("z cost scaling", 0.2 * best_quality)

            pair_cost_dicts = []
            N = len(candidates_filtered)
            for i in range(N):
                for j in range(N):
                    pair_cost_dicts.append({
                        "grasp1_breakdown": candidate_cost_dicts_filtered[i],
                        "grasp2_breakdown": candidate_cost_dicts_filtered[j],
                        "grasp1_quality": candidate_costs_filtered[i],
                        "grasp2_quality": candidate_costs_filtered[j],
                        "grasps_quality": grasps_quality[i, j],
                        "translation_cost": 0.1 * best_quality * translation_cost[i, j],
                        "rotation_cost": 0.2 * best_quality * rotation_cost[i, j],
                        "z_rotation_cost": 0.5 * best_quality * z_rotation_cost[i, j]
                    })

            pair_costs = np.array([
                d["grasps_quality"] + d["translation_cost"] + d["z_rotation_cost"]
                for d in pair_cost_dicts
            ]) # Shape (NxN,)

            pair_costs = pair_costs.reshape(N, N)
            pair_costs = np.triu(pair_costs, k=1) + np.tril(np.inf * np.ones_like(pair_costs)) # make lower + diagonal infinity to avoid double counting
            pair_costs = pair_costs.flatten()
            pairs = [(X_WG1, X_WG2) for X_WG1 in candidates_filtered for X_WG2 in candidates_filtered]

            sorted_pair_indices = np.argsort(pair_costs)
            pair_lst_sorted = [pairs[idx] for idx in sorted_pair_indices]

            print("pair selection time:", time.time()-start)
            # List of grasp pairs
            self.grasp_candidates: List[Tuple[np.ndarray]] = pair_lst_sorted
            self.sorted_costs = [pair_costs[i] for i in sorted_pair_indices]

            if VISUALIZE_SORTED_WITH_COSTS:
                sorted_costs = [pair_costs[i] for i in sorted_pair_indices]
                sorted_cost_dicts = [pair_cost_dicts[i] for i in sorted_pair_indices]

                background_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_with_background.xyzs().T))
                background_cloud.paint_uniform_color([1.0, 0.0, 1.0])
                manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
                manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

                for X_WGs, cost, cost_dict in zip(pair_lst_sorted, sorted_costs, sorted_cost_dicts):
                    print("Cost:", cost)
                    print("Cost dict:\n", cost_dict)

                    X_WG1, X_WG2 = X_WGs[0], X_WGs[1]

                    gripper1_xyzs = self.hand_collision_model.to_pcd()
                    gripper1_cloud = o3d.geometry.PointCloud(
                        o3d.utility.Vector3dVector(gripper1_xyzs)).voxel_down_sample(0.005).transform(
                            (X_WG1 @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4()))
                    gripper1_cloud.paint_uniform_color([1.0, 0.0, 0.0])

                    gripper2_xyzs = self.hand_collision_model.to_pcd()
                    gripper2_cloud = o3d.geometry.PointCloud(
                        o3d.utility.Vector3dVector(gripper2_xyzs)).voxel_down_sample(0.005).transform(
                            (X_WG2 @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4()))
                    gripper2_cloud.paint_uniform_color([0.0, 1.0, 0.0])

                    mean_gripper_point = np.mean(np.concatenate([gripper1_cloud.points, gripper2_cloud.points]), axis=0)
                    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.05, origin=[mean_gripper_point[0], mean_gripper_point[1], 0])
                    
                    grasp_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05).transform(
                        X_WG1
                    )
                    grasp_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05).transform(
                        X_WG2
                    )

                    viz_geoms = [
                        background_cloud, manipuland_cloud, gripper1_cloud, gripper2_cloud, world_frame, grasp_frame1, grasp_frame2
                    ]
                    o3d.visualization.draw_geometries(viz_geoms)
        else:
            if grasp_type == GraspType.STABLE:
                candidate_lst_temp = []
                for X_WG in candidate_lst:
                    X_GW = (RigidTransform(X_WG) @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0])).inverse()
                    pcd_G_np = X_GW.multiply(pcd.xyzs())

                    # get points in closing region
                    crop_min = [-0.0125, -0.053, 0.03]
                    crop_max = [0.0125, 0.053, 0.03+0.14] # finray is ~14cm long

                    # Check if there are any points within the cropped region.
                    pcd_enclosed_check = RigidTransform(X_WG).inverse().multiply(pcd.xyzs())
                    mask = (
                        (crop_min[0] <= pcd_enclosed_check[0, :])
                        * (pcd_enclosed_check[0, :] <= crop_max[0])
                        * (crop_min[1] <= pcd_enclosed_check[1, :])
                        * (pcd_enclosed_check[1, :] <= crop_max[1])
                        * (crop_min[2] <= pcd_enclosed_check[2, :])
                        * (pcd_enclosed_check[2, :] <= crop_max[2])
                    )
                    indices = np.nonzero(mask)[0]
                    pcd_enclosed = pcd_G_np[:, indices]

                    manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_G_np.T))
                    manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])
                    enclosed_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_enclosed.T))
                    enclosed_cloud.paint_uniform_color([0.0, 1.0, 0.0])

                    gripper_xyzs = self.hand_collision_model.to_pcd()
                    # RollPitchYaw(np.pi/2, 0, np.pi/2) is world to wsg specific transform.
                    # WSG y axis needs to be aligned with world -z.
                    gripper_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(0.005)
                    gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])
                    viz_geoms = [manipuland_cloud, enclosed_cloud, gripper_cloud]

                    x_mid = (np.max(pcd_enclosed[0, :]) + np.min(pcd_enclosed[0, :])) / 2
                    X_WG_new = np.copy(X_WG)
                    X_WG_new[:3, 3] += x_mid * X_WG[1, :3]
                    candidate_lst_temp.append(X_WG_new)
                candidate_lst = candidate_lst_temp

            sorted_indices = np.argsort(candidate_costs)
            candidate_lst_sorted = [candidate_lst[idx] for idx in sorted_indices]

            self.grasp_candidates: List[np.ndarray] = candidate_lst_sorted
            self.sorted_costs = [candidate_costs[i] for i in sorted_indices]

            if VISUALIZE_SORTED_WITH_COSTS:
                manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.xyzs().T))
                manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

                sorted_costs = [candidate_costs[i] for i in sorted_indices]
                sorted_cost_dicts = [candidiate_cost_dicts[i] for i in sorted_indices]
                for X_WG, cost, cost_dict in zip(candidate_lst_sorted, sorted_costs, sorted_cost_dicts):
                    print("Cost:", cost)
                    print("Cost dict:\n", cost_dict)

                    gripper_xyzs = self.hand_collision_model.to_pcd()
                    # RollPitchYaw(np.pi/2, 0, np.pi/2) is world to wsg specific transform.
                    # WSG y axis needs to be aligned with world -z.
                    gripper_cloud = o3d.geometry.PointCloud(
                        o3d.utility.Vector3dVector(gripper_xyzs)
                    ).voxel_down_sample(0.005).transform(
                        (RigidTransform(X_WG) @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0])
                    ).GetAsMatrix4())
                    gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])

                    mean_gripper_point = np.mean(gripper_cloud.points, axis=0)
                    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.05, origin=[mean_gripper_point[0], mean_gripper_point[1], 0])
                    
                    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05).transform(
                        X_WG
                    )

                    viz_geoms = [manipuland_cloud, gripper_cloud, world_frame, grasp_frame]
                    o3d.visualization.draw_geometries(viz_geoms)

    def get_best_grasps(self, candidate_num=-1) -> List[Tuple[np.ndarray]]:
        """Returns a list of the `candidate_num` grasps with the lowest cost."""
        if candidate_num == -1:
            return self.grasp_candidates, self.sorted_costs
        return self.grasp_candidates[:candidate_num], self.sorted_costs[:candidate_num]