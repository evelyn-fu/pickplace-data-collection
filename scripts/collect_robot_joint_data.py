"""
This script collects joint position and torque data at multiple gripper openings. The
data is collected multiple times at each gripper opening.

The output is saved to a directory with the gripper position as the subdirectory name
with the following format:

<save_data_path>/gripper_position_<gripper_position>/
    run_<run_idx>/
        joint_positions.npy
        joint_torques.npy
        sample_times_s.npy
"""

import argparse
import logging
import os

from pathlib import Path

import numpy as np
import sys
PYCUCI_ROOT = os.path.dirname(__file__) + "/../../" + "cuciv0" 
sys.path.append(PYCUCI_ROOT+'/bazel-bin/cuci/src/pybind/pycuci')
import pycuci as cci

from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    DiagramBuilder,
    PiecewisePolynomial,
    Simulator,
    TrajectorySource,
    VectorLogSink,
    StartMeshcat,
    LeafSystem,CompositeTrajectory,PathParameterizedTrajectory,
    RotationMatrix,
    RollPitchYaw,
    RigidTransform,
    Concatenate,
    DepthImageToPointCloud,
    CameraInfo,
    HPolyhedron,
    AbstractValue,
    PointCloud
)
import open3d as o3d
import copy
from tqdm import tqdm
import shutil
from perception.camera_in_world import CameraPoseInWorldSource
from planning.grasp import GraspListener, GraspType
from planning.inverse_kinematics import solve_via_analytic_IK
from planning.trajectories import (
    MakePickGripperFrames, 
    MakeGripperCommandTrajectory, 
    MakeGripperPoseTrajectory, 
    TrajType
)
from planning.toppra import reparameterize_with_toppra
from robot_payload_id.control.trajectory import FourierSeriesTrajectory
from robot_payload_id.utils import FourierSeriesTrajectoryAttributes
from manipulation.station import MakeHardwareStation, RobotDiagram
from mmt_gcs.planning.mintime_scs import MintimeSCSWithPathFixing
from mmt_gcs.planning.corridor_planning_utils import CCICollisionChecker
from mmt_gcs.planning.region_generation import CCI_inflate_edges_given_pwl_path

PETE_ASSETS = os.path.dirname(__file__) + "/../pete_assets/"
MMT_GCS_ROOT = os.path.abspath(os.path.join(__file__, "../../../mmt_gcs/"))
ONLINE_VOXEL_RADIUS = 0.005
SYS_ID_TRAJ_PARAMETER_PATH = Path(os.path.abspath(os.path.join(__file__ ,"../../traj_feb8")))
platform_height = 0.065
pregrasp_dist = 0.17
eef_to_gripper_length = 0.16
opened = np.array([0.107])
closed = np.array([0.00])

def generate_points_in_cube(center, side_lengths, spacing=0.02):
    # Extract the center and side lengths
    x, y, z = center
    x_side, y_side, z_side = side_lengths
    
    # Generate the grid of points along each axis
    x_points = np.linspace(x - x_side/2, x + x_side/2, int(x_side / spacing) + 1)
    y_points = np.linspace(y - y_side/2, y + y_side/2, int(y_side / spacing) + 1)
    z_points = np.linspace(z - z_side/2, z + z_side/2, int(z_side / spacing) + 1)
    
    # Create the 3D grid of points
    grid_x, grid_y, grid_z = np.meshgrid(x_points, y_points, z_points)
    
    # Reshape the grid into a Nx3 array
    points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    
    return points
    
ceiling_vox = generate_points_in_cube((0.4, 0.0, 0.9), (0.4, 0.5, 0.001))
camera_vox = generate_points_in_cube((0.8, 0.0, 0.3), (0.0, 0.15, 0.6))
bin_cam_vox = generate_points_in_cube((-0.0338161, 0.84, 0.2), (0.02, 0.10, 0.4))
bin_cam_pole_vox = generate_points_in_cube((-0.07, 0.44, 0.21), (0.04, 0.08, 0.42))


def get_cci_edge_inflator(verbose=False):
    cci_parser = cci.URDFParser()
    cci_parser.register_package("adaptive_decomp", PETE_ASSETS + "assets")
    cci_parser.register_package("iiwa_description", PETE_ASSETS + "assets/iiwa")
    cci_parser.register_package(
        "wsg_description", PETE_ASSETS + "assets/wsg_description"
    )
    cci_parser.register_package(
        "tri_finray_gripper", PETE_ASSETS + "assets/tri_finray_gripper"
    )
    cci_parser.parse_directives(PETE_ASSETS + "assets/directives/iiwa7_on_table.yaml")
    cci_plant = cci_parser.build_plant()
    cci_mplant = cci_plant.getMinimalPlant()
    cci_domain = cci.HPolyhedron()
    cci_domain.MakeBox(
        cci_plant.getPositionLowerLimits(), cci_plant.getPositionUpperLimits()
    )
    cci_objects = {
        "cci_plant": cci_plant,
        "cci_mplant": cci_mplant,
        "cci_domain": cci_domain,
    }

    cci_fei_opts = cci.FastEdgeInflationOptions()
    cci_fei_opts.num_particles = 10000
    cci_fei_opts.max_hyperplanes_per_iteration = 20
    cci_fei_opts.epsilon = 0.005
    cci_fei_opts.delta = 0.005
    cci_fei_opts.max_iterations = 30
    cci_fei_opts.mixing_steps = 60
    cci_fei_opts.configurataon_margin = 0.01
    cci_fei_opts.verbose = verbose

    edge_inflator = cci.CudaEdgeInflator(
        cci_objects["cci_mplant"],
        cci_objects["cci_plant"].getRobotGeometryIds(),
        cci_fei_opts,
        cci_objects["cci_domain"],
    )

    return edge_inflator, cci_objects


def get_drm_planner(cci_obj, vox=None):
    drm_pl_opts = cci.DrmPlannerOptions()
    drm_pl_opts.max_number_planning_attempts = 50
    drm_pl_opts.try_shortcutting = True
    drm_pl_opts.online_edge_step_size = 0.005

    drm_planner = cci.DrmPlanner(cci_obj["cci_plant"], drm_pl_opts)
    drm_planner.LoadRoadmap(
        MMT_GCS_ROOT
        + "/tmp/iiwa_hardware/iiwa_roadmap_1_0_0.01_0.2_50000_10_4.5_0.45.rm"
    )

    if vox is not None:
        online_voxel_observation = cci.Voxels(vox.T)
        drm_planner.BuildCollisionSet(online_voxel_observation)

    return drm_planner


def scs_trajopt(
    start, goal, drm_planner, cci_obj, edge_inflator, vox, vel_limits, acc_limits
):
    online_voxel_observation = cci.Voxels(vox) if vox is not None else cci.Voxels()

    success, pwl_plan = drm_planner.Plan(
        start, goal, online_voxel_observation, ONLINE_VOXEL_RADIUS
    )

    regions, edges = CCI_inflate_edges_given_pwl_path(
        pwl_plan,
        edge_inflator,
        online_voxel_observation,
        ONLINE_VOXEL_RADIUS,
        verbose=True,
    )

    cci_checker = CCICollisionChecker(
        cci_obj["cci_mplant"],
        cci_obj["cci_plant"].getRobotGeometryIds(),
        online_voxel_observation,
        ONLINE_VOXEL_RADIUS,
    )

    vel_limits_reflected = [-vel_limits, vel_limits]
    acc_limits_reflected = [-acc_limits, acc_limits]
    traj, cost, timing_info, traj_col_free, first_solve_collision_free, collisions = (
        MintimeSCSWithPathFixing(
            start,
            goal,
            regions,
            edges,
            vel_limits_reflected,
            acc_limits_reflected,
            cci_checker,
            edge_inflator,
            online_voxel_observation,
            ONLINE_VOXEL_RADIUS,
        )
    )

    return traj

class TrajSourceInitializer(LeafSystem):
    """Prevents the robot from falling down uppon simulator creation."""

    def __init__(self,
                excitation_traj, 
                controller_plant,
                X_WC0,
                X_WC1,
                X_WC2,
                save_data_path
            ):
        super().__init__()
        self.DeclareAbstractInputPort("cloud_front", AbstractValue.Make(PointCloud(0)))
        self.DeclareAbstractInputPort("cloud_back_right", AbstractValue.Make(PointCloud(0)))
        self.DeclareAbstractInputPort("cloud_back_left", AbstractValue.Make(PointCloud(0)))
        self._X_WC0 = X_WC0
        self._X_WC1 = X_WC1
        self._X_WC2 = X_WC2
        self.save_data_path = save_data_path

        self._excitation_traj = excitation_traj

        self._traj_source: TrajectorySource = None
        self._wsg_traj_source: TrajectorySource = None
        self._initialized = False

        self._iiwa_position_measured_input_port = self.DeclareVectorInputPort(
            "iiwa.position_measured", 7
        )

         # Create cci planner.
        self._edge_inflator, self._cci_objects = get_cci_edge_inflator()
        self._drm_planner = get_drm_planner(self._cci_objects)

        self.ik_domain = HPolyhedron.MakeBox(controller_plant.GetPositionLowerLimits()+1e-2, 
                                    controller_plant.GetPositionUpperLimits()-1e-2)

        self.DeclareInitializationDiscreteUpdateEvent(self._init)

    def set_traj_source(self, traj_source):
        self._traj_source = traj_source

    def _init(self, context, discrete_values):
        if self._initialized:
            return

        assert self._traj_source is not None

        # Plan pick 
        manipuland_pcd, scene_pcd = self.GetPointCloud(context)
        pcd_with_background = Concatenate([manipuland_pcd, scene_pcd])

        # get end effector pose from grasp pose
        X_GE = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 0)), [0, 0, -eef_to_gripper_length])

        # pregrasp is negative z in the gripper frame
        X_GgraspGpregrasp = RigidTransform([0, 0.0, -pregrasp_dist])

        # Initialize DRM with current pcd
        edge_inflator, cci_objects = get_cci_edge_inflator()
        drm_planner = get_drm_planner(cci_objects, manipuland_pcd.xyzs().T)

        cci_checker = CCICollisionChecker(cci_objects['cci_mplant'], 
                                        cci_objects['cci_plant'].getRobotGeometryIds(), 
                                        cci.Voxels(manipuland_pcd.xyzs()), 
                                        ONLINE_VOXEL_RADIUS)
        
        gripper_model_path = "file://./home/real2sim/src/Real2SimObjectManipulation/models/schunk_wsg_50_welded_fingers_w_buffer.sdf"
        grasp_node = GraspListener(gripper_model_path=gripper_model_path)
        is_manual = True
        grasp_node.compute_candidate_grasps(
            manipuland_pcd,
            pcd_with_background,
            candidate_num=1,
            num_samples=30,
            random_seed=np.random.randint(1000),
            grasp_type=GraspType.STABLE,
            split_ratio_threshold=0.5,
            is_manual=is_manual
        )

        grasps, grasp_costs = grasp_node.get_best_grasps()
        if not is_manual:
            grasps = copy.deepcopy(grasps)
            grasp_costs = copy.deepcopy(grasp_costs)
            while len(grasps) < 5:
                print(f"not enough grasps ({len(grasps)}), sampling more points")
                grasp_node.compute_candidate_grasps(
                    manipuland_pcd,
                    pcd_with_background,
                    candidate_num=1,
                    num_samples=30,
                    random_seed=np.random.randint(1000),
                    grasp_type=GraspType.STABLE,
                    split_ratio_threshold=0.65
                )
                new_grasps, new_grasp_costs = grasp_node.get_best_grasps()
                print("num new grasps:", len(new_grasps))
                grasps += new_grasps
                grasp_costs += new_grasp_costs

                new_sorted_inds = np.argsort(np.array(grasp_costs))
                grasps = [grasps[idx] for idx in new_sorted_inds]

        print("grasps:", len(grasps))
        q = self._iiwa_position_measured_input_port.Eval(context)
        for i in range(len(grasps)):
            print("Grasp:", grasps[i])

            X_WG = grasps[i]
            X_Wgrasp = (RigidTransform(X_WG) @ X_GE)

            q_goal = solve_via_analytic_IK(
                pose=X_Wgrasp,
                current_config=q,
                ik_domain=self.ik_domain,
                checker=cci_checker
            )

            if q_goal is None:
                continue
            
            q_sys_id_grasp = q_goal
            X_WG_sys_id = RigidTransform(X_WG)
            break

        X_WE = X_WG_sys_id.multiply(X_GE) # E = link 7 frame
        X_EW = X_WE.inverse()

        # Save manipuland pcd and grasp for system ID alignment.
        # TODO: Express manipuland_cloud_points in link 7 frame
        manipuland_cloud_points = manipuland_pcd.xyzs() # X_WP, Shape (3, N)
        manipuland_cloud_points_link7_frame = X_EW @ manipuland_cloud_points # X_EP, Shape (3, N)
        manipuland_cloud_points_link7_frame = manipuland_cloud_points_link7_frame.T # Shape (N,3)
        np.save(
            os.path.join(self.save_data_path, "manipuland_cloud_link7_frame.npy"),
            manipuland_cloud_points_link7_frame,
        )

        manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(manipuland_pcd.xyzs().T))
        manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

        gripper_xyzs = grasp_node.hand_collision_model.to_pcd()
        gripper_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(gripper_xyzs)
        ).voxel_down_sample(ONLINE_VOXEL_RADIUS).transform(
            (X_WG @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4()))
        gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])

        viz_geoms = [manipuland_cloud, gripper_cloud]
        o3d.visualization.draw_plotly(viz_geoms)

        input("press enter to execute pick")

        # Solve for pick trajectory before moving
        # Pose of the 2nd bin to place the object into.
        end_bin = RigidTransform(RotationMatrix(RollPitchYaw(0.0, np.pi/2, np.pi)), [0.30, -0.55, 0.2])

        X_G = {
            "pick": X_WE,
            "prepick": X_WE @ X_GgraspGpregrasp,
            "place": end_bin,
            "postplace": end_bin,
        }

        X_G, times = MakePickGripperFrames(X_G)

        q_postgrasp_bin = solve_via_analytic_IK(
            pose=X_G["place"],
            current_config=q,
            ik_domain=self.ik_domain,
            checker=cci_checker
        )

        # Go to pick
        obstacles_vox = np.hstack([camera_vox, bin_cam_vox, bin_cam_pole_vox])

        traj = scs_trajopt(
            q, 
            q_sys_id_grasp, 
            drm_planner,
            cci_objects, 
            edge_inflator, 
            obstacles_vox, 
            vel_limits=np.ones(7)*0.5,
            acc_limits=np.ones(7)*0.5,
        )
        
        breaks = np.linspace(0, traj.end_time(), int(1e3), endpoint=False)
        knots = traj.vector_values(breaks)

        to_pick_traj = reparameterize_with_toppra(
            trajectory=knots.T,
            plant=self._iiwa_controller_plant,
            velocity_limits=self.velocity_limits,
            acceleration_limits=self.acceleration_limits,
            num_grid_points=100,
        )
        to_pick_wsg_command = PiecewisePolynomial.ZeroOrderHold(
            [to_pick_traj.start_time(), to_pick_traj.end_time()], np.hstack([[opened], [opened]]))
        
        wait_at_pick_traj = PiecewisePolynomial.ZeroOrderHold(
            breaks=[
                to_pick_traj.end_time(),
                to_pick_traj.end_time() + 2.0,
            ],
            samples=np.stack([to_pick_traj.value(to_pick_traj.end_time()), to_pick_traj.value(to_pick_traj.end_time())], axis=1),
        )
        wait_at_pick_wsg_command = PiecewisePolynomial.ZeroOrderHold(
            [wait_at_pick_traj.start_time(), wait_at_pick_traj.start_time() + 0.5, wait_at_pick_traj.end_time()], 
            np.hstack([[opened], [closed], [closed]])
        )
        
        q_start = self._excitation_traj.value(0.0)
        traj = scs_trajopt(
            start=wait_at_pick_traj.value(wait_at_pick_traj.end_time()),
            goal=q_start,
            drm_planner=self._drm_planner,
            cci_obj=self._cci_objects,
            edge_inflator=self._edge_inflator,
            vox=None,
            vel_limits=np.ones(7)*0.5,
            acc_limits=np.ones(7)*0.5,
        )
        
        breaks = np.linspace(wait_at_pick_traj.end_time(), wait_at_pick_traj.end_time()+ traj.end_time(), int(1e3), endpoint=False)
        knots = traj.vector_values(breaks)

        to_start_traj = reparameterize_with_toppra(
            trajectory=knots.T,
            plant=self._iiwa_controller_plant,
            velocity_limits=self.velocity_limits,
            acceleration_limits=self.acceleration_limits,
            num_grid_points=100,
        )
        to_start_wsg_command = PiecewisePolynomial.ZeroOrderHold(
            [to_start_traj.start_time(), to_start_traj.end_time()], 
            np.hstack([[closed], [closed]])
        )

        # Add all trajs before this
        wait_at_start_traj = PiecewisePolynomial.ZeroOrderHold(
            breaks=[
                to_start_traj.end_time(),
                to_start_traj.end_time() + 2.0,
            ],
            samples=np.stack([q_start, q_start], axis=1),
        )
        wait_at_start_wsg_command = PiecewisePolynomial.ZeroOrderHold(
            [wait_at_start_traj.start_time(), wait_at_start_traj.end_time()], 
            np.hstack([[closed], [closed]])
        )

        excitation_traj_time = PiecewisePolynomial().FirstOrderHold(
            [0.0, self._excitation_traj.end_time()],
            [[0.0, self._excitation_traj.end_time()]],
        )
        excitation_traj_time.shiftRight(wait_at_start_traj.end_time())
        shifted_excitation_traj = PathParameterizedTrajectory(
            path=self._excitation_traj, time_scaling=excitation_traj_time
        )

        self.excitation_traj_start_time = shifted_excitation_traj.start_time()
        self.excitation_traj_end_time = shifted_excitation_traj.end_time()
        shifted_excitation_wsg_command = PiecewisePolynomial.ZeroOrderHold(
            [shifted_excitation_traj.start_time(), shifted_excitation_traj.end_time()], 
            np.hstack([[closed], [closed]])
        )

        traj = scs_trajopt(
            q, 
            q_postgrasp_bin, 
            drm_planner,
            cci_objects, 
            edge_inflator, 
            obstacles_vox, 
            vel_limits=np.ones(7)*0.5,
            acc_limits=np.ones(7)*0.5,
        )
        
        breaks = np.linspace(self.excitation_traj_end_time, self.excitation_traj_end_time+ traj.end_time(), int(1e3), endpoint=False)
        knots = traj.vector_values(breaks)

        to_bin_traj = reparameterize_with_toppra(
            trajectory=knots.T,
            plant=self._iiwa_controller_plant,
            velocity_limits=self.velocity_limits,
            acceleration_limits=self.acceleration_limits,
            num_grid_points=100,
        )
        to_bin_wsg_command = PiecewisePolynomial.ZeroOrderHold(
            [to_bin_traj.start_time(), to_bin_traj.end_time()], 
            np.hstack([[closed], [closed]])
        )

        wait_at_bin_traj = PiecewisePolynomial.ZeroOrderHold(
            breaks=[
                to_bin_traj.end_time(),
                to_bin_traj.end_time() + 2.0,
            ],
            samples=np.stack([to_bin_traj.value(to_bin_traj.end_time()), to_bin_traj.value(to_bin_traj.end_time())], axis=1),
        )
        wait_at_bin_wsg_command = PiecewisePolynomial.ZeroOrderHold(
            [wait_at_bin_traj.start_time(), wait_at_bin_traj.start_time() + 0.5, wait_at_bin_traj.end_time()], 
            np.hstack([[closed], [opened], [opened]])
        )

        self.end_time = wait_at_bin_traj.end_time()

        composite_traj = CompositeTrajectory(
            [to_pick_traj, wait_at_pick_traj, to_start_traj, wait_at_start_traj, shifted_excitation_traj, to_bin_traj, wait_at_bin_traj])
        self._traj_source.UpdateTrajectory(composite_traj)

        composite_wsg_traj = CompositeTrajectory(
            [to_pick_wsg_command, wait_at_pick_wsg_command, to_start_wsg_command, wait_at_start_wsg_command, 
             shifted_excitation_wsg_command, to_bin_wsg_command, wait_at_bin_wsg_command])
        self._wsg_traj_source.UpdateTrajectory(composite_wsg_traj)

        print("Initialized traj source.")

    def reset(self):
        self._initialized = False

    def GetPointCloud(self, context):
        # Get manipuland pcd 
        cloud0 = self.GetInputPort("cloud_front").Eval(context)
        X_adjust0 = RigidTransform(RotationMatrix(RollPitchYaw(0.0, 0.0, 0.0)),[-0.017, -0.025, -0.015])
        transformed_xyzs = X_adjust0 @ cloud0.xyzs()
        cloud0.mutable_xyzs()[:] = transformed_xyzs
        pcd0 = cloud0.Crop(lower_xyz=[0.23, -0.17, platform_height], upper_xyz=[0.7, 0.17, 0.27])
        pcd0.EstimateNormals(radius=0.1, num_closest=30)
        pcd0.FlipNormalsTowardPoint(self._X_WC0.translation())

        cloud1 = self.GetInputPort("cloud_back_left").Eval(context)
        X_adjust1 = RigidTransform(RotationMatrix(RollPitchYaw(0.0, 0.0, 0.0)),[-0.03, 0.0, -0.015])
        transformed_xyzs = X_adjust1 @ cloud1.xyzs()
        cloud1.mutable_xyzs()[:] = transformed_xyzs
        pcd1 = cloud1.Crop(lower_xyz=[0.23, -0.17, platform_height], upper_xyz=[0.7, 0.17, 0.27])
        pcd1.EstimateNormals(radius=0.1, num_closest=30)
        pcd1.FlipNormalsTowardPoint(self._X_WC2.translation()) # I screwed up the cameras somewhere so the left anf right are flipped, need to fix

        cloud2 = self.GetInputPort("cloud_back_right").Eval(context)
        X_adjust2 = RigidTransform(RotationMatrix(RollPitchYaw(0.0, 0.0, 0.0)),[0.0, 0.005, 0.005])
        transformed_xyzs = X_adjust2 @ cloud2.xyzs()
        cloud2.mutable_xyzs()[:] = transformed_xyzs
        pcd2 = cloud2.Crop(lower_xyz=[0.23, -0.17, platform_height], upper_xyz=[0.7, 0.17, 0.27])
        pcd2.EstimateNormals(radius=0.1, num_closest=30)
        pcd2.FlipNormalsTowardPoint(self._X_WC1.translation())

        merged_pcd = Concatenate([pcd0, pcd1, pcd2])
        down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=ONLINE_VOXEL_RADIUS)

        
        # remove outliers
        o3d_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(down_sampled_pcd.xyzs().T))
        try:
            cl, ind = o3d_cloud.remove_statistical_outlier(
                nb_neighbors=int(down_sampled_pcd.xyzs().shape[1] // 10), 
                std_ratio=2.0)
            filtered_pts = np.asarray(o3d_cloud.points)[ind]
            down_sampled_pcd.resize(filtered_pts.shape[0])
            down_sampled_pcd.mutable_xyzs()[:] = filtered_pts.T
        except:
            pass
        
        VISUALIZE_MERGED_PCD = False
        if VISUALIZE_MERGED_PCD:
            pcd = down_sampled_pcd.xyzs().T # Shape (N,3)
            pcd_normals = down_sampled_pcd.normals().T # Shape (N,3)
            o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
            o3d_pcd.paint_uniform_color([0,0,1])
            o3d_pcd.normals = o3d.utility.Vector3dVector(pcd_normals)
            o3d.visualization.draw_geometries([o3d_pcd], point_show_normal=True, window_name="merged pcd")

        current_manipuland_pcd = down_sampled_pcd

        # Get table pcd 
        cloud0 = self.GetInputPort("cloud_front").Eval(context)
        transformed_xyzs = X_adjust0 @ cloud0.xyzs()
        cloud0.mutable_xyzs()[:] = transformed_xyzs
        pcd0 = cloud0.Crop(lower_xyz=[0.1, -0.3, 0.0], upper_xyz=[0.85, 0.3, 0.07]).VoxelizedDownSample(voxel_size=0.01)
        pcd0.EstimateNormals(radius=0.1, num_closest=30)
        pcd0.FlipNormalsTowardPoint(self._X_WC0.translation())

        cloud1 = self.GetInputPort("cloud_back_left").Eval(context)
        transformed_xyzs = X_adjust1 @ cloud1.xyzs()
        cloud1.mutable_xyzs()[:] = transformed_xyzs
        pcd1 = cloud1.Crop(lower_xyz=[0.1, -0.3, 0.0], upper_xyz=[0.85, 0.3, 0.07]).VoxelizedDownSample(voxel_size=0.01)
        pcd1.EstimateNormals(radius=0.1, num_closest=30)
        pcd1.FlipNormalsTowardPoint(self._X_WC2.translation())

        cloud2 = self.GetInputPort("cloud_back_right").Eval(context)
        transformed_xyzs = X_adjust2 @ cloud2.xyzs()
        cloud2.mutable_xyzs()[:] = transformed_xyzs
        pcd2 = cloud2.Crop(lower_xyz=[0.1, -0.3, 0.0], upper_xyz=[0.85, 0.3, 0.07]).VoxelizedDownSample(voxel_size=0.01)
        pcd2.EstimateNormals(radius=0.1, num_closest=30)
        pcd2.FlipNormalsTowardPoint(self._X_WC1.translation())

        merged_pcd = Concatenate([pcd0, pcd1, pcd2])
        
        # remove outliers
        o3d_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(merged_pcd.xyzs().T))
        cl, ind = o3d_cloud.remove_statistical_outlier(
            nb_neighbors=int(merged_pcd.xyzs().shape[1] // 10), 
            std_ratio=2.0)
        filtered_pts = np.asarray(o3d_cloud.points)[ind]
        merged_pcd.resize(filtered_pts.shape[0])
        merged_pcd.mutable_xyzs()[:] = filtered_pts.T
        current_scene_pcd = merged_pcd

        return current_manipuland_pcd, current_scene_pcd
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario_path",
        type=str,
        default="scenario_datas/scenario_data_grasping_hardware.yml",
        help="Path to the scenario file. This must contain an iiwa model named 'iiwa' "
        "and a gripper model named 'wsg'.",
    )
    parser.add_argument(
        "--traj_parameter_path",
        type=Path,
        default=SYS_ID_TRAJ_PARAMETER_PATH,
        help="Path to the trajectory parameter folder. The folder must contain "
        + "'a_value.npy', 'b_value.npy', and 'q0_value.npy' or 'control_points.npy', "
        + "'knots.npy', and 'spline_order.npy'.",
    )
    parser.add_argument(
        "--save_data_path",
        type=Path,
        required=True,
        help="Path to save the data to. Each data collection run will be saved to a "
        + "separate subdirectory of this path.",
    )
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )
    parser.add_argument(
        "--time_horizon",
        type=float,
        default=10.0,
        help="The time horizon/ duration of the trajectory. Only used for Fourier "
        + "series trajectories.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    scenario_path = args.scenario_path
    traj_parameter_path = args.traj_parameter_path
    save_data_path = args.save_data_path
    use_hardware = args.use_hardware
    time_horizon = args.time_horizon

    if os.path.exists(save_data_path):
        input(f"{save_data_path} already exists. Press enter to delete and re-create. Ctr+c to stop.")
        shutil.rmtree(save_data_path)

    builder = DiagramBuilder()
    scenario = LoadScenario(filename=scenario_path)
    # Ensure correct timestep for position control mode.
    scenario.plant_config.time_step == 5e-3

    meshcat = StartMeshcat()

    station: RobotDiagram = builder.AddNamedSystem(
        "hardware_station",
        MakeHardwareStation(
            scenario=scenario,
            meshcat=meshcat,
            hardware=use_hardware,
        ),
    )

    # Load trajectory parameters
    is_fourier_series = os.path.exists(traj_parameter_path / "a_value.npy")
    if is_fourier_series:
        traj_attrs = FourierSeriesTrajectoryAttributes.load(traj_parameter_path)
        excitation_traj = FourierSeriesTrajectory(
            traj_attrs=traj_attrs,
            time_horizon=time_horizon,
        )
    else:
        raise ValueError(f"Invalid trajectory type: {traj_parameter_path}")
    start_positions = excitation_traj.value(0.0)

    # Placeholder trajectory
    traj_source: TrajectorySource = builder.AddNamedSystem(
        "trajectory_source",
        TrajectorySource(
            trajectory=PiecewisePolynomial.ZeroOrderHold(
                [0.0, 1.0], np.zeros((len(excitation_traj.value(0.0)), 2))
            ),
            output_derivative_order=0,
        ),
    )
    builder.Connect(
        traj_source.get_output_port(), station.GetInputPort("iiwa.position")
    )

    # Add a placeholder traj.
    wsg_traj_source: TrajectorySource = builder.AddNamedSystem(
        "wsg_trajectory_source",
        TrajectorySource(
            trajectory=PiecewisePolynomial.ZeroOrderHold([0.0, 1.0], np.zeros((1, 2))),
        ),
    )
    builder.Connect(
        wsg_traj_source.get_output_port(), station.GetInputPort("wsg.position")
    )

    # Add data loggers
    num_positions = station.GetInputPort("iiwa.position").size()
    logging_period = 1e-3
    measured_position_logger: VectorLogSink = builder.AddNamedSystem(
        "measured_position_logger",
        VectorLogSink(num_positions, publish_period=logging_period),
    )
    measured_torque_logger: VectorLogSink = builder.AddNamedSystem(
        "measured_torque_logger",
        VectorLogSink(num_positions, publish_period=logging_period),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        measured_position_logger.get_input_port(),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.torque_measured"),
        measured_torque_logger.get_input_port(),
    )

    # Add cameras
    camera0_pcd = builder.AddSystem(DepthImageToPointCloud(CameraInfo(848, 480, 600.165, 600.165, 429.152, 232.822)))
    camera1_pcd = builder.AddSystem(DepthImageToPointCloud(CameraInfo(848, 480, 626.633, 626.633, 432.041, 245.465)))
    camera2_pcd = builder.AddSystem(DepthImageToPointCloud(CameraInfo(848, 480, 596.492, 596.492, 416.694, 240.225)))
    
    builder.Connect(station.GetOutputPort("camera0.depth_image"), camera0_pcd.GetInputPort("depth_image"))
    builder.Connect(station.GetOutputPort("camera1.depth_image"), camera1_pcd.GetInputPort("depth_image"))
    builder.Connect(station.GetOutputPort("camera2.depth_image"), camera2_pcd.GetInputPort("depth_image"))

    x_front_rgb = RigidTransform(np.loadtxt("/home/real2sim/calibrations/2_10_calibrations_aligned/front.txt"))

    # Back Right camera
    x_back_right_rgb = RigidTransform(np.loadtxt("/home/real2sim/calibrations/2_10_calibrations_aligned/back_right.txt"))

    # Back Left camera
    x_back_left_rgb = RigidTransform(np.loadtxt("/home/real2sim/calibrations/2_10_calibrations_aligned/back_left.txt"))

    # rgb calibration to depth calibration (from realsense specs)
    # Front camera
    x_depth_rgb_front = RigidTransform([[0.999986,      -0.000127587,   0.00531376, 0.015102],
                                        [0.000116105,   0.999998,       0.00216102, 6.44158e-05],
                                        [-0.00531402,   -0.00216038,    0.999984,   -0.000426644],
                                        [0,             0,              0,          1]])
    x_front_camera = x_front_rgb @ x_depth_rgb_front

    # Back Right camera
    x_depth_rgb_back_right = RigidTransform([[0.999968,  -0.00700185,   0.00399879,     0.015085],
                                            [ 0.00701494,      0.99997,  -0.00326805,  -2.1265e-05],
                                            [-0.00397579,   0.00329599,     0.999987, -0.000455872],
                                            [          0,            0,            0,            1]])
    x_back_right_camera = x_back_right_rgb @ x_depth_rgb_back_right

    # Back Left camera
    x_depth_rgb_back_left = RigidTransform([[0.999998, -0.000191981,  -0.00215977,    0.0150991],
                                            [0.000214442,     0.999946,    0.0104041,  7.71731e-05],
                                            [0.00215765,   -0.0104046,     0.999944, -0.000317806],
                                            [          0,            0,            0,            1]])
    x_back_left_camera = x_back_left_rgb @ x_depth_rgb_back_left

    # connect stationary camera pcd source
    camera0_pose_source = builder.AddSystem(CameraPoseInWorldSource(x_front_camera, handeye=False))
    camera1_pose_source = builder.AddSystem(CameraPoseInWorldSource(x_back_right_camera, handeye=False))
    camera2_pose_source = builder.AddSystem(CameraPoseInWorldSource(x_back_left_camera, handeye=False))

    builder.Connect(
        camera0_pose_source.GetOutputPort("X_WC"),
        camera0_pcd.GetInputPort("camera_pose"),
    )

    builder.Connect(
        camera1_pose_source.GetOutputPort("X_WC"),
        camera1_pcd.GetInputPort("camera_pose"),
    )

    builder.Connect(
        camera2_pose_source.GetOutputPort("X_WC"),
        camera2_pcd.GetInputPort("camera_pose"),
    )

    initializer: TrajSourceInitializer = builder.AddSystem(
        TrajSourceInitializer(excitation_traj)
    )
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        initializer.get_input_port("iiwa.position_measured"),
    )
    builder.Connect(
        camera0_pcd.GetOutputPort("point_cloud"),
        initializer.GetInputPort("cloud_front"),
    )
    builder.Connect(
        camera1_pcd.GetOutputPort("point_cloud"),
        initializer.GetInputPort("cloud_back_left"),
    )
    builder.Connect(
        camera2_pcd.GetOutputPort("point_cloud"),
        initializer.GetInputPort("cloud_back_right"),
    )
    initializer.set_traj_source(traj_source)


    # Build and setup simulation
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)

    # Do pick + excitation traj
    simulator.AdvanceTo(initializer.end_time + 1.0)

    # Save data
    measured_position_data = (
        measured_position_logger.FindLog(simulator.get_context()).data().T
    )
    measured_torque_data = (
        measured_torque_logger.FindLog(simulator.get_context()).data().T
    )
    sample_times_s = measured_position_logger.FindLog(
        simulator.get_context()
    ).sample_times()

    # Only keep data during excitation trajectory execution
    data_start_time = initializer.excitation_traj_start_time
    excitation_traj_end_time = initializer.excitation_traj_end_time
    excitation_traj_start_idx = np.argmax(sample_times_s >= data_start_time)
    excitation_traj_end_idx = np.argmax(
        sample_times_s >= excitation_traj_end_time
    )
    measured_position_data = measured_position_data[
        excitation_traj_start_idx:excitation_traj_end_idx
    ]
    measured_torque_data = measured_torque_data[
        excitation_traj_start_idx:excitation_traj_end_idx
    ]
    sample_times_s = sample_times_s[
        excitation_traj_start_idx:excitation_traj_end_idx
    ]
    # Shift sample times to start at 0
    sample_times_s -= sample_times_s[0]

    # Remove duplicated samples
    _, unique_indices = np.unique(sample_times_s, return_index=True)
    if len(unique_indices) < len(sample_times_s):
        print(
            f"{len(unique_indices)} out of {len(sample_times_s)} data points "
            "are unique!"
        )
        measured_position_data = measured_position_data[unique_indices]
        measured_torque_data = measured_torque_data[unique_indices]
        sample_times_s = sample_times_s[unique_indices]

    # Save data
    np.save(save_data_path / "joint_positions.npy", measured_position_data)
    np.save(save_data_path / "joint_torques.npy", measured_torque_data)
    np.save(save_data_path / "sample_times_s.npy", sample_times_s)

    print("Saved system id data to", save_data_path)


if __name__ == "__main__":
    main()
