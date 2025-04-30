import numpy as np
import logging
import pickle
import datetime
import time
import open3d as o3d
import os
import copy
from scipy.spatial.transform import Rotation as R
from perception.teaser import icp  
from perception.pcd_util import compute_principal_minor_components, crop_connected_points
from planning.grasp import GraspListener, GraspType
from planning.toppra import reparameterize_with_toppra
from planning.trajectories import (
    MakePickAndDisplayGripperFrames,
    MakeDisplayJointPositionsTrajectory,
    MakeGripperCommandTrajectory,
    MakeGripperPoseTrajectory,
    MakePickGripperFrames,
    TrajType
)
from planning.gcs import plan_unconstrained_gcs_path_start_to_goal
from planning.inverse_kinematics import solve_via_analytic_IK, solve_via_analytic_IK_with_retries
from planning.motion_planning import plan_path_custom, plan_drm
from planning.misc.uncertainty_mapping import VoxelMap
from iiwa_setup_dataclasses.trajectories import TrajectoryWithTimingInformation, Trajectory
from iiwa_setup_dataclasses.bspline_trajectory import CompositeBezierCurveTrajectoryAttributes
from pydrake.systems.framework import LeafSystem
from pydrake.perception import (
    Concatenate,
    PointCloud
)
from pydrake.systems.sensors import ImageLabel16I
from pydrake.math import (
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
)
from pydrake.common.value import (
    AbstractValue
)
from pydrake.trajectories import (
    PiecewisePose,
    PiecewisePolynomial,
)
from pydrake.all import BsplineTrajectory, BsplineBasis
from robot_payload_id.control.trajectory import FourierSeriesTrajectory

from robot_payload_id.utils import (
    BsplineTrajectoryAttributes,
    FourierSeriesTrajectoryAttributes,
)

import pydrake.planning as mut
from pydrake.common import RandomGenerator, use_native_cpp_logging
from pydrake.planning import (RobotDiagramBuilder,
                              SceneGraphCollisionChecker)
from pydrake.solvers import MosekSolver, GurobiSolver
from pydrake.all import (
    IrisZo, 
    IrisZoOptions, 
    Hyperellipsoid, 
    HPolyhedron,
    LoadModelDirectives,
    ProcessModelDirectives,
    InputPortIndex
)
from pydrake.geometry import Rgba

from manipulation.meshcat_utils import AddMeshcatTriad
from planning.utils.planner_states import PlannerState, PickState
from pathlib import Path

import sys
from planning.utils.csdecomp_path import CSDECOMP_PATH
sys.path.append(f'{CSDECOMP_PATH}/bazel-bin/csdecomp/src/pybind/pycsdecomp')
import pycsdecomp as csd

ONLINE_VOXEL_RADIUS = 0.005
PETE_ASSETS =  os.path.dirname(__file__)+"/../pete_assets/"
SAFE_DIRECTIVES = PETE_ASSETS+'assets/directives/iiwa7_on_table_with_ceiling.yaml'
SYS_ID_TRAJ_PARAMETER_PATH = Path(os.path.abspath(os.path.join(__file__ ,"../../traj_feb8")))

def get_seeded_region_safe(q_nominal):
    models_path = PETE_ASSETS+'assets/directives/iiwa7_on_table_with_ceiling.dmd.yaml'
    print("getting safe seeded region around", q_nominal)

    use_native_cpp_logging()
    params = dict(edge_step_size=0.125)
    builder = RobotDiagramBuilder()
    builder.parser().package_map().Add("adaptive_decomp", PETE_ASSETS+"assets")
    builder.parser().package_map().Add("iiwa_description", PETE_ASSETS+"assets/iiwa")
    builder.parser().package_map().Add("wsg_description", PETE_ASSETS+"assets/wsg_description")
    builder.parser().package_map().Add("tri_finray_gripper", PETE_ASSETS+"assets/tri_finray_gripper")
    builder.parser().AddModels(models_path)
    iiwa_model_instance_index = builder.plant().GetModelInstanceByName("iiwa7")
    wsg_model_instance_index = builder.plant().GetModelInstanceByName("wsg")
    plant = builder.plant()
    plant.Finalize()
    params["robot_model_instances"] = [iiwa_model_instance_index, wsg_model_instance_index]
    params["model"] = builder.Build()
    domain = HPolyhedron.MakeBox(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits())
    checker = SceneGraphCollisionChecker(**params)

    opts = IrisZoOptions()
    opts.max_iterations = 10
    region = IrisZo(checker, Hyperellipsoid.MakeHypersphere(1e-4, q_nominal), domain, opts)

    return region

def get_seeded_region(models_path, com, rot, dims, q_nominal):
    # Get bounding box of object
    rpy = rot.ToRollPitchYaw().vector()
    bounding_box_urdf = """<?xml version="1.0"?>
<robot name="bounding_box">
  <link name="bounding_box">
    <collision name="bounding_box">
        <origin rpy="%f %f %f" xyz="%f %f %f"/>
      <geometry>
        <box size="%f %f %f"/>
      </geometry>
    </collision>
  </link>
    <joint name="fixed_link_weld" type="fixed">
    <parent link="world"/>
    <child link="bounding_box"/>
    </joint>
</robot>
    """ % (rpy[0], rpy[1], rpy[2], com[0], com[1], com[2], dims[0], dims[1], dims[2])

    print("getting seeded region around", q_nominal)

    use_native_cpp_logging()
    params = dict(edge_step_size=0.125)
    builder = RobotDiagramBuilder()
    builder.parser().AddModels(models_path)
    builder.parser().AddModelsFromString(bounding_box_urdf, "urdf")
    iiwa_model_instance_index = builder.plant().GetModelInstanceByName("iiwa")
    wsg_model_instance_index = builder.plant().GetModelInstanceByName("wsg")
    plant = builder.plant()
    plant.Finalize()
    params["robot_model_instances"] = [iiwa_model_instance_index, wsg_model_instance_index]
    params["model"] = builder.Build()
    domain = HPolyhedron.MakeBox(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits())
    checker = SceneGraphCollisionChecker(**params)

    opts = IrisZoOptions()
    opts.max_iterations = 10
    region = IrisZo(checker, Hyperellipsoid.MakeHypersphere(1e-4, q_nominal), domain, opts)

    return region


def get_regions(models_path, com, rot, dims):
    # Get bounding box of object
    rpy = rot.ToRollPitchYaw().vector()
    bounding_box_urdf = """<?xml version="1.0"?>
<robot name="bounding_box">
  <link name="bounding_box">
    <collision name="bounding_box">
        <origin rpy="%f %f %f" xyz="%f %f %f"/>
      <geometry>
        <box size="%f %f %f"/>
      </geometry>
    </collision>
  </link>
    <joint name="fixed_link_weld" type="fixed">
    <parent link="world"/>
    <child link="bounding_box"/>
    </joint>
</robot>
    """ % (rpy[0], rpy[1], rpy[2], com[0], com[1], com[2], dims[0], dims[1], dims[2])
    print(bounding_box_urdf)

    use_native_cpp_logging()
    params = dict(edge_step_size=0.125)
    builder = RobotDiagramBuilder()
    builder.parser().AddModels(models_path)
    builder.parser().AddModelsFromString(bounding_box_urdf, "urdf")
    iiwa_model_instance_index = builder.plant().GetModelInstanceByName("iiwa")
    wsg_model_instance_index = builder.plant().GetModelInstanceByName("wsg")
    params["robot_model_instances"] = [iiwa_model_instance_index, wsg_model_instance_index]
    params["model"] = builder.Build()
    checker = SceneGraphCollisionChecker(**params)

    options = mut.IrisFromCliqueCoverOptions()
    options.num_points_per_coverage_check = 5000
    options.num_points_per_visibility_round = 1000
    options.minimum_clique_size = 16
    options.coverage_termination_threshold = 0.7

    generator = RandomGenerator(0)

    if (MosekSolver().available() and MosekSolver().enabled()) or (
            GurobiSolver().available() and GurobiSolver().enabled()):
        # We need a MIP solver to be available to run this method.
        sets = mut.IrisInConfigurationSpaceFromCliqueCover(
            checker=checker, options=options, generator=generator,
            sets=[]
        )

        if len(sets) < 1:
            raise("No regions found")

        return sets
    else:
        print("No solvers available")
    
def get_csd_plant():
    parser = csd.URDFParser()
    parser.register_package("adaptive_decomp", PETE_ASSETS+"assets")
    parser.register_package("iiwa_description", PETE_ASSETS+"assets/iiwa")
    parser.register_package("wsg_description", PETE_ASSETS+"assets/wsg_description")
    parser.register_package("tri_finray_gripper", PETE_ASSETS+"assets/tri_finray_gripper")
    parser.parse_directives(PETE_ASSETS+"assets/directives/iiwa7_on_table.yaml")
    plant = parser.build_plant()

    return plant

def make_iiwa_plant():
    directives_file = PETE_ASSETS+'assets/directives/iiwa7_on_table.yaml'
    builder = RobotDiagramBuilder()
    plant = builder.plant()
    scene_graph = builder.scene_graph()
    parser = builder.parser()

    parser.package_map().Add("adaptive_decomp", PETE_ASSETS+"assets")
    parser.package_map().Add("iiwa_description", PETE_ASSETS+"assets/iiwa")
    parser.package_map().Add("wsg_description", PETE_ASSETS+"assets/wsg_description")
    parser.package_map().Add("tri_finray_gripper", PETE_ASSETS+"assets/tri_finray_gripper")

    directives = LoadModelDirectives(directives_file)
    models = ProcessModelDirectives(directives, plant, parser)
    plant.Finalize()

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)
    diagram.ForcedPublish(diagram_context)

    return plant, plant_context

def drm_planner(csd_plant, vox=None):
    drm_pl_opts = csd.DrmPlannerOptions()
    drm_pl_opts.max_number_planning_attempts = 50
    drm_pl_opts.try_shortcutting = True
    drm_pl_opts.online_edge_step_size = 0.005

    drm_planner = csd.DrmPlanner(csd_plant, drm_pl_opts)
    root = os.path.abspath(os.path.dirname(__file__)+'/../')
    drm_planner.LoadRoadmap(root+"/planning/roadmaps/iiwa_roadmap_0_0_0.01_0.2_100000_10_4.5_0.45.rm")

    if vox is not None:
        online_voxel_observation = csd.Voxels(vox.T)
        drm_planner.BuildCollisionSet(online_voxel_observation)
    
    return drm_planner

def save_regions_pkl(sets, models_path, dirstr, name=None):
    pkl_path = dirstr + f'/{name}.pkl'
    if name == None:
        time_str = datetime.datetime.now().strftime('%d%m%y_%H%M%S')
        pkl_path = dirstr+f'/{models_path.split("/")[-1]}_{time_str}_regions.pkl'

    with open(pkl_path, 'wb') as f:
        pickle.dump(sets, f)

def make_trajectory_save_dirs(dirstr, traj_name):
    if not os.path.exists(dirstr+f"/{traj_name}"):
        os.makedirs(dirstr+f"/{traj_name}")
    if not os.path.exists(dirstr+f"/{traj_name}" + "/control_points"):
        os.makedirs(dirstr+f"/{traj_name}" + "/control_points")
    if not os.path.exists(dirstr+f"/{traj_name}" + "/start_times"):
        os.makedirs(dirstr+f"/{traj_name}" + "/start_times")
    if not os.path.exists(dirstr+f"/{traj_name}" + "/end_times"):
        os.makedirs(dirstr+f"/{traj_name}" + "/end_times")

def check_configuration_has_collisions(models_path, com, rot, dims, q):
    # Get bounding box of object
    rpy = rot.ToRollPitchYaw().vector()
    bounding_box_urdf = """<?xml version="1.0"?>
<robot name="bounding_box">
  <link name="bounding_box">
    <collision name="bounding_box">
        <origin rpy="%f %f %f" xyz="%f %f %f"/>
      <geometry>
        <box size="%f %f %f"/>
      </geometry>
    </collision>
  </link>
    <joint name="fixed_link_weld" type="fixed">
    <parent link="world"/>
    <child link="bounding_box"/>
    </joint>
</robot>
    """ % (rpy[0], rpy[1], rpy[2], com[0], com[1], com[2], dims[0], dims[1], dims[2])
    builder = RobotDiagramBuilder()
    plant = builder.plant()
    scene_graph = builder.scene_graph()
    builder.parser().AddModels(models_path)
    builder.parser().AddModelsFromString(bounding_box_urdf, "urdf")
    diagram = builder.Build()

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, q)

    scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)
    query_object = scene_graph.GetOutputPort("query").Eval(scene_graph_context)
    return query_object.HasCollisions()

def apply_centered_rotation_and_translation(pose: np.ndarray, rotation: np.ndarray, translation: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    Apply a rotation and translation to a pose matrix, centered around a specific point.
    
    Args:
        pose: 4x4 homogeneous transformation matrix
        rotation: 3x3 rotation matrix
        translation: 3x1 translation vector
        center: (3,) array specifying the center of rotation
    
    Returns:
        4x4 transformed pose matrix
    """
    # Create translation matrices
    to_center = np.eye(4)
    to_center[:3, 3] = -center
    
    from_center = np.eye(4)
    from_center[:3, 3] = center
    
    # Create rotation matrix in homogeneous coordinates
    rotation_h = np.eye(4)
    rotation_h[:3, :3] = rotation
    
    # Apply transformations in sequence:
    # 1. Translate to center
    # 2. Apply rotation
    # 3. Translate back from center
    # 4. Apply to original pose
    transform = from_center @ rotation_h @ to_center
    transform = transform @ pose
    transform[:3, 3] = transform[:3, 3] + translation
    return transform

lift_for_display_height = 0.05
display_traj_height_buffer = 0.05 # Height between manipuland bottom and floor

q_home = [0.3, 0.4, 0.0, -1.2, 0.0, 1.0, -1.57]

def get_yaw_display_traj(scanning_traj_height=0.50, scanning_traj_robot_to_workspace_dist = 0.5) -> list[RigidTransform]:
    yaw_display_traj = [
        RigidTransform(
            RotationMatrix(RollPitchYaw(np.pi, 0.0, -np.pi/4 * i)),
            [scanning_traj_robot_to_workspace_dist, 0.0, scanning_traj_height]
        ) for i in range(8)
    ]
    
    return yaw_display_traj


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
    
ceiling_vox = generate_points_in_cube((0.6, 0.0, 0.85), (0.4, 0.5, 0.001))
camera_vox = generate_points_in_cube((0.8, 0.0, 0.3), (0.01, 0.15, 0.6))
bin_cam_vox = generate_points_in_cube((-0.0338161, 0.84, 0.2), (0.02, 0.10, 0.4))
bin_cam_pole_vox = generate_points_in_cube((-0.07, 0.44, 0.21), (0.04, 0.08, 0.42))

def generate_rectangle_points(corner1, corner2, corner3, corner4, separation, z_offset=0):
    # Function to generate points along an edge
    def edge_points(start, end, z_offset):
        return np.column_stack((
            np.linspace(start[0], end[0], int(np.linalg.norm(np.subtract(end, start)) / separation) + 1),
            np.linspace(start[1], end[1], int(np.linalg.norm(np.subtract(end, start)) / separation) + 1),
            np.linspace(start[2] + z_offset, end[2] + z_offset, int(np.linalg.norm(np.subtract(end, start)) / separation) + 1)
        ))
    
    # Generate points along each edge and concatenate
    rectangle_points = np.concatenate([
        edge_points(corner1, corner2, z_offset), 
        edge_points(corner2, corner3, z_offset)[1:], 
        edge_points(corner3, corner4, z_offset)[1:], 
        edge_points(corner4, corner1, z_offset)[1:]
    ], axis=0)
    
    return rectangle_points

corner1, corner2, corner3, corner4 = (-0.04, 0.42, 0.05), (-0.04, 0.77, 0.05), (0.155, 0.77, 0.05), (0.155, 0.42, 0.05)
bin_sides_components = []
for z_offset in np.linspace(0, -0.1, int(0.1/0.005)):
    bin_sides_components.append(generate_rectangle_points(corner1, corner2, corner3, corner4, 0.005, z_offset))
bin_sides_vox = np.concatenate(bin_sides_components, axis=0).T

# Stage center transforms are used for placing the object onto the workspace after bin picking.
stage_center0 = RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, -np.pi/2)), [0.5, 0.0, 0.1])
stage_center90 = RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, 0.0)), [0.5, 0.0, 0.0])

bin_depth = 0.06
x_points = np.linspace(corner1[0], corner3[0])
y_points = np.linspace(corner1[1], corner3[1])

# Create the grid of points
x, y = np.meshgrid(x_points, y_points)

# Reshape the grid into a Nx3 array
bin_bottom_box = np.vstack((x.flatten(), y.flatten(), np.zeros_like(x.flatten())))
bin_bottom_box += np.array([0.0, 0.0, -bin_depth])[:, np.newaxis]

platform_height = 0.065 # use value a bit higher than it is to reduce pcd noise

# Pose of the 2nd bin to place the object into.
end_bin = RigidTransform(RotationMatrix(RollPitchYaw(0.0, np.pi/2, np.pi)), [0.30, -0.55, 0.2])

class RegraspPlanner(LeafSystem):
    def __init__(
            self, 
            plant,
            controller_plant,
            X_WC0,
            X_WC1,
            X_WC2,
            X_WC_bin,
            X_WC_obs,
            cam_obs_K,
            obs_width_px,
            obs_height_px,
            meshcat, 
            dirstr,
            time_horizon,
            models_path=None,
            gripper_model_path=None,
            default_home=q_home,
            gripper_length=0.145,
            pregrasp_dist=0.17,
            eef_to_gripper_length=0.16, # 0.16 for the real value,
            num_objs=1,
            is_manual=False,
            is_semimanual=True,
            sys_id_grasp_only=False,
            use_custom_path_planner=False,
        ):
        LeafSystem.__init__(self)

        # For grasp planner
        self.current_scene_pcd = PointCloud(0)
        self.current_manipuland_pcd = PointCloud(0)
        if not sys_id_grasp_only:   
            self.DeclareAbstractInputPort("cloud_bin", AbstractValue.Make(PointCloud(0)))
        self.DeclareAbstractInputPort("cloud_front", AbstractValue.Make(PointCloud(0)))
        self.DeclareAbstractInputPort("cloud_back_right", AbstractValue.Make(PointCloud(0)))
        self.DeclareAbstractInputPort("cloud_back_left", AbstractValue.Make(PointCloud(0)))
        self._X_WC0 = X_WC0
        self._X_WC1 = X_WC1
        self._X_WC2 = X_WC2
        self._X_WC_bin = X_WC_bin
        self._X_WC_obs = X_WC_obs
        self._cam_obs_K = cam_obs_K
        self.obs_width_px, self.obs_height_px = obs_width_px, obs_height_px
        self.objs_left = num_objs
        self.is_manual = is_manual
        self.is_semimanual = is_semimanual
        self.sys_id_grasp_only = sys_id_grasp_only
        self.bin_pick_grasped_nothing = False
        self.use_custom_path_planner = use_custom_path_planner
        self.voxel_map : VoxelMap = None

        # for getting current positions
        self._ee_index = plant.GetBodyByName("iiwa_link_7").index()
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )

        self._wsg_position_input_port = self.DeclareVectorInputPort(
            "wsg.position_measured", size=1
        )

        # for getting object observations
        self.DeclareAbstractInputPort(
            "object_pose", AbstractValue.Make(RigidTransform())
        )
        self.DeclareAbstractInputPort(
            "gripper_mask", AbstractValue.Make(ImageLabel16I())
        )
        # store observations in bank to update confidence model after full display trajectory
        self.object_poses = []
        self.gripper_masks = []

        # FSM state
        self._mode_index = self.DeclareAbstractState(
            AbstractValue.Make(PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE)
        )
        self._pick_mode_index = self.DeclareAbstractState(
            AbstractValue.Make(PickState.IDLE)
        )

        # Store calculated grasp pose
        self.X_WG1 = None
        self.X_WG2 = None

        # Store the path parameterized display trajectories
        self._display_traj_index1 = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )
        self._display_traj_index2 = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )
        self._display_traj_indexn = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )


        # Store planned grasp trajectories
        self._traj_X_G_index = self.DeclareAbstractState( # pose traj
            AbstractValue.Make(PiecewisePose())
        )
        self._traj_wsg_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )
        self._times_index_single_grasp = self.DeclareAbstractState(
            AbstractValue.Make({"initial": 0.0})
        )
        self._times_index1 = self.DeclareAbstractState(
            AbstractValue.Make({"initial": 0.0})
        )
        self._times_index2 = self.DeclareAbstractState(
            AbstractValue.Make({"initial": 0.0})
        )
        self._gripper_pose_index_single_grasp = self.DeclareAbstractState(
            AbstractValue.Make({"initial": RigidTransform()})
        )
        self._gripper_pose_index1 = self.DeclareAbstractState(
            AbstractValue.Make({"initial": RigidTransform()})
        )
        self._gripper_pose_index2 = self.DeclareAbstractState(
            AbstractValue.Make({"initial": RigidTransform()})
        )
        self._current_joint_traj_idx = self.DeclareAbstractState( # joint positions traj
            AbstractValue.Make(TrajectoryWithTimingInformation())
        )

        # output pose (for pick and display)
        self.DeclareAbstractOutputPort(
            "X_WG",
            lambda: AbstractValue.Make(RigidTransform()),
            self.CalcGripperPose,
        )
        self.DeclareVectorOutputPort(
            "wsg_position", 
            1, 
            self.CalcWsgPosition
        )

        # output joint position trajectory
        self.DeclareAbstractOutputPort(
            "joint_position_trajectory",
            lambda: AbstractValue.Make(TrajectoryWithTimingInformation()),
            self.GetCurrentJointPositionTrajectory,
        )

        # output state (for data saving)
        self.DeclareAbstractOutputPort(
            "planner_state", 
            lambda: AbstractValue.Make(PlannerState.START),
            self.GetState
        )
        self.DeclareAbstractOutputPort(
            "pick_state", 
            lambda: AbstractValue.Make(PickState.IDLE),
            self.GetPickState
        )

        # To get iiwa position
        num_positions = 7
        self._iiwa_position_index = self.DeclareVectorInputPort(
            "iiwa_position", num_positions
        ).get_index()
        
        self.DeclareAbstractOutputPort(
            "control_mode",
            lambda: AbstractValue.Make(InputPortIndex(0)),
            self.CalcControlMode,
        )
        self.DeclareAbstractOutputPort(
            "reset_diff_ik",
            lambda: AbstractValue.Make(False),
            self.CalcDiffIKReset,
        )

        self._q0_index = self.DeclareDiscreteState(num_positions)  # for q0
        self.default_home = default_home
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)

        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)

        self.grasp_node = GraspListener(gripper_length=gripper_length, gripper_model_path=gripper_model_path)
        self.q_pregrasp1 = None
        self.q_pregrasp2 = None
        self.q_pregraspn = None
        self.q_bin_pregrasp = None
        self.q_display_center1 = None
        self.q_display_center2 = None
        self.q_display_centern = None
        self.q_sys_id_pregrasp = None
        self.q_postgrasp1 = None
        self.q_postgrasp2 = None
        self.q_postgraspn = None
        self.q_postgrasp_bin = None
        self.place_flipped1 = False
        self.place_flipped2 = True
        self.meshcat = meshcat
        self.plant = plant
        self._iiwa_controller_plant = controller_plant
        self.fake_plant, self.fake_plant_context = make_iiwa_plant()
        self.velocity_limits = 0.4 * np.ones(7)
        self.acceleration_limits = 0.4 * np.ones(7)
        self.display_velocity_limits = 0.1 * np.ones(7)
        self.rotate_velocity_limits = 0.1 * np.ones(7)
        self.rotate_velocity_limits[6] = 0.4
        # self.rotate_velocity_limits[6] = 1.0 # Uncomment this for fast debug runs but bad scanning data
        self.display_acceleration_limits = 0.1 * np.ones(7)
        self.regions = [] #regions
        self.gripper_length = gripper_length
        self.pregrasp_dist = pregrasp_dist
        self.eef_to_gripper_length = eef_to_gripper_length
        self.joint_limits = np.zeros((7, 2))
        for i in range(7):
            joint = controller_plant.GetJointByName("iiwa_joint_%i" % (i + 1))
            self.joint_limits[i, 0] = joint.position_lower_limits()
            self.joint_limits[i, 1] = joint.position_upper_limits()
        self.ik_domain = HPolyhedron.MakeBox(controller_plant.GetPositionLowerLimits()+1e-2, 
                                        controller_plant.GetPositionUpperLimits()-1e-2)
        self.csd_plant = get_csd_plant()
        self.drm_planner = drm_planner(self.csd_plant)

        self.models_path = models_path
        self.savedir = dirstr
        self.system_id_savedir = os.path.join(dirstr, "system_id_data")
        if not os.path.exists(self.system_id_savedir):
            os.makedirs(self.system_id_savedir)
        self.done = False
        if sys_id_grasp_only:
            self.doing_sys_id = False

        self.pcd0 = None
        self.pcd1 = None
        self.pcd2 = None
        self.pcd3 = None
        self.bin_points = None

        # Load trajectory parameters
        is_fourier_series = os.path.exists(SYS_ID_TRAJ_PARAMETER_PATH / "a_value.npy")
        if is_fourier_series:
            traj_attrs = FourierSeriesTrajectoryAttributes.load(SYS_ID_TRAJ_PARAMETER_PATH)
            excitation_traj = FourierSeriesTrajectory(
                traj_attrs=traj_attrs,
                time_horizon=time_horizon,
            )
        else:
            traj_attrs = BsplineTrajectoryAttributes.load(SYS_ID_TRAJ_PARAMETER_PATH)
            excitation_traj = BsplineTrajectory(
                basis=BsplineBasis(order=traj_attrs.spline_order, knots=traj_attrs.knots),
                control_points=traj_attrs.control_points,
            )
        self.excitation_traj = excitation_traj
        self.excitation_traj_start_q = excitation_traj.value(0)


    def Update(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        current_time = context.get_time()

        if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            if current_time > 1.0:
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.START)
                self.GoHome(context, state)
            return
        if mode == PlannerState.START:
            traj_q = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().trajectory
            start_time = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().start_time_s
            if context.get_time() > traj_q.end_time() + start_time:
                if not self.sys_id_grasp_only:
                    state.get_mutable_abstract_state(
                        int(self._mode_index)
                    ).set_value(PlannerState.PLAN_PICK)
                else:
                    state.get_mutable_abstract_state(
                        int(self._mode_index)
                    ).set_value(PlannerState.PLAN_SYS_ID)
            return
        if mode == PlannerState.PLAN_PICK:
            self.PlanBinPick(context, state, PlannerState.GO_TO_PICK_PREGRASP)
            return
        if mode == PlannerState.GO_TO_PICK_PREGRASP:
            traj_q = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().trajectory
            start_time = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().start_time_s
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.PICK_GRASP)
                # Plan pick grasp
                self.PlanBinPickPoseTraj(context, state)
                self.bin_pick_grasped_nothing = False
            return
        if mode == PlannerState.PICK_GRASP:
            traj_pose = context.get_abstract_state(
                int(self._traj_X_G_index)
            ).get_value()
            if traj_pose.get_number_of_segments() > 0 and context.get_time() > traj_pose.end_time():
                if self.bin_pick_grasped_nothing:
                    state.get_mutable_abstract_state(
                        int(self._mode_index)
                    ).set_value(PlannerState.START)
                    self.GoHome(context, state)
                else:
                    state.get_mutable_abstract_state(
                        int(self._mode_index)
                    ).set_value(PlannerState.GO_HOME0)
                    self.GoHome(context, state)
            else:
                wsg_position = self.GetInputPort("wsg.position_measured").Eval(context)
                if wsg_position < 0.005: # try again if nothing was grasped
                    self.bin_pick_grasped_nothing = True
            return
        if mode == PlannerState.GO_HOME0:
            traj_q = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().trajectory
            start_time = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().start_time_s
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.SCANNING1)
            return
        if mode == PlannerState.SCANNING1:
            self.GetPointCloud(context, state, PlannerState.GO_TO_PREGRASP1)
            # input("Next: Pregrasp 1 (scs)") # pause for debugging
            return
        if mode == PlannerState.GO_TO_PREGRASP1:
            traj_q = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().trajectory
            start_time = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().start_time_s
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.GRASP1)
                # Update pick + display state
                state.get_mutable_abstract_state(
                    int(self._pick_mode_index)
                ).set_value(PickState.PICK)
                self.DoPoseTraj(context, state)
                # input("Next: Pick 1 (DiffIK)") # pause for debugging
            return
        if mode == PlannerState.GRASP1:
            self.UpdateInGrasp(context, state, PlannerState.GO_HOME1)
            X_WO = self.GetInputPort("object_pose").Eval(context)
            self.object_poses.append(X_WO)
            gripper_mask = self.GetInputPort("gripper_mask").Eval(context)
            self.gripper_masks.append(gripper_mask)
            return
        if mode == PlannerState.GO_HOME1:
            traj_q = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().trajectory
            start_time = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().start_time_s
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.SCANNING2)
            return
        if mode == PlannerState.SCANNING2:
            self.GetPointCloud(context, state, PlannerState.GO_TO_PREGRASP2)
            # input("Next: Pregrasp 2 (scs)") # pause for debugging
            return
        if mode == PlannerState.GO_TO_PREGRASP2:
            traj_q = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().trajectory
            start_time = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().start_time_s
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.GRASP2)
                # Update pick + display state
                state.get_mutable_abstract_state(
                    int(self._pick_mode_index)
                ).set_value(PickState.PICK)
                self.DoPoseTraj(context, state)
                # input("Next: Pick 2 (DiffIK)") # pause for debugging
            return
        if mode == PlannerState.GRASP2:
            self.UpdateInGrasp(context, state, PlannerState.GO_HOME2)
            X_WO = self.GetInputPort("object_pose").Eval(context)
            self.object_poses.append(X_WO)
            gripper_mask = self.GetInputPort("gripper_mask").Eval(context)
            self.gripper_masks.append(gripper_mask)
            return
        if mode == PlannerState.GO_HOME2:
            traj_q = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().trajectory
            start_time = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().start_time_s
            if context.get_time() > traj_q.end_time() + start_time:
                # Add observations to voxel map
                for i in range(len(self.object_poses)):
                    X_WO = self.object_poses[i]
                    gripper_mask = self.gripper_masks[i]

                    X_OC = X_WO.inverse() @ self._X_WC_obs

                    self.voxel_map.update_with_observation(
                        self._cam_obs_K, 
                        X_OC.GetAsMatrix4(), 
                        self.obs_width_px, self.obs_height_px, 
                        occlusion_mask=gripper_mask.get_image(), 
                        visualize=False,
                        visualize_all=False
                    )
                
                # Reset object poses and gripper masks
                self.object_poses = []
                self.gripper_masks = []

                confident = self.voxel_map.fully_observed()
                if confident:
                    state.get_mutable_abstract_state(
                        int(self._mode_index)
                    ).set_value(PlannerState.PLAN_SYS_ID)
                else:
                    state.get_mutable_abstract_state(
                        int(self._mode_index)
                    ).set_value(PlannerState.SCANNINGN)
            return
        if mode == PlannerState.SCANNINGN:
            self.PlanRegraspPick(context, state, PlannerState.GO_TO_PREGRASPN)
            return
        if mode == PlannerState.GO_TO_PREGRASPN:
            traj_q = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().trajectory
            start_time = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().start_time_s
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.GRASPN)
                # Update pick + display state
                state.get_mutable_abstract_state(
                    int(self._pick_mode_index)
                ).set_value(PickState.PICK)
                self.DoPoseTraj(context, state)
                # input("Next: Pick 2 (DiffIK)") # pause for debugging
            return
        if mode == PlannerState.GRASPN:
            self.UpdateInGrasp(context, state, PlannerState.GO_HOMEN)
            X_WO = self.GetInputPort("object_pose").Eval(context)
            self.object_poses.append(X_WO)
            gripper_mask = self.GetInputPort("gripper_mask").Eval(context)
            self.gripper_masks.append(gripper_mask)
            return
        if mode == PlannerState.GO_HOMEN:
            traj_q = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().trajectory
            start_time = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().start_time_s
            if context.get_time() > traj_q.end_time() + start_time:
                # Add observations to voxel map
                for i in range(len(self.object_poses)):
                    X_WO = self.object_poses[i]
                    gripper_mask = self.gripper_masks[i]

                    X_OC = X_WO.inverse() @ self._X_WC_obs

                    self.voxel_map.update_with_observation(
                        self._cam_obs_K, 
                        X_OC.GetAsMatrix4(), 
                        self.obs_width_px, self.obs_height_px, 
                        occlusion_mask=gripper_mask.get_image(), 
                        visualize=False,
                        visualize_all=False
                    )
                
                # Reset object poses and gripper masks
                self.object_poses = []
                self.gripper_masks = []

                confident = self.voxel_map.fully_observed()
                if confident:
                    state.get_mutable_abstract_state(
                        int(self._mode_index)
                    ).set_value(PlannerState.PLAN_SYS_ID)
                else:
                    state.get_mutable_abstract_state(
                        int(self._mode_index)
                    ).set_value(PlannerState.SCANNINGN)
            return
        if mode == PlannerState.PLAN_SYS_ID:
            self.PlanSysIdPick(context, state, PlannerState.GO_TO_SYS_ID_PREGRASP)
            return
        if mode == PlannerState.GO_TO_SYS_ID_PREGRASP:
            traj_q = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().trajectory
            start_time = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().start_time_s
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.SYS_ID_GRASP)
                # Update pick + display state
                state.get_mutable_abstract_state(
                    int(self._pick_mode_index)
                ).set_value(PickState.PICK)
                self.DoPoseTraj(context, state)
            return
        if mode == PlannerState.SYS_ID_GRASP:
            self.UpdateInGrasp(context, state, PlannerState.GO_HOME3)
            return
        if mode == PlannerState.GO_HOME3:
            traj_q = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().trajectory
            start_time = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().start_time_s
            if context.get_time() > traj_q.end_time() + start_time:
                self.objs_left -= 1
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.RESET)
            return
        if mode == PlannerState.RESET:
            if self.objs_left > 0:
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.START)
            else:
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.DONE)
                self.done = True
            return
        
    def UpdateInGrasp(self, context, state, after_grasp_state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        sys_id = False
        if mode == PlannerState.SYS_ID_GRASP:
            sys_id = True
        pick_mode = context.get_abstract_state(int(self._pick_mode_index)).get_value()
        traj_q = context.get_abstract_state(
            int(self._current_joint_traj_idx)
        ).get_value().trajectory
        start_time = context.get_abstract_state(
            int(self._current_joint_traj_idx)
        ).get_value().start_time_s
        traj_pose = context.get_abstract_state(
            int(self._traj_X_G_index)
        ).get_value()

        if pick_mode == PickState.PICK:
            if traj_pose.get_number_of_segments() > 0 and context.get_time() > traj_pose.end_time():
                state.get_mutable_abstract_state(
                    int(self._pick_mode_index)
                ).set_value(PickState.TO_DISPLAY)
                if sys_id:
                    self.GoToSysID(context, state)
                else:
                    self.GoToDisplay(context, state)
                # input("Next: GoToDisplay (scs)") # pause for debugging
        if pick_mode == PickState.TO_DISPLAY:
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._pick_mode_index)
                ).set_value(PickState.DISPLAY)
                if sys_id:
                    if self.sys_id_grasp_only:
                        self.doing_sys_id = True
                        return
                    self.DoSysID(context, state)
                else:
                    self.DoDisplay(context, state)
                # input("Next: DoDisplay (IK + Toppra)") # pause for debugging
        if pick_mode == PickState.DISPLAY:
            if (self.sys_id_grasp_only and not self.doing_sys_id) or (context.get_time() > traj_q.end_time() + start_time):
                state.get_mutable_abstract_state(
                    int(self._pick_mode_index)
                ).set_value(PickState.TO_PLACE)
                self.GoToPlace(context, state)
                # input("Next: GoToPreplace (scs)") # pause for debugging
        if pick_mode == PickState.TO_PLACE:
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._pick_mode_index)
                ).set_value(PickState.PLACE)
                print("placing")
                self.DoPoseTraj(context, state)
                # input("Next: Place (DiffIK)") # pause for debugging
        if pick_mode == PickState.PLACE:
            if traj_pose.get_number_of_segments() > 0 and context.get_time() > traj_pose.end_time():
                state.get_mutable_abstract_state(
                    int(self._pick_mode_index)
                ).set_value(PickState.IDLE)
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(after_grasp_state)
                self.GoHome(context, state)
                # input("Next: Postgrasp (scs)") # pause for debugging
        return

    def PlanBinPick(self, context, state, after_scan_state):
        # Get pcd 
        cloud = self.GetInputPort("cloud_bin").Eval(context)
        X_adjust = RigidTransform(RotationMatrix(RollPitchYaw(0.0, 0.0, 0.0)),[0.017, 0.003, 0.0])
        transformed_xyzs = X_adjust @ cloud.xyzs()
        cloud.mutable_xyzs()[:] = transformed_xyzs
        bin_pcd = cloud.Crop(lower_xyz=[-0.03, 0.44, -bin_depth], upper_xyz=[0.145, 0.75, 0.2])
        bin_pcd.EstimateNormals(radius=0.1, num_closest=30)
        bin_pcd.FlipNormalsTowardPoint(self._X_WC_bin.translation())
        bin_pcd = bin_pcd.VoxelizedDownSample(voxel_size=ONLINE_VOXEL_RADIUS)

        # Get background pcd 
        scene_pcd = cloud.Crop(lower_xyz=[-0.05, 0.42, -bin_depth-0.1], upper_xyz=[0.165, 0.77, 0.2])
        scene_pcd.EstimateNormals(radius=0.1, num_closest=30)
        scene_pcd.FlipNormalsTowardPoint(self._X_WC_bin.translation())
        scene_pcd = scene_pcd.VoxelizedDownSample(voxel_size=ONLINE_VOXEL_RADIUS)
        new_scene_pts = np.concatenate([scene_pcd.xyzs(), bin_sides_vox, bin_bottom_box], axis=1)
        scene_pcd.resize(new_scene_pts.shape[1])
        scene_pcd.mutable_xyzs()[:] = new_scene_pts
        
        # np.save(os.path.abspath(os.path.join(__file__ ,"../../bin_background.npy")), scene_pcd.xyzs())
        # np.save(os.path.abspath(os.path.join(__file__ ,"../../bin_contents.npy")), bin_pcd.xyzs())
        # np.save(os.path.abspath(os.path.join(__file__ ,"../../bin_background_normals.npy")), scene_pcd.normals())
        # np.save(os.path.abspath(os.path.join(__file__ ,"../../bin_contents_normals.npy")), bin_pcd.normals())
        # print("saved bin pcds")

        # remove outliers
        o3d_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bin_pcd.xyzs().T))
        try:
            cl, ind = o3d_cloud.remove_statistical_outlier(
                nb_neighbors=int(bin_pcd.xyzs().shape[1] // 20), 
                std_ratio=0.8)
            filtered_pts = np.asarray(o3d_cloud.points)[ind]
            bin_pcd.resize(filtered_pts.shape[0])
            bin_pcd.mutable_xyzs()[:] = filtered_pts.T
        except:
            pass
        
        self.meshcat.SetObject("bin_cloud", scene_pcd, point_size=0.001, rgba=Rgba(0,1,0,1))
        self.meshcat.SetObject("bin_contents_cloud", bin_pcd, point_size=0.001, rgba=Rgba(1,0,1,1))
        bin_cam_pcd = PointCloud(bin_cam_vox.shape[1])
        bin_cam_pcd.mutable_xyzs()[:] = bin_cam_vox
        self.meshcat.SetObject("bin_cam", bin_cam_pcd, point_size=0.001, rgba=Rgba(1,1,1,1))
        bin_cam_pole_pcd = PointCloud(bin_cam_pole_vox.shape[1])
        bin_cam_pole_pcd.mutable_xyzs()[:] = bin_cam_pole_vox
        self.meshcat.SetObject("bin_cam_pole", bin_cam_pole_pcd, point_size=0.001, rgba=Rgba(1,1,1,1))
        ceiling_pcd = PointCloud(ceiling_vox.shape[1])
        ceiling_pcd.mutable_xyzs()[:] = ceiling_vox
        self.meshcat.SetObject("ceiling_pcd", ceiling_pcd, point_size=0.001, rgba=Rgba(1,1,1,1))
        camera_pcd = PointCloud(camera_vox.shape[1])
        camera_pcd.mutable_xyzs()[:] = camera_vox
        self.meshcat.SetObject("camera_pcd", camera_pcd, point_size=0.001, rgba=Rgba(1,1,1,1))
        scene_pcd_downsampled = scene_pcd.VoxelizedDownSample(voxel_size=ONLINE_VOXEL_RADIUS*2)
        self.bin_points = scene_pcd_downsampled.xyzs()

        # get end effector pose from grasp pose
        X_GE = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 0)), [0, 0, -self.eef_to_gripper_length])

        # pregrasp is negative z in the gripper frame
        X_GgraspGpregrasp = RigidTransform([0, 0.0, -self.pregrasp_dist])

        # Initialize DRM with current pcd
        self.drm_planner = drm_planner(self.csd_plant, self.current_manipuland_pcd.xyzs().T)

        q_goal = None
        first_attempt = True
        while q_goal is None:
            # Planning first grasping trajectory
            self.grasp_node.compute_candidate_grasps(
                bin_pcd,
                scene_pcd,
                candidate_num=1,
                num_samples=20 if first_attempt else 50,
                random_seed=np.random.randint(1000),
                grasp_type=GraspType.TOP,
                roll_min = 0.0,
                roll_max = 0.0,
                num_roll_samples=1,
                pitch_min = 0.0,
                pitch_max = 0.0,
                num_pitch_samples=1,
                num_yaw_samples=5,
                point_up=True,
                split_ratio_threshold=0.5,
                voxel_radius=ONLINE_VOXEL_RADIUS,
                is_manual=self.is_manual
            )

            grasps, grasp_costs = self.grasp_node.get_best_grasps(-1 if first_attempt else 10)
            if not self.is_manual:
                grasps = copy.deepcopy(grasps)
                grasp_costs = copy.deepcopy(grasp_costs)
                while len(grasps) < 5:
                    print(f"not enough grasps ({len(grasps)}), sampling more points")
                    self.grasp_node.compute_candidate_grasps(
                        bin_pcd,
                        scene_pcd,
                        candidate_num=1,
                        num_samples=20,
                        random_seed=np.random.randint(1000),
                        grasp_type=GraspType.TOP,
                        roll_min = 0.0,
                        roll_max = 0.0,
                        num_roll_samples=1,
                        pitch_min = 0.0,
                        pitch_max = 0.0,
                        num_pitch_samples=1,
                        yaw_min = -np.pi / 2,
                        yaw_max = np.pi / 2,
                        num_yaw_samples=20,
                        point_up=True,
                        split_ratio_threshold=0.15,
                        voxel_radius=ONLINE_VOXEL_RADIUS
                    )
                    new_grasps, new_grasp_costs = self.grasp_node.get_best_grasps()
                    print("num new grasps:", len(new_grasps))
                    grasps += new_grasps
                    grasp_costs += new_grasp_costs

                    new_sorted_inds = np.argsort(np.array(grasp_costs))
                    grasps = [grasps[idx] for idx in new_sorted_inds]

            first_attempt = False
            print("bin grasps:", len(grasps))
            q = self.get_input_port(self._iiwa_position_index).Eval(context)
            for i in range(len(grasps)):
                # import IPython; IPython.embed()

                print("Solving IK for grasp:", grasps[i])

                X_WG = grasps[i]
                X_WG_eef = (RigidTransform(X_WG) @ X_GE)
                X_WPregrasp = RigidTransform(X_WG_eef.rotation(), X_WG_eef.translation() + [0, 0, 0.18])

                q_goal = solve_via_analytic_IK(
                    pose=X_WPregrasp,
                    current_config=q,
                    ik_domain=self.ik_domain,
                    csd_plant=self.csd_plant,
                )

                if q_goal is None:
                    X_WPregrasp = RigidTransform(X_WG_eef.rotation(), X_WG_eef.translation() + [0, 0, 0.14])
                    q_goal = solve_via_analytic_IK(
                        pose=X_WPregrasp,
                        current_config=q,
                        ik_domain=self.ik_domain,
                    csd_plant=self.csd_plant,
                    )

                if q_goal is None:
                    print("Failed to find IK, trying next best grasp.")
                    continue
                
                self.q_bin_pregrasp = q_goal
                X_WG_bin = RigidTransform(X_WG)

                # Crop point cloud around grasp point
                grasp_center = X_WG_bin @ RigidTransform([0, 0.0, self.gripper_length/2])
                closest_idx = np.argmin(np.linalg.norm(bin_pcd.xyzs() - grasp_center.translation()[:, np.newaxis], axis=0))
                cropped_cloud, _ = crop_connected_points(
                    bin_pcd,
                    bin_pcd.xyzs()[:, closest_idx],
                    radius=0.05,  # 10cm radius
                    voxel_radius=ONLINE_VOXEL_RADIUS
                )
                grasp_center_cloud = PointCloud(1)
                grasp_center_cloud.mutable_xyzs()[:] = np.array([grasp_center.translation()]).T
                self.meshcat.SetObject("grasp_center", grasp_center_cloud, point_size=0.01, rgba=Rgba(1,1,0,1))
                self.meshcat.SetObject("cropped_cloud", cropped_cloud, point_size=0.001, rgba=Rgba(0,0,1,1))

                manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bin_pcd.xyzs().T))
                manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

                gripper_xyzs = self.grasp_node.hand_collision_model.to_pcd()
                gripper_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(ONLINE_VOXEL_RADIUS).transform(
                    (X_WG @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4())
                )
                gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])

                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(X_WG)

                viz_geoms = [manipuland_cloud, gripper_cloud, frame]
                o3d.visualization.draw_plotly(viz_geoms)

                if self.is_semimanual:
                    reject = input("Enter 'n' to reject this grasp, press Enter to continue ")
                    if reject != 'n':
                        break
                else:
                    break

        # Flatten cloud to determine what way to place object after bin picking. We need to place it in a
        # way such that good scanning grasps are kinematically feasible.
        flattened_cloud = np.copy(cropped_cloud.xyzs())
        flattened_cloud[2,:] = np.clip(
            flattened_cloud[2,:],
            np.max(flattened_cloud[2,:]) - 0.001,
            np.max(flattened_cloud[2,:])
        )
        principal_component, secondary_component, _ = compute_principal_minor_components(flattened_cloud.T)
        
        # visualize axes, principal axis is z axis (blue), secondary axis is x axis (red)
        z_axis, x_axis = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
        rot_principal_component_to_axes, _ = R.align_vectors(
            np.array([z_axis, x_axis]), np.stack([principal_component, secondary_component])
        )
        com = np.mean(cropped_cloud.xyzs().T, axis=0)
        rot = RotationMatrix(rot_principal_component_to_axes.as_matrix().T)
        AddMeshcatTriad(self.meshcat, "principal axis", 
                        X_PT=RigidTransform(rot,
                        [com[0], com[1], com[2]]))
        rot_mat = X_WG_bin.GetAsMatrix4()[:3,:3]
        eff_perpendicular_vec = rot_mat.dot(np.array([1, 0, 0]))
        eff_parallel_vec = rot_mat.dot(np.array([0, 1, 0]))
        perpendicular_score = np.abs(eff_perpendicular_vec @ principal_component)
        parallel_score = np.abs(eff_parallel_vec @ principal_component)
        
        stage_center = stage_center0
        if perpendicular_score > parallel_score:
            stage_center = stage_center0
        else:
            stage_center = stage_center90

        # Solve for pick trajectory before moving
        X_WE = X_WG_bin.multiply(X_GE)
        pick_height = X_WE.translation()[2]
        
        place_t = np.copy(stage_center.translation())
        place_t[2] = pick_height + bin_depth + platform_height
        place = RigidTransform(stage_center.rotation(), place_t)
        X_G = {
            "pick": X_WE,
            "prepick": X_WE @ X_GgraspGpregrasp,
            "place": place,
            "postplace": place @ RigidTransform([0, 0.0, -0.2]),
        }

        X_G, times = MakePickGripperFrames(X_G, 8.0, 0.35)
        
        state.get_mutable_abstract_state(int(self._times_index_single_grasp)).set_value(
            times
        )
        state.get_mutable_abstract_state(int(self._gripper_pose_index_single_grasp)).set_value(
            X_G
        )

        self.PlanToPregrasp(context, state)
        state.get_mutable_abstract_state(
            int(self._mode_index)
        ).set_value(after_scan_state)
    
    def PlanSysIdPick(self, context, state, after_scan_state):
        self.GetPointCloud(context, state, after_scan_state)
        self.meshcat.SetObject("cloud", self.current_manipuland_pcd, point_size=0.001, rgba=Rgba(0,0,1,1))
        self.meshcat.SetObject("scene_cloud", self.current_scene_pcd, point_size=0.001, rgba=Rgba(0,1,0,1))
        pcd_with_background = Concatenate([self.current_manipuland_pcd, self.current_scene_pcd])

        # get end effector pose from grasp pose
        X_GE = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 0)), [0, 0, -self.eef_to_gripper_length])

        # pregrasp is negative z in the gripper frame
        X_GgraspGpregrasp = RigidTransform([0, 0.0, -self.pregrasp_dist])

        # Initialize DRM with current pcd
        self.drm_planner = drm_planner(self.csd_plant, self.current_manipuland_pcd.xyzs().T)
        
        self.grasp_node.compute_candidate_grasps(
            self.current_manipuland_pcd,
            pcd_with_background,
            candidate_num=1,
            num_samples=30,
            random_seed=np.random.randint(1000),
            grasp_type=GraspType.STABLE,
            split_ratio_threshold=0.15,
            is_manual=self.is_manual,
        )

        grasps, grasp_costs = self.grasp_node.get_best_grasps()
        if not self.is_manual:
            grasps = copy.deepcopy(grasps)
            grasp_costs = copy.deepcopy(grasp_costs)
            while len(grasps) < 5:
                print(f"not enough grasps ({len(grasps)}), sampling more points")
                self.grasp_node.compute_candidate_grasps(
                    self.current_manipuland_pcd,
                    pcd_with_background,
                    candidate_num=1,
                    num_samples=30,
                    random_seed=np.random.randint(1000),
                    grasp_type=GraspType.STABLE,
                    split_ratio_threshold=0.5
                )
                new_grasps, new_grasp_costs = self.grasp_node.get_best_grasps()
                print("num new grasps:", len(new_grasps))
                grasps += new_grasps
                grasp_costs += new_grasp_costs

                new_sorted_inds = np.argsort(np.array(grasp_costs))
                grasps = [grasps[idx] for idx in new_sorted_inds]

        print("grasps:", len(grasps))
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        for i in range(len(grasps)):
            print("Grasp:", grasps[i])

            X_WG = grasps[i]
            X_WPregrasp = (RigidTransform(X_WG) @ X_GE) @ X_GgraspGpregrasp

            q_goal = solve_via_analytic_IK_with_retries(
                pose=X_WPregrasp,
                current_config=q,
                ik_domain=self.ik_domain,
                csd_plant=self.csd_plant,
            )

            if q_goal is None:
                continue
            
            self.q_sys_id_pregrasp = q_goal
            X_WG_sys_id = RigidTransform(X_WG)

            X_WE = X_WG_sys_id.multiply(X_GE) # E = link 7 frame
            X_EW = X_WE.inverse()

            # Save manipuland pcd and grasp for system ID alignment.
            manipuland_cloud_points = self.current_manipuland_pcd.xyzs() # X_WP, Shape (3, N)
            manipuland_cloud_points_link7_frame = X_EW @ manipuland_cloud_points # X_EP, Shape (3, N)
            manipuland_cloud_points_link7_frame = manipuland_cloud_points_link7_frame.T # Shape (N,3)
            np.save(
                os.path.join(self.system_id_savedir, "manipuland_cloud_link7_frame.npy"),
                manipuland_cloud_points_link7_frame,
            )

            manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.current_manipuland_pcd.xyzs().T))
            manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

            gripper_xyzs = self.grasp_node.hand_collision_model.to_pcd()
            gripper_cloud = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(gripper_xyzs)
            ).voxel_down_sample(ONLINE_VOXEL_RADIUS).transform(
                (X_WG @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4()))
            gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])

            viz_geoms = [manipuland_cloud, gripper_cloud]
            o3d.visualization.draw_plotly(viz_geoms)

            if self.is_semimanual:
                reject = input("Enter 'n' to reject this grasp, press Enter to continue ")
                if reject != 'n':
                    break
            else:
                break

        # Solve for pick trajectory before moving
        X_G = {
            "pick": X_WE,
            "prepick": X_WE @ X_GgraspGpregrasp,
            "place": end_bin,
            "postplace": end_bin,
        }

        X_G, times = MakePickGripperFrames(X_G, preplace_time=0.5)
        
        state.get_mutable_abstract_state(int(self._times_index_single_grasp)).set_value(
            times
        )
        state.get_mutable_abstract_state(int(self._gripper_pose_index_single_grasp)).set_value(
            X_G
        )

        q_home = context.get_discrete_state(self._q0_index).get_value().copy()
        self.q_postgrasp_bin = solve_via_analytic_IK_with_retries(
            pose=X_G["preplace"],
            current_config=q_home,
            ik_domain=self.ik_domain,
            csd_plant=self.csd_plant,
        )

        self.PlanToPregrasp(context, state)
        state.get_mutable_abstract_state(
            int(self._mode_index)
        ).set_value(after_scan_state)
    
    def PlanRegraspPick(self, context, state, after_scan_state):
        self.GetPointCloud(context, state, after_scan_state)
        self.meshcat.SetObject("cloud", self.current_manipuland_pcd, point_size=0.001, rgba=Rgba(0,0,1,1))
        self.meshcat.SetObject("scene_cloud", self.current_scene_pcd, point_size=0.001, rgba=Rgba(0,1,0,1))
        pcd_with_background = Concatenate([self.current_manipuland_pcd, self.current_scene_pcd])

        # get end effector pose from grasp pose
        X_GE = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 0)), [0, 0, -self.eef_to_gripper_length])

        # pregrasp is negative z in the gripper frame
        X_GgraspGpregrasp = RigidTransform([0, 0.0, -self.pregrasp_dist])

        # Initialize DRM with current pcd
        self.drm_planner = drm_planner(self.csd_plant, self.current_manipuland_pcd.xyzs().T)

        self.grasp_node.compute_candidate_grasps(
            self.current_manipuland_pcd,
            pcd_with_background,
            candidate_num=1,
            num_samples=30,
            random_seed=np.random.randint(1000),
            grasp_type=GraspType.ADDITIONAL,
            split_ratio_threshold=0.15,
            ground_z=platform_height,
            is_manual=self.is_manual,
            use_extra_buffer=False,
            intrinsic_matrix=self._cam_obs_K,
            width_px=self.obs_width_px,
            height_px=self.obs_height_px,
            display_traj_height_buffer=display_traj_height_buffer,
        )

        grasps, grasp_costs = self.grasp_node.get_best_grasps()
        if not self.is_manual:
            grasps = copy.deepcopy(grasps)
            grasp_costs = copy.deepcopy(grasp_costs)
            while len(grasps) < 5:
                print(f"not enough grasps ({len(grasps)}), sampling more points")
                self.grasp_node.compute_candidate_grasps(
                    self.current_manipuland_pcd,
                    pcd_with_background,
                    candidate_num=1,
                    num_samples=30,
                    random_seed=np.random.randint(1000),
                    grasp_type=GraspType.ADDITIONAL,
                    split_ratio_threshold=0.15,
                    ground_z=platform_height,
                    is_manual=self.is_manual,
                    use_extra_buffer=False,
                    intrinsic_matrix=self._cam_obs_K,
                    width_px=self.obs_width_px,
                    height_px=self.obs_height_px,
                    display_traj_height_buffer=display_traj_height_buffer,
                )
                new_grasps, new_grasp_costs = self.grasp_node.get_best_grasps()
                print("num new grasps:", len(new_grasps))
                grasps += new_grasps
                grasp_costs += new_grasp_costs

                new_sorted_inds = np.argsort(np.array(grasp_costs))
                grasps = [grasps[idx] for idx in new_sorted_inds]

        print("grasps:", len(grasps))
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        for i in range(len(grasps)):
            print("Grasp:", grasps[i])

            X_WG = grasps[i]
            X_WPregrasp = (RigidTransform(X_WG) @ X_GE) @ X_GgraspGpregrasp

            q_goal = solve_via_analytic_IK_with_retries(
                pose=X_WPregrasp,
                current_config=q,
                ik_domain=self.ik_domain,
                csd_plant=self.csd_plant,
            )

            if q_goal is None:
                continue
            
            self.q_pregraspn = q_goal

            X_WE = RigidTransform(X_WG).multiply(X_GE) # E = link 7 frame

            manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.current_manipuland_pcd.xyzs().T))
            manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

            gripper_xyzs = self.grasp_node.hand_collision_model.to_pcd()
            gripper_cloud = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(gripper_xyzs)
            ).voxel_down_sample(ONLINE_VOXEL_RADIUS).transform(
                (X_WG @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4()))
            gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])

            viz_geoms = [manipuland_cloud, gripper_cloud]
            o3d.visualization.draw_plotly(viz_geoms)

            if self.is_semimanual:
                reject = input("Enter 'n' to reject this grasp, press Enter to continue ")
                if reject != 'n':
                    break
            else:
                break

        X_G = {
            "pick": X_WE,
            "prepick": X_WE @ X_GgraspGpregrasp
        }

        height = max(self.CalcLink7ToManipulandEndLength(X_WE) + display_traj_height_buffer, 0.35)
        X_G["display_traj"] = get_yaw_display_traj(scanning_traj_height=height)


        X_G, times = MakePickAndDisplayGripperFrames(
            X_G, self.gripper_length, self.pregrasp_dist,
            self.place_flipped1, X_GE, lift_for_display_height
        )

        display_traj, self.q_display_centern = MakeDisplayJointPositionsTrajectory(
            X_G,
            times, 
            self.q_pregrasp1,
            self.ik_domain,
            self.csd_plant
        )
        
        state.get_mutable_abstract_state(int(self._times_index_single_grasp)).set_value(
            times
        )
        state.get_mutable_abstract_state(int(self._gripper_pose_index_single_grasp)).set_value(
            X_G
        )
        state.get_mutable_abstract_state(self._display_traj_indexn).set_value(display_traj)

        self.PlanToPregrasp(context, state)
        state.get_mutable_abstract_state(
            int(self._mode_index)
        ).set_value(after_scan_state)

    def PlanBinPickPoseTraj(self, context, state):
        current_time = context.get_time()

        X_G = context.get_abstract_state(int(self._gripper_pose_index_single_grasp)).get_value()
        times = context.get_abstract_state(int(self._times_index_single_grasp)).get_value()
        
        X_G0 = self.GetInputPort("body_poses").Eval(context)[int(self._ee_index)]
        traj_wsg_command = MakeGripperCommandTrajectory(times, TrajType.BIN, current_time)
        traj_gripper_pose = MakeGripperPoseTrajectory(X_G, times, X_G0, TrajType.BIN, current_time)
        
        state.get_mutable_abstract_state(int(self._traj_wsg_index)).set_value(
            traj_wsg_command
        )

        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(
            traj_gripper_pose
        )

    def GetPointCloud(self, context, state, after_scan_state):
        start = time.time()
        # Get manipuland pcd 
        cloud0 = self.GetInputPort("cloud_front").Eval(context)
        X_adjust0 = RigidTransform(RotationMatrix(RollPitchYaw(0.0, 0.0, 0.0)),[0.005, -0.025, -0.015])
        transformed_xyzs = X_adjust0 @ cloud0.xyzs()
        cloud0.mutable_xyzs()[:] = transformed_xyzs
        pcd0 = cloud0.Crop(lower_xyz=[0.23, -0.17, platform_height], upper_xyz=[0.7, 0.17, 0.35])
        pcd0.EstimateNormals(radius=0.1, num_closest=30)
        pcd0.FlipNormalsTowardPoint(self._X_WC0.translation())

        cloud1 = self.GetInputPort("cloud_back_left").Eval(context)
        X_adjust1 = RigidTransform(RotationMatrix(RollPitchYaw(0.0, 0.0, 0.0)),[0.00, 0.0, -0.015])
        transformed_xyzs = X_adjust1 @ cloud1.xyzs()
        cloud1.mutable_xyzs()[:] = transformed_xyzs
        pcd1 = cloud1.Crop(lower_xyz=[0.23, -0.17, platform_height], upper_xyz=[0.7, 0.17, 0.35])
        pcd1.EstimateNormals(radius=0.1, num_closest=30)
        pcd1.FlipNormalsTowardPoint(self._X_WC2.translation()) # I screwed up the cameras somewhere so the left anf right are flipped, need to fix

        cloud2 = self.GetInputPort("cloud_back_right").Eval(context)
        X_adjust2 = RigidTransform(RotationMatrix(RollPitchYaw(0.0, 0.0, 0.0)),[0.03, 0.005, 0.005])
        transformed_xyzs = X_adjust2 @ cloud2.xyzs()
        cloud2.mutable_xyzs()[:] = transformed_xyzs
        pcd2 = cloud2.Crop(lower_xyz=[0.23, -0.17, platform_height], upper_xyz=[0.7, 0.17, 0.35])
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
        
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        # Use RANSAC to find transformation from original pose during second grasp
        if mode == PlannerState.SCANNING2:
            transformation = icp(self.current_manipuland_pcd.xyzs(), down_sampled_pcd.xyzs(), ONLINE_VOXEL_RADIUS)
            print("Received transformation:", transformation)
            print("Inverted transformation:", np.linalg.inv(transformation))

            # visualize old manipuland cloud
            old_manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.current_manipuland_pcd.xyzs().T))
            old_manipuland_cloud.paint_uniform_color([0.0, 1.0, 1.0]) # Cyan
            print("Initial manipuland center:", old_manipuland_cloud.get_center())

            # Visualize transformed old manipuland cloud
            transformed_manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.current_manipuland_pcd.xyzs().T))
            transformed_manipuland_cloud.transform(transformation)
            transformed_manipuland_cloud.paint_uniform_color([1.0, 0.0, 0.0]) # Red
            print("Transformed manipuland center:", transformed_manipuland_cloud.get_center())

            # Visualize updated manipuland cloud
            manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(down_sampled_pcd.xyzs().T))
            manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0]) # Blue
            print("Updated manipuland center:", manipuland_cloud.get_center())

            # Roundabout transformation
            numpy_transformed_manipuland_points = self.current_manipuland_pcd.xyzs().T
            numpy_transformed_manipuland_points = numpy_transformed_manipuland_points - old_manipuland_cloud.get_center()
            numpy_transformed_manipuland_points = numpy_transformed_manipuland_points @ transformation[:3, :3].T
            numpy_transformed_manipuland_points = numpy_transformed_manipuland_points + old_manipuland_cloud.get_center()
            numpy_transformed_manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(numpy_transformed_manipuland_points))
            translation_diff = transformed_manipuland_cloud.get_center() - numpy_transformed_manipuland_cloud.get_center()
            numpy_transformed_manipuland_points = numpy_transformed_manipuland_points + translation_diff
            numpy_transformed_manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(numpy_transformed_manipuland_points))
            numpy_transformed_manipuland_cloud.paint_uniform_color([1.0, 0.0, 1.0]) # Magenta
            
            # Check if transformation is too large
            translation_magnitude = np.linalg.norm(translation_diff)
            rotation_magnitude = np.arccos((np.trace(transformation[:3, :3]) - 1) / 2)
            print(translation_magnitude, rotation_magnitude)
            
            old_gripper2_xyzs = self.grasp_node.hand_collision_model.to_pcd()
            old_gripper2_cloud = (
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(old_gripper2_xyzs))
                .voxel_down_sample(ONLINE_VOXEL_RADIUS)
                .transform(
                    (self.X_WG2.multiply(
                        RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2), [0,0,0])
                    )).GetAsMatrix4()
                )
            )
            old_gripper2_cloud.paint_uniform_color([1.0, 1.0, 0.0])

            temp_X_WG2 = apply_centered_rotation_and_translation(
                self.X_WG2.GetAsMatrix4(), 
                transformation[:3, :3], 
                translation_diff, 
                old_manipuland_cloud.get_center()
            )
            temp_X_WG2 = RigidTransform(temp_X_WG2)
            gripper2_xyzs = self.grasp_node.hand_collision_model.to_pcd()
            gripper2_cloud = (
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper2_xyzs))
                .voxel_down_sample(ONLINE_VOXEL_RADIUS)
                .transform(
                    (temp_X_WG2.multiply(
                        RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2), [0,0,0])
                    )).GetAsMatrix4()
                )
            )
            gripper2_cloud.paint_uniform_color([0.0, 1.0, 0.0])

            # viz_geoms = [old_manipuland_cloud, old_gripper2_cloud, manipuland_cloud, gripper2_cloud]
            # o3d.visualization.draw_plotly(viz_geoms)
            # viz_geoms = [old_manipuland_cloud, transformed_manipuland_cloud, manipuland_cloud, numpy_transformed_manipuland_cloud]
            # o3d.visualization.draw_plotly(viz_geoms)

            if translation_magnitude > 0.01 or rotation_magnitude > np.pi * 10.0/180.0:
                print("Transformation is too large, realigning grasp 2")

                self.X_WG2 = temp_X_WG2
                
                # get end effector pose from grasp pose
                X_GE = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 0)), [0, 0, -self.eef_to_gripper_length])

                # pregrasp is negative z in the gripper frame
                X_GgraspGpregrasp = RigidTransform([0, 0.0, -self.pregrasp_dist])
                X_WPregrasp2 = (RigidTransform(self.X_WG2) @ X_GE) @ X_GgraspGpregrasp

                q_goal2 = solve_via_analytic_IK_with_retries(
                    pose=X_WPregrasp2,
                    current_config=self.q_pregrasp2,
                    ik_domain=self.ik_domain,
                    csd_plant=self.csd_plant,
                    retries=50
                )

                if q_goal2 is None:
                    print("Failed to solve IK for grasp 2")

                self.q_pregrasp2 = q_goal2
                self.PlanPickAndDisplay(context, state, skip_first=True)
                

        self.current_manipuland_pcd = down_sampled_pcd
        print("object pcd got in", time.time()-start, "seconds")

        if mode == PlannerState.SCANNING1:
            self.voxel_map = VoxelMap(
                self.current_manipuland_pcd, 
                ONLINE_VOXEL_RADIUS
            )
            self.grasp_node.set_voxel_map(self.voxel_map)

        start = time.time()
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
        self.current_scene_pcd = merged_pcd
        print("table pcd got in", time.time()-start, "seconds")
        
        # np.save(os.path.abspath(os.path.join(__file__ ,"../../dual_grasp_background_pcd.npy")), self.current_scene_pcd.xyzs())
        # np.save(os.path.abspath(os.path.join(__file__ ,"../../dual_grasp_pcd.npy")), self.current_manipuland_pcd.xyzs())
        # np.save(os.path.abspath(os.path.join(__file__ ,"../../dual_grasp_background_normals.npy")), self.current_scene_pcd.normals())
        # np.save(os.path.abspath(os.path.join(__file__ ,"../../dual_grasp_normals.npy")), self.current_manipuland_pcd.normals())
        # print("saved dual grasp pcds")

        # Find grasp candidates
        if mode == PlannerState.SCANNING1:
            self.PlanGrasp(context, state)
        if mode == PlannerState.PLAN_SYS_ID:
            return
        
        self.PlanToPregrasp(context, state)
        state.get_mutable_abstract_state(
            int(self._mode_index)
        ).set_value(after_scan_state)

        return

    def GoHome(self, context, state):
        '''
        Reset to default home position to move arm out of the way of the camera
        '''

        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        q_goal = context.get_discrete_state(self._q0_index).get_value().copy() # initial pose

        try:
            obstacles_vox = np.hstack([self.current_manipuland_pcd.xyzs()])

            if self.use_custom_path_planner:
                # Use custom path planner. Warning: Must be implemented by user
                traj = plan_path_custom(
                    q, 
                    q_goal,
                )
            else:
                # Use drm path planner
                traj = plan_drm(
                    self.drm_planner,
                    q, 
                    q_goal, 
                    obstacles_vox,
                    ONLINE_VOXEL_RADIUS
                )
        except:
            input("Press enter to continue with unconstrained plan to home. Else terminate.")
            traj = plan_unconstrained_gcs_path_start_to_goal(
                plant=self._iiwa_controller_plant, q_start=q, q_goal=q_goal, regions=None, no_obstacles=True
            )

        breaks = np.linspace(0, traj.end_time(), int(1e3), endpoint=False)
        knots = traj.vector_values(breaks)

        toppra_traj = reparameterize_with_toppra(
            trajectory=knots.T,
            plant=self._iiwa_controller_plant,
            velocity_limits=self.velocity_limits,
            acceleration_limits=self.acceleration_limits,
            num_grid_points=100,
        )

        current_time = context.get_time()
        state.get_mutable_abstract_state(self._current_joint_traj_idx).set_value(
            TrajectoryWithTimingInformation(
                trajectory=toppra_traj,
                start_time_s=current_time,
            )
        )

    def PlanToPregrasp(self, context, state):
        '''
        Determine grasp pose and plan trajectory using GCS to the pregrasp pose.
        '''
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        if mode == PlannerState.SCANNING1:
            q_goal = self.q_pregrasp1
            obstacles_vox = np.hstack([self.current_manipuland_pcd.xyzs()])
        if mode == PlannerState.SCANNING2:
            q_goal = self.q_pregrasp2
            obstacles_vox = np.hstack([self.current_manipuland_pcd.xyzs()])
        if mode == PlannerState.SCANNINGN:
            q_goal = self.q_pregraspn
            obstacles_vox = np.hstack([self.current_manipuland_pcd.xyzs()])
        if mode == PlannerState.PLAN_PICK:
            q_goal = self.q_bin_pregrasp
            print(self.bin_points.shape)
            obstacles_vox = np.hstack([self.bin_points])
        if mode == PlannerState.PLAN_SYS_ID:
            q_goal = self.q_sys_id_pregrasp
            obstacles_vox = np.hstack([self.current_manipuland_pcd.xyzs()])

        try:
            if self.use_custom_path_planner:
                # Use custom path planner. Warning: Must be implemented by user
                traj = plan_path_custom(
                    q, 
                    q_goal,
                )
            else:
                # Use drm path planner
                traj = plan_drm(
                    self.drm_planner,
                    q, 
                    q_goal, 
                    obstacles_vox,
                    ONLINE_VOXEL_RADIUS
                )
        except:
            return
        if traj is None:
            logging.error("Failed to find a path to the grasping start positions.")
            return

        breaks = np.linspace(0, traj.end_time(), int(1e3), endpoint=False)
        knots = traj.vector_values(breaks)

        toppra_traj = reparameterize_with_toppra(
            trajectory=knots.T,
            plant=self._iiwa_controller_plant,
            velocity_limits=self.velocity_limits,
            acceleration_limits=self.acceleration_limits,
        )

        current_time = context.get_time()
        state.get_mutable_abstract_state(self._current_joint_traj_idx).set_value(
            TrajectoryWithTimingInformation(
                trajectory=toppra_traj,
                start_time_s=current_time,
            )
        )

    def PlanGrasp(self, context, state):
        '''
        Determine grasp pose
        '''
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        # get end effector pose from grasp pose
        X_GE = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 0)), [0, 0, -self.eef_to_gripper_length])

        # pregrasp is negative z in the gripper frame
        X_GgraspGpregrasp = RigidTransform([0, 0.0, -self.pregrasp_dist])
        
        # if self.pcd0:
        #     self.meshcat.SetObject("cloud0", self.pcd0, point_size=0.0001, rgba=Rgba(1,0,0,1))
        # if self.pcd1:
        #     self.meshcat.SetObject("cloud1", self.pcd1, point_size=0.0001, rgba=Rgba(1,1,0,1))
        # if self.pcd2:
        #     self.meshcat.SetObject("cloud2", self.pcd2, point_size=0.0001, rgba=Rgba(0,1,0,1))
        # if self.pcd3:
        #     self.meshcat.SetObject("cloud3", self.pcd3, point_size=0.0001, rgba=Rgba(0,0,1,1))
        self.meshcat.SetObject("cloud", self.current_manipuland_pcd, point_size=0.001, rgba=Rgba(0,0,1,1))
        self.meshcat.SetObject("floor", self.current_scene_pcd, point_size=0.001)
        pcd_with_background = Concatenate([self.current_manipuland_pcd, self.current_scene_pcd])

        pcd_points = self.current_manipuland_pcd.xyzs().T
        principal_component, secondary_component, minor_component = compute_principal_minor_components(pcd_points)

        # visualize axes, principal axis is z axis (blue), minor axis is x axis (red)
        z_axis, x_axis = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
        rot_principal_component_to_axes, _ = R.align_vectors(
            np.array([z_axis, x_axis]), np.stack([principal_component, minor_component])
        )
        com = np.mean(pcd_points, axis=0)
        pcd_points_axis_aligned = pcd_points @ rot_principal_component_to_axes.as_matrix().T
        dims = np.max(pcd_points_axis_aligned, axis=0) - np.min(pcd_points_axis_aligned, axis=0)
        rot = RotationMatrix(rot_principal_component_to_axes.as_matrix().T)
        AddMeshcatTriad(self.meshcat, "principal axis", 
                        X_PT=RigidTransform(rot,
                        [com[0], com[1], com[2]]))
        

        # Initialize DRM with current pcd
        self.drm_planner = drm_planner(self.csd_plant, self.current_manipuland_pcd.xyzs().T)

        # while not grasps_found:
        # Planning first grasping trajectory
        self.grasp_node.compute_candidate_grasps(
            self.current_manipuland_pcd,
            pcd_with_background,
            candidate_num=1,
            num_samples=30,
            random_seed=np.random.randint(1000),
            grasp_type=GraspType.PAIR,
            split_ratio_threshold=0.15,
            ground_z=platform_height,
            is_manual=self.is_manual,
            use_extra_buffer=False
        )

        manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.current_manipuland_pcd.xyzs().T))
        manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

        VISUALIZE_FAIL = False
        VISUALIZE_FLIPPED = False

        grasp_pairs, grasp_costs = self.grasp_node.get_best_grasps()
        if not self.is_manual:
            grasp_pairs = copy.deepcopy(grasp_pairs)
            grasp_costs = copy.deepcopy(grasp_costs)
            while len(grasp_pairs) < 10:
                print("not enough pairs, sampling more points")
                self.grasp_node.compute_candidate_grasps(
                    self.current_manipuland_pcd,
                    pcd_with_background,
                    candidate_num=1,
                    num_samples=30,
                    random_seed=np.random.randint(1000),
                    grasp_type=GraspType.PAIR,
                    split_ratio_threshold=0.15,
                    ground_z=platform_height,
                    use_extra_buffer=False,
                )
                new_grasp_pairs, new_grasp_costs = self.grasp_node.get_best_grasps()
                print("num new grasps:", len(new_grasp_pairs))
                grasp_pairs += new_grasp_pairs
                grasp_costs += new_grasp_costs

                new_sorted_inds = np.argsort(np.array(grasp_costs))
                grasp_pairs = [grasp_pairs[idx] for idx in new_sorted_inds]

        print("grasp pairs:", len(grasp_pairs))
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        for i in range(len(grasp_pairs)):
            print("Trying to solve ik for grasp Pair:", grasp_pairs[i])

            X_WG1 = grasp_pairs[i][0]
            X_WG2 = grasp_pairs[i][1]
            X_WPregrasp1 = (RigidTransform(X_WG1) @ X_GE) @ X_GgraspGpregrasp
            X_WPregrasp2 = (RigidTransform(X_WG2) @ X_GE) @ X_GgraspGpregrasp

            q_goal1 = solve_via_analytic_IK_with_retries(
                pose=X_WPregrasp1,
                current_config=q,
                ik_domain=self.ik_domain,
                csd_plant=self.csd_plant
            )

            if q_goal1 is None:
                print("Failed to find ik for first grasp, trying next pair")

                if VISUALIZE_FAIL:
                    gripper1_xyzs = self.grasp_node.hand_extra_buffer_collision_model.to_pcd()
                    gripper1_cloud = o3d.geometry.PointCloud(
                        o3d.utility.Vector3dVector(gripper1_xyzs)).voxel_down_sample(ONLINE_VOXEL_RADIUS).transform(
                            (X_WG1 @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4()))
                    gripper1_cloud.paint_uniform_color([1.0, 0.0, 0.0])

                    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.05, origin=[0, 0, 0])
                    viz_geoms = [world_frame, manipuland_cloud, gripper1_cloud]
                    o3d.visualization.draw_plotly(viz_geoms)
                continue
                
            # check if configuration is in collision with scene
            if check_configuration_has_collisions(self.models_path, com, rot, dims, q_goal1):
                print("grasp 1 configuration has collision with scene")

                if VISUALIZE_FAIL:
                    gripper1_xyzs = self.grasp_node.hand_extra_buffer_collision_model.to_pcd()
                    gripper1_cloud = o3d.geometry.PointCloud(
                        o3d.utility.Vector3dVector(gripper1_xyzs)).voxel_down_sample(ONLINE_VOXEL_RADIUS).transform(
                            (X_WG1 @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4()))
                    gripper1_cloud.paint_uniform_color([1.0, 0.0, 0.0])

                    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.05, origin=[0, 0, 0])
                    viz_geoms = [world_frame, manipuland_cloud, gripper1_cloud]
                    o3d.visualization.draw_plotly(viz_geoms)
                    
                continue

            q_goal2 = solve_via_analytic_IK_with_retries(
                pose=X_WPregrasp2,
                current_config=q,
                ik_domain=self.ik_domain,
                csd_plant=self.csd_plant
            )

            if q_goal2 is None:
                print("Failed to find ik for second grasp, trying next pair")

                if VISUALIZE_FAIL:
                    gripper1_xyzs = self.grasp_node.hand_extra_buffer_collision_model.to_pcd()
                    gripper1_cloud = o3d.geometry.PointCloud(
                        o3d.utility.Vector3dVector(gripper1_xyzs)).voxel_down_sample(ONLINE_VOXEL_RADIUS).transform(
                            (X_WG2 @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4()))
                    gripper1_cloud.paint_uniform_color([1.0, 0.0, 0.0])

                    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.05, origin=[0, 0, 0])
                    viz_geoms = [world_frame, manipuland_cloud, gripper1_cloud]
                    o3d.visualization.draw_plotly(viz_geoms)
                    
                continue

            # check if configuration is in collision with scene
            if check_configuration_has_collisions(self.models_path, com, rot, dims, q_goal2):
                print("grasp 2 configuration has collision with scene")

                if VISUALIZE_FAIL:
                    gripper1_xyzs = self.grasp_node.hand_extra_buffer_collision_model.to_pcd()
                    gripper1_cloud = o3d.geometry.PointCloud(
                        o3d.utility.Vector3dVector(gripper1_xyzs)).voxel_down_sample(ONLINE_VOXEL_RADIUS).transform(
                            (X_WG2 @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4()))
                    gripper1_cloud.paint_uniform_color([1.0, 0.0, 0.0])

                    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.05, origin=[0, 0, 0])
                    viz_geoms = [world_frame, manipuland_cloud, gripper1_cloud]
                    o3d.visualization.draw_plotly(viz_geoms)
                continue

            # Check which grasp is more vertical (z component of z axis more negative), swap grasps if first is more vertical
            if X_WG1[2, 2] < X_WG2[2, 2]:
                temp_X_WG1 = X_WG1
                temp_q1 = q_goal1
                X_WG1 = X_WG2
                q_goal1 = q_goal2
                X_WG2 = temp_X_WG1
                q_goal2 = temp_q1

            if self.place_flipped1:
                print("Solving for flipped place 1")
                t_gripper_center = RigidTransform(X_WG1) @ [0, 0, 3*self.gripper_length/4] # assuming picking with only end half of gripper
                
                # Calculate rotation matrix for 180 degrees around z-axis
                R_z180 = np.array([
                    [-1, 0, 0],
                    [0, -1, 0], 
                    [0, 0, 1]
                ])
                
                # Get current rotation and translation
                R_current = X_WG1[:3, :3]
                t_current = X_WG1[:3, 3]
                
                # Apply rotation around z-axis centered at t_gripper_center
                t_centered = t_current - t_gripper_center
                t_rotated = R_z180 @ t_centered
                t_final = t_rotated + t_gripper_center
                
                # Create new rotation matrix combining current rotation with z-axis rotation
                R_final = R_z180 @ R_current
                
                # Set X_G["place"] to the rotated pose
                X_Gplace = RigidTransform(RotationMatrix(R_final), t_final)
                X_Eplace = X_Gplace.multiply(X_GE)
                X_Epreplace = RigidTransform(
                    X_Eplace.rotation(), X_Eplace.translation() + [0, 0, lift_for_display_height]
                )

                self.q_postgrasp1 = solve_via_analytic_IK_with_retries(
                    pose=X_Epreplace,
                    current_config=q,
                    ik_domain=self.ik_domain,
                    csd_plant=self.csd_plant
                )
                if self.q_postgrasp1 is None:
                    print("failed to solve for place flipped, trying next grasp pair")
                    continue

                if VISUALIZE_FLIPPED:
                    pick_gripper_xyzs = self.grasp_node.hand_collision_model.to_pcd()
                    pick_gripper_cloud = (
                        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pick_gripper_xyzs))
                        .voxel_down_sample(ONLINE_VOXEL_RADIUS)
                        .transform(
                            (RigidTransform(X_WG1).multiply(
                                RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2), [0,0,0])
                            )).GetAsMatrix4()
                        )
                    )
                    pick_gripper_cloud.paint_uniform_color([0.0, 1.0, 0.0])

                    place_gripper_xyzs = self.grasp_node.hand_collision_model.to_pcd()
                    place_gripper_cloud = (
                        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(place_gripper_xyzs))
                        .voxel_down_sample(ONLINE_VOXEL_RADIUS)
                        .transform(
                            (X_Gplace.multiply(
                                RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2), [0,0,0])
                            )).GetAsMatrix4()
                        )
                    )
                    place_gripper_cloud.paint_uniform_color([1.0, 0.0, 1.0])

                    manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.current_manipuland_pcd.xyzs().T))
                    manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

                    viz_geoms = [manipuland_cloud, pick_gripper_cloud, place_gripper_cloud]
                    o3d.visualization.draw_plotly(viz_geoms)

            if self.place_flipped2:
                print("Solving for flipped place 2")
                t_gripper_center = RigidTransform(X_WG2) @ [0, 0, 3*self.gripper_length/4] # assuming picking with only end half of gripper
                
                # Calculate rotation matrix for 180 degrees around z-axis
                R_z180 = np.array([
                    [-1, 0, 0],
                    [0, -1, 0], 
                    [0, 0, 1]
                ])
                
                # Get current rotation and translation
                R_current = X_WG2[:3, :3]
                t_current = X_WG2[:3, 3]
                
                # Apply rotation around z-axis centered at t_gripper_center
                t_centered = t_current - t_gripper_center
                t_rotated = R_z180 @ t_centered
                t_final = t_rotated + t_gripper_center
                
                # Create new rotation matrix combining current rotation with z-axis rotation
                R_final = R_z180 @ R_current
                
                # Set X_G["place"] to the rotated pose
                X_Gplace = RigidTransform(RotationMatrix(R_final), t_final)
                X_Eplace = X_Gplace.multiply(X_GE)
                X_Epreplace = RigidTransform(
                    X_Eplace.rotation(), X_Eplace.translation() + [0, 0, lift_for_display_height]
                )

                self.q_postgrasp2 = solve_via_analytic_IK_with_retries(
                    pose=X_Epreplace,
                    current_config=q,
                    ik_domain=self.ik_domain,
                    csd_plant=self.csd_plant
                )
                if self.q_postgrasp2 is None:
                    print("failed to solve for place flipped, trying next grasp pair")
                    continue

                if VISUALIZE_FLIPPED:
                    pick_gripper_xyzs = self.grasp_node.hand_collision_model.to_pcd()
                    pick_gripper_cloud = (
                        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pick_gripper_xyzs))
                        .voxel_down_sample(ONLINE_VOXEL_RADIUS)
                        .transform(
                            (RigidTransform(X_WG2).multiply(
                                RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2), [0,0,0])
                            )).GetAsMatrix4()
                        )
                    )
                    pick_gripper_cloud.paint_uniform_color([0.0, 1.0, 0.0])

                    place_gripper_xyzs = self.grasp_node.hand_collision_model.to_pcd()
                    place_gripper_cloud = (
                        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(place_gripper_xyzs))
                        .voxel_down_sample(ONLINE_VOXEL_RADIUS)
                        .transform(
                            (X_Gplace.multiply(
                                RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2), [0,0,0])
                            )).GetAsMatrix4()
                        )
                    )
                    place_gripper_cloud.paint_uniform_color([1.0, 0.0, 1.0])

                    manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.current_manipuland_pcd.xyzs().T))
                    manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

                    viz_geoms = [manipuland_cloud, pick_gripper_cloud, place_gripper_cloud]
                    o3d.visualization.draw_plotly(viz_geoms)

            self.q_pregrasp1 = q_goal1
            self.q_pregrasp2 = q_goal2
            self.X_WG1 = RigidTransform(X_WG1)
            self.X_WG2 = RigidTransform(X_WG2)

            gripper1_xyzs = self.grasp_node.hand_extra_buffer_collision_model.to_pcd()
            gripper1_cloud = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(gripper1_xyzs)).voxel_down_sample(ONLINE_VOXEL_RADIUS).transform(
                    (X_WG1 @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4()))
            gripper1_cloud.paint_uniform_color([1.0, 0.0, 0.0])

            gripper2_xyzs = self.grasp_node.hand_extra_buffer_collision_model.to_pcd()
            gripper2_cloud = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(gripper2_xyzs)).voxel_down_sample(ONLINE_VOXEL_RADIUS).transform(
                    (X_WG2 @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4()))
            gripper2_cloud.paint_uniform_color([0.0, 1.0, 0.0])


            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.05, origin=[0, 0, 0])
            viz_geoms = [world_frame, manipuland_cloud, gripper1_cloud, gripper2_cloud]
            o3d.visualization.draw_plotly(viz_geoms)

            if self.is_semimanual:
                reject = input("Enter 'n' to reject this grasp, press Enter to continue")
                if reject != 'n':
                    break
            else:
                break

        # Solve for pick and display trajectories before moving
        self.PlanPickAndDisplay(context, state)

    def CalcLink7ToManipulandEndLength(self, X_WE: RigidTransform) -> float:
        X_EW = X_WE.inverse()
        manipuland_cloud_points = self.current_manipuland_pcd.xyzs() # X_WP, Shape (3, N)
        manipuland_cloud_points_link7_frame = X_EW @ manipuland_cloud_points # X_EP, Shape (3, N)
        manipuland_cloud_points_link7_frame = manipuland_cloud_points_link7_frame.T # Shape (N,3)

        # The link 7 z-axis points towards the gripper.
        length = np.max(manipuland_cloud_points_link7_frame, axis=0)[2]
        return length

    def PlanPickAndDisplay(self, context, state, skip_first=False):
        # get end effector pose from grasp pose
        X_GE = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 0)), [0, 0, -self.eef_to_gripper_length])
        X_GgraspGpregrasp = RigidTransform([0, 0.0, -self.pregrasp_dist])
        if not skip_first:
            # First Pick
            X_WG1 = self.X_WG1
            X_WE1 = X_WG1.multiply(X_GE)

            X_G1 = {
                "pick": X_WE1,
                "prepick": X_WE1 @ X_GgraspGpregrasp
            }

            height = max(self.CalcLink7ToManipulandEndLength(X_WE1) + display_traj_height_buffer, 0.35)
            X_G1["display_traj"] = get_yaw_display_traj(scanning_traj_height=height)


            X_G1, times1 = MakePickAndDisplayGripperFrames(
                X_G1, self.gripper_length, self.pregrasp_dist,
                self.place_flipped1, X_GE, lift_for_display_height
            )

            display_traj1, self.q_display_center1 = MakeDisplayJointPositionsTrajectory(
                X_G1, 
                times1, 
                self.q_pregrasp1,
                self.ik_domain,
                self.csd_plant
                )
            
            state.get_mutable_abstract_state(int(self._times_index1)).set_value(
                times1
            )
            state.get_mutable_abstract_state(int(self._gripper_pose_index1)).set_value(
                X_G1
            )
            
            state.get_mutable_abstract_state(self._display_traj_index1).set_value(display_traj1)

        # Second Pick
        X_WG2 = self.X_WG2
        X_WE2 = X_WG2.multiply(X_GE)
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        if mode == PlannerState.SCANNING2:
            print("only updating pick and prepick")
            X_G2 = context.get_abstract_state(int(self._gripper_pose_index2)).get_value()
            X_G2["pick"] = X_WE2
            X_G2["prepick"] = X_WE2 @ X_GgraspGpregrasp
            X_G2["pick_start"] = X_G2["pick"]
            X_G2["pick_end"] = X_G2["pick"]
            X_G2["postpick"] = RigidTransform(X_G2["pick"].rotation(), X_G2["pick"].translation() + [0, 0, 0.2])
            
            print(X_G2["preplace"])
            print("Place Start:", X_G2["place_start"])
            print(X_G2["place_end"])
            print(X_G2["postplace"])
            state.get_mutable_abstract_state(int(self._gripper_pose_index2)).set_value(
                X_G2
            )
        else:
            X_G2 = {
                "pick_gripper_frame": X_WG2,
                "pick": X_WE2,
                "prepick": X_WE2 @ X_GgraspGpregrasp
            }
            height = max(self.CalcLink7ToManipulandEndLength(X_WE2) + display_traj_height_buffer, 0.35)
            X_G2["display_traj"] = get_yaw_display_traj(scanning_traj_height=height)
            X_G2, times2 = MakePickAndDisplayGripperFrames(
                X_G2, self.gripper_length, self.pregrasp_dist,
                self.place_flipped2, X_GE, lift_for_display_height
            )

            display_traj2, self.q_display_center2 = MakeDisplayJointPositionsTrajectory(
                X_G2, 
                times2, 
                self.q_pregrasp2,
                self.ik_domain,
                self.csd_plant)

            state.get_mutable_abstract_state(int(self._times_index2)).set_value(
                times2
            )
            state.get_mutable_abstract_state(int(self._gripper_pose_index2)).set_value(
                X_G2
            )
            
            state.get_mutable_abstract_state(self._display_traj_index2).set_value(display_traj2)

    def DoPoseTraj(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        pick_mode = context.get_abstract_state(int(self._pick_mode_index)).get_value()
        current_time = context.get_time()

        if mode == PlannerState.GO_TO_PREGRASP1 or mode == PlannerState.GRASP1:
            X_G = context.get_abstract_state(int(self._gripper_pose_index1)).get_value()
            times = context.get_abstract_state(int(self._times_index1)).get_value()
        elif mode == PlannerState.GO_TO_PREGRASP2 or mode == PlannerState.GRASP2:
            X_G = context.get_abstract_state(int(self._gripper_pose_index2)).get_value()
            times = context.get_abstract_state(int(self._times_index2)).get_value()
        elif mode == PlannerState.GO_TO_SYS_ID_PREGRASP or mode == PlannerState.SYS_ID_GRASP \
            or mode == PlannerState.GO_TO_PREGRASPN or mode == PlannerState.GRASPN:
            X_G = context.get_abstract_state(int(self._gripper_pose_index_single_grasp)).get_value()
            times = context.get_abstract_state(int(self._times_index_single_grasp)).get_value()
        
        traj_type = TrajType.POSTDISPLAY
        if pick_mode == PickState.IDLE:
            traj_type = TrajType.PREDISPLAY
        
        X_G0 = self.GetInputPort("body_poses").Eval(context)[int(self._ee_index)]
        traj_wsg_command = MakeGripperCommandTrajectory(times, traj_type, current_time)
        traj_gripper_pose = MakeGripperPoseTrajectory(X_G, times, X_G0, traj_type, current_time)
        
        state.get_mutable_abstract_state(int(self._traj_wsg_index)).set_value(
            traj_wsg_command
        )

        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(
            traj_gripper_pose
        )
    
    def GoToDisplay(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        
        if mode == PlannerState.GRASP1:
            q_goal = self.q_display_center1
            if not self.place_flipped1:
                self.q_postgrasp1 = q
        elif mode == PlannerState.GRASP2:
            q_goal = self.q_display_center2
            if not self.place_flipped2:
                self.q_postgrasp2 = q
        else:
            q_goal = self.q_display_centern
            self.q_postgraspn = q
        
        obstacles_vox = np.hstack([ceiling_vox])

        if self.use_custom_path_planner:
            # Use custom path planner. Warning: Must be implemented by user
            traj = plan_path_custom(
                q, 
                q_goal,
            )
        else:
            # Use drm path planner
            traj = plan_drm(
                self.drm_planner,
                q, 
                q_goal, 
                obstacles_vox,
                ONLINE_VOXEL_RADIUS
            )
        
        breaks = np.linspace(0, traj.end_time(), int(1e3), endpoint=False)
        knots = traj.vector_values(breaks)

        toppra_traj = reparameterize_with_toppra(
            trajectory=knots.T,
            plant=self._iiwa_controller_plant,
            velocity_limits=self.display_velocity_limits,
            acceleration_limits=self.display_acceleration_limits,
            num_grid_points=100,
        )

        current_time = context.get_time()
        state.get_mutable_abstract_state(self._current_joint_traj_idx).set_value(
            TrajectoryWithTimingInformation(
                trajectory=toppra_traj,
                start_time_s=current_time,
            )
        )

    def DoDisplay(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        
        current_time = context.get_time()
        if mode == PlannerState.GRASP1:
            display_traj = context.get_abstract_state(int(self._display_traj_index1)).get_value()
        elif mode == PlannerState.GRASP2:
            display_traj = context.get_abstract_state(int(self._display_traj_index2)).get_value()
        else:
            display_traj = context.get_abstract_state(int(self._display_traj_indexn)).get_value()
        
        toppra_traj = reparameterize_with_toppra(
            trajectory=display_traj,
            plant=self._iiwa_controller_plant,
            velocity_limits=self.rotate_velocity_limits,
            acceleration_limits=self.display_acceleration_limits,
            num_grid_points=100,
            is_pl=True,
        )

        state.get_mutable_abstract_state(self._current_joint_traj_idx).set_value(
            TrajectoryWithTimingInformation(
                trajectory=toppra_traj,
                start_time_s=current_time,
            )
        )

    def GoToSysID(self, context, state):
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        
        q_goal = self.excitation_traj_start_q
        
        # traj = PiecewisePolynomial.FirstOrderHold([0.0, 4.0], np.array([q, q_goal]).T)
        obstacles_vox = None

        if self.use_custom_path_planner:
            # Use custom path planner. Warning: Must be implemented by user
            traj = plan_path_custom(
                q, 
                q_goal,
            )
        else:
            # Use drm path planner
            traj = plan_drm(
                self.drm_planner,
                q, 
                q_goal, 
                obstacles_vox,
                ONLINE_VOXEL_RADIUS
            )
        
        breaks = np.linspace(0, traj.end_time(), int(1e3), endpoint=False)
        knots = traj.vector_values(breaks)

        toppra_traj = reparameterize_with_toppra(
            trajectory=knots.T,
            plant=self._iiwa_controller_plant,
            velocity_limits=self.velocity_limits,
            acceleration_limits=self.acceleration_limits,
            num_grid_points=100,
        )

        current_time = context.get_time()
        state.get_mutable_abstract_state(self._current_joint_traj_idx).set_value(
            TrajectoryWithTimingInformation(
                trajectory=toppra_traj,
                start_time_s=current_time,
            )
        )

    def DoSysID(self, context, state):
        current_time = context.get_time()

        state.get_mutable_abstract_state(self._current_joint_traj_idx).set_value(
            TrajectoryWithTimingInformation(
                trajectory=self.excitation_traj,
                start_time_s=current_time + 2.0,
            )
        )
    
    def GoToPlace(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        q = self.get_input_port(self._iiwa_position_index).Eval(context)

        if mode == PlannerState.GRASP1:        
            q_goal = self.q_postgrasp1
        if mode == PlannerState.GRASP2:        
            q_goal = self.q_postgrasp2
        if mode == PlannerState.SYS_ID_GRASP:        
            q_goal = self.q_postgrasp_bin
        if mode == PlannerState.GRASPN:
            q_goal = self.q_postgraspn
        
        if mode == PlannerState.SYS_ID_GRASP:
            print("going to bin place")
            obstacles_vox = None
        else:
            obstacles_vox = np.hstack([ceiling_vox])
            
        if self.use_custom_path_planner:
            # Use custom path planner. Warning: Must be implemented by user
            traj = plan_path_custom(
                q, 
                q_goal,
            )
        else:
            # Use drm path planner
            traj = plan_drm(
                self.drm_planner,
                q, 
                q_goal, 
                obstacles_vox,
                ONLINE_VOXEL_RADIUS
            )
        
        breaks = np.linspace(0, traj.end_time(), int(1e3), endpoint=False)
        knots = traj.vector_values(breaks)

        toppra_traj = reparameterize_with_toppra(
            trajectory=knots.T,
            plant=self._iiwa_controller_plant,
            velocity_limits=self.velocity_limits,
            acceleration_limits=self.acceleration_limits,
            num_grid_points=100,
        )

        current_time = context.get_time()
        state.get_mutable_abstract_state(self._current_joint_traj_idx).set_value(
            TrajectoryWithTimingInformation(
                trajectory=toppra_traj,
                start_time_s=current_time,
            )
        )

    def PlanGripper(self, context, state, direction="open"):
        opened = np.array([0.107])
        closed = np.array([0.00])
        current_time = context.get_time()

        traj_wsg_command = PiecewisePolynomial.FirstOrderHold(
            [current_time, current_time+1.0],
            np.hstack([[closed], [opened]]) if direction == "open" else np.hstack([[opened], [closed]]) 
        )

        state.get_mutable_abstract_state(int(self._traj_wsg_index)).set_value(
            traj_wsg_command
        )

    def CalcGripperPose(self, context, output):
        context.get_abstract_state(int(self._mode_index)).get_value()

        traj_X_G = context.get_abstract_state(
            int(self._traj_X_G_index)
        ).get_value()
        if traj_X_G.get_number_of_segments() > 0 and traj_X_G.is_time_in_range(
            context.get_time()
        ):
            # Evaluate the trajectory at the current time, and write it to the
            # output port.
            output.set_value(
                traj_X_G.GetPose(context.get_time())
            )
            return

        # Command the current position (note: this is not particularly good if the velocity is non-zero)
        output.set_value(
            self.GetInputPort("body_poses").Eval(context)[int(self._ee_index)]
        )
    
    def GetCurrentJointPositionTrajectory(self, context, output):
        current_joint_traj = context.get_abstract_state(
            self._current_joint_traj_idx
        ).get_value()
        output.set_value(current_joint_traj)

    def CalcWsgPosition(self, context, output):
        pick_mode = context.get_abstract_state(int(self._pick_mode_index)).get_value()
        opened = np.array([0.107])
        closed = np.array([0.00])

        if pick_mode == PickState.DISPLAY or pick_mode == PickState.TO_DISPLAY or pick_mode == PickState.TO_PLACE:
            output.SetFromVector([closed])
            return

        traj_wsg = context.get_abstract_state(
            int(self._traj_wsg_index)
        ).get_value()
        if traj_wsg.get_number_of_segments() > 0 and traj_wsg.is_time_in_range(
            context.get_time()
        ):
            # Evaluate the trajectory at the current time, and write it to the
            # output port.
            output.SetFromVector(traj_wsg.value(context.get_time()))
            return
        
        if pick_mode == PickState.PICK and traj_wsg.get_number_of_segments() > 0 and context.get_time() > traj_wsg.end_time():
            output.SetFromVector([closed])
            return
        
        output.SetFromVector([opened])
    
    def GetState(self, context, output):
        state = context.get_abstract_state(int(self._mode_index)).get_value()
        output.set_value(state)

    def GetPickState(self, context, output):
        state = context.get_abstract_state(int(self._pick_mode_index)).get_value()
        output.set_value(state)

    def CalcControlMode(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        pick_mode = context.get_abstract_state(int(self._pick_mode_index)).get_value()

        if (mode == PlannerState.GRASP1 or mode == PlannerState.GRASP2 or mode == PlannerState.SYS_ID_GRASP) \
            and (pick_mode == PickState.PICK or pick_mode == PickState.PLACE):
            output.set_value(InputPortIndex(1))  # Diff IK
        # elif mode == PlannerState.SYS_ID_GRASP and pick_mode == PickState.TO_PLACE:
        #     output.set_value(InputPortIndex(1))  # Diff IK
        elif mode == PlannerState.PICK_GRASP:
            output.set_value(InputPortIndex(1))  # Diff IK
        else:
            output.set_value(InputPortIndex(2))  # Joint position control

    def CalcDiffIKReset(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        pick_mode = context.get_abstract_state(int(self._pick_mode_index)).get_value()

        if (mode == PlannerState.GRASP1 or mode == PlannerState.GRASP2 or mode == PlannerState.SYS_ID_GRASP) \
            and (pick_mode == PickState.PICK or pick_mode == PickState.PLACE):
            output.set_value(False)
        # elif mode == PlannerState.SYS_ID_GRASP and pick_mode == PickState.TO_PLACE:
        #     output.set_value(False)
        elif mode == PlannerState.PICK_GRASP:
            output.set_value(False)
        else:
            output.set_value(True)

    def Initialize(self, context, discrete_state):
        discrete_state.set_value(
            int(self._q0_index),
            self.default_home
        )
