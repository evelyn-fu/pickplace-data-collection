import numpy as np
import logging
import pickle
import datetime
import time
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as R
from perception.teaser import icp  
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
from planning.inverse_kinematics import solve_via_analytic_IK
from iiwa_setup_dataclasses.trajectories import TrajectoryWithTimingInformation, Trajectory
from iiwa_setup_dataclasses.bspline_trajectory import CompositeBezierCurveTrajectoryAttributes
from pydrake.systems.framework import LeafSystem
from pydrake.perception import (
    Concatenate,
    PointCloud
)
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
from enum import Enum
from pathlib import Path

import sys
# append path to pycuci to system path
MMT_GCS_ROOT = os.path.abspath(os.path.join(__file__ ,"../../../mmt_gcs/"))
PYCUCI_ROOT = os.path.dirname(__file__) + "/../../" + "cuciv0" 
ONLINE_VOXEL_RADIUS = 0.005
sys.path.append(PYCUCI_ROOT+'/bazel-bin/cuci/src/pybind/pycuci')
import pycuci as cci
PETE_ASSETS =  os.path.dirname(__file__)+"/../pete_assets/"
SAFE_DIRECTIVES = PETE_ASSETS+'assets/directives/iiwa7_on_table_with_ceiling.yaml'
SYS_ID_TRAJ_PARAMETER_PATH = Path(os.path.abspath(os.path.join(__file__ ,"../../../robot_payload_id/logs/corner_setup_payload_box/iiwa_eoptimality_10s_5Fterm_1000timesteps_20_100000/al_19")))

from mmt_gcs.planning.mintime_scs import MintimeSCSWithPathFixing
from mmt_gcs.planning.corridor_planning_utils import CCICollisionChecker, CollisionCheckerBase
from mmt_gcs.planning.region_generation import CCI_inflate_edges_given_pwl_path

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
    
def get_regions_cci(waypoints, voxels, voxel_radius, verbose=False):
    '''
    Inputs:
        waypoints: List[np.ndarray(7,1)], list of precomputed waypoints for a
        trajectory in configuration space around which we want to compute regions
        voxels: np.ndarray(3, n), 3xn array of occupied 3D voxels in space
        voxel_radius: radius of voxels in voxels
    Returns:
        regions: List[HPolyhedron]
    '''
    cci_parser = cci.URDFParser()
    cci_parser.register_package("adaptive_decomp", PETE_ASSETS+"assets")
    cci_parser.register_package("iiwa_description", PETE_ASSETS+"assets/iiwa")
    cci_parser.register_package("wsg_description", PETE_ASSETS+"assets/wsg_description")
    cci_parser.register_package("tri_finray_gripper", PETE_ASSETS+"assets/tri_finray_gripper")
    cci_parser.parse_directives(PETE_ASSETS+"assets/directives/iiwa7_on_table.yaml")
    cci_plant = cci_parser.build_plant()
    cci_mplant = cci_plant.getMinimalPlant()
    cci_domain = cci.HPolyhedron()
    cci_domain.MakeBox(cci_plant.getPositionLowerLimits(), 
                    cci_plant.getPositionUpperLimits())
    cci_objects = {
        'cci_plant' : cci_plant,
        'cci_mplant' : cci_mplant,
        'cci_domain' : cci_domain
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


    edge_inflator = cci.CudaEdgeInflator(cci_objects['cci_mplant'], 
                                        cci_objects['cci_plant'].getRobotGeometryIds(), 
                                        cci_fei_opts, 
                                        cci_objects['cci_domain'])

    regions = []
    for i in range(len(waypoints)-1):
        cci_region : cci.HPolyhedron = edge_inflator.inflateEdge(waypoints[i], 
                                                                waypoints[i+1], 
                                                                cci.Voxels(voxels), 
                                                                voxel_radius, 
                                                                verbose=verbose)

        regions.append(HPolyhedron(cci_region.A(), cci_region.b()))
    
    return regions

def get_cci_edge_inflator(verbose=False):
    cci_parser = cci.URDFParser()
    cci_parser.register_package("adaptive_decomp", PETE_ASSETS+"assets")
    cci_parser.register_package("iiwa_description", PETE_ASSETS+"assets/iiwa")
    cci_parser.register_package("wsg_description", PETE_ASSETS+"assets/wsg_description")
    cci_parser.register_package("tri_finray_gripper", PETE_ASSETS+"assets/tri_finray_gripper")
    cci_parser.parse_directives(PETE_ASSETS+"assets/directives/iiwa7_on_table.yaml")
    cci_plant = cci_parser.build_plant()
    cci_mplant = cci_plant.getMinimalPlant()
    cci_domain = cci.HPolyhedron()
    cci_domain.MakeBox(cci_plant.getPositionLowerLimits(), 
                    cci_plant.getPositionUpperLimits())
    cci_objects = {
        'cci_plant' : cci_plant,
        'cci_mplant' : cci_mplant,
        'cci_domain' : cci_domain
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


    edge_inflator = cci.CudaEdgeInflator(cci_objects['cci_mplant'], 
                                        cci_objects['cci_plant'].getRobotGeometryIds(), 
                                        cci_fei_opts, 
                                        cci_objects['cci_domain'])

    return edge_inflator, cci_objects

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

def sample_goal_drm(
        goal_pose: RigidTransform, 
        current_config,
        drm_planner : cci.DrmPlanner,
        domain : HPolyhedron,
        checker : CollisionCheckerBase,
        num_configs_to_try : int = 10,
        search_cutoff_distance : float = 0.5,
        ):

    close_configs = drm_planner.GetClosestNonCollidingConfigurationsByPose(goal_pose.GetAsMatrix4(), 
                                                                    num_configs_to_try, 
                                                                    search_cutoff_distance)
    close_configs_array = np.array(close_configs)
    delta = close_configs_array - current_config
    sorted = close_configs_array[np.argsort(np.linalg.norm(delta,axis =1))]

    for c in sorted[:2]:
        goal_config = solve_via_analytic_IK(goal_pose, current_config, domain, checker)
        
        if goal_config is None:
            print("[sample_goal_drm] IK failed")
        elif not checker.CheckConfigsCollisionFree(goal_config.reshape(-1,1))[0]:
            print("[sample_goal_biased] IK solution in collision")
            goal_config = None
        else:
            break

        attempts = 0
        while goal_config is None and attempts < 10:
            print("trying global inverse kinematics with new initial guess randomized around c")
            goal_config = solve_via_analytic_IK(goal_pose, current_config, domain, checker)
            if goal_config is None:
                print("[sample_goal_drm] IK failed")
            elif not checker.CheckConfigsCollisionFree(goal_config.reshape(-1,1))[0]:
                print("[sample_goal_biased] IK solution in collision")
                goal_config = None
            attempts += 1
        
        if goal_config is not None:
            break
    
    return goal_config, goal_pose

def drm_planner(cci_obj, vox=None):
    drm_pl_opts = cci.DrmPlannerOptions()
    drm_pl_opts.max_number_planning_attempts = 50
    drm_pl_opts.try_shortcutting = True
    drm_pl_opts.online_edge_step_size = 0.005

    drm_planner = cci.DrmPlanner(cci_obj['cci_plant'], drm_pl_opts)
    drm_planner.LoadRoadmap(MMT_GCS_ROOT+"/tmp/iiwa_hardware/iiwa_roadmap_1_0_0.01_0.2_50000_10_4.5_0.45.rm")

    if vox is not None:
        online_voxel_observation = cci.Voxels(vox.T)
        drm_planner.BuildCollisionSet(online_voxel_observation)
    
    return drm_planner

def scs_trajopt(start, goal, drm_planner, cci_obj, edge_inflator, vox, vel_limits, acc_limits):
    online_voxel_observation = cci.Voxels(vox)
    
    success, pwl_plan = drm_planner.Plan(start,
                     goal,
                     online_voxel_observation,
                     ONLINE_VOXEL_RADIUS)
    
    regions, edges = CCI_inflate_edges_given_pwl_path(pwl_plan, 
                                     edge_inflator, 
                                     online_voxel_observation,
                                     ONLINE_VOXEL_RADIUS,
                                     verbose = True
                                     )
    
    cci_checker = CCICollisionChecker(cci_obj['cci_mplant'], 
                                    cci_obj['cci_plant'].getRobotGeometryIds(), 
                                    online_voxel_observation, 
                                    ONLINE_VOXEL_RADIUS)

    vel_limits_reflected = [-vel_limits, vel_limits]
    acc_limits_reflected = [-acc_limits, acc_limits]
    traj, cost, timing_info, traj_col_free, \
    first_solve_collision_free, collisions =  MintimeSCSWithPathFixing(start,
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

    return traj

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

def compute_principal_minor_components(pcd):
    cov = np.cov(pcd.T)
    eigval, eigvec = np.linalg.eig(cov)

    order = eigval.argsort()
    principal_component = eigvec[:, order[-1]]
    secondary_component = eigvec[:, order[1]]
    minor_component = eigvec[:, order[0]]

    return principal_component, secondary_component, minor_component

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

class PlannerState(Enum):
    WAIT_FOR_OBJECTS_TO_SETTLE = 1
    START = 2
    PLAN_PICK = 12
    GO_TO_PICK_PREGRASP = 13
    PICK_GRASP = 14
    GO_HOME0 = 15
    SCANNING1 = 3
    GO_TO_PREGRASP1 = 4
    GRASP1 = 5
    GO_HOME1 = 6
    SCANNING2 = 7
    GO_TO_PREGRASP2 = 8
    GRASP2 = 9
    GO_HOME2 = 10
    PLAN_SYS_ID = 16
    GO_TO_SYS_ID_PREGRASP = 17
    SYS_ID_GRASP = 18
    GO_HOME3 = 19
    RESET = 20
    DONE = 11

class PickState(Enum):
    IDLE = 1
    PICK = 2
    TO_DISPLAY = 3
    DISPLAY = 4
    TO_PLACE = 5
    PLACE = 6

yaw_display_traj = []

yaw_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, 0)), [0.4, 0.0, 0.6]))
yaw_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, -np.pi/4)), [0.4, 0.0, 0.6]))
yaw_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, -np.pi/2)), [0.4, 0.0, 0.6]))
yaw_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, -3*np.pi/4)), [0.4, 0.0, 0.6]))
yaw_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, -np.pi)), [0.4, 0.0, 0.6]))
yaw_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, -5*np.pi)), [0.4, 0.0, 0.6]))
yaw_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, -3*np.pi/2)), [0.4, 0.0, 0.6]))

q_home = [0.3, 0.4, 0.0, -1.2, 0.0, 1.0, -1.57]

x, y = np.meshgrid(np.arange(0.0, 1.02, 0.02), np.arange(-0.4, 0.42, 0.02))
ceiling_vox = np.vstack((x.flatten(), y.flatten(), np.zeros_like(x.flatten())))
ceiling_vox += np.array([0.4, 0.0, 0.9])[:, np.newaxis]

x, y = np.meshgrid(np.arange(0.0, 0.075, 0.3), np.arange(0.0, -0.075, -0.3))
camera_vox = np.vstack((x.flatten(), y.flatten(), np.zeros_like(x.flatten())))
camera_vox += np.array([0.9, 0.0, 0.3])[:, np.newaxis]

x, y = np.meshgrid(np.arange(-0.2, 0.075, 0.02), np.arange(0.02, -0.075, -0.36))
bin_cam_vox = np.vstack((x.flatten(), y.flatten(), np.zeros_like(x.flatten())))
bin_cam_vox += np.array([-0.0338161, 0.62563, 0.360087])[:, np.newaxis]

stage_center = RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, 0.0)), [0.4, 0.0, 0.0])
stage_center90 = RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, np.pi/2)), [0.4, 0.0, 0.0])
bin_depth = 0.031
platform_height = 0.068

end_bin = RigidTransform(RotationMatrix(RollPitchYaw(0.0, np.pi/2, np.pi)), [0.05, -0.55, 0.2])

class TwoGraspPlanner(LeafSystem):
    def __init__(
            self, 
            plant,
            controller_plant,
            X_WC0,
            X_WC1,
            X_WC2,
            X_WC_bin,
            meshcat, 
            dirstr,
            time_horizon,
            models_path=None,
            gripper_model_path=None,
            default_home=q_home,
            gripper_length=0.125,
            pregrasp_dist=0.18,
            eef_to_gripper_length=0.12,
        ):
        LeafSystem.__init__(self)

        # For grasp planner
        self.current_scene_pcd = PointCloud(0)
        self.current_manipuland_pcd = PointCloud(0)
        self.DeclareAbstractInputPort("cloud_bin", AbstractValue.Make(PointCloud(0)))
        self.DeclareAbstractInputPort("cloud_front", AbstractValue.Make(PointCloud(0)))
        self.DeclareAbstractInputPort("cloud_back_right", AbstractValue.Make(PointCloud(0)))
        self.DeclareAbstractInputPort("cloud_back_left", AbstractValue.Make(PointCloud(0)))
        self._X_WC0 = X_WC0
        self._X_WC1 = X_WC1
        self._X_WC2 = X_WC2
        self._X_WC_bin = X_WC_bin

        # for getting current positions
        self._ee_index = plant.GetBodyByName("iiwa_link_7").index()
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )

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

        # output state (for image saving)
        self.DeclareAbstractOutputPort(
            "planner_state", 
            lambda: AbstractValue.Make(PlannerState.START),
            self.GetState
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

        self.grasp_node = GraspListener(gripper_model_path=gripper_model_path)
        self.q_pregrasp1 = None
        self.q_pregrasp2 = None
        self.q_bin_pregrasp = None
        self.q_display_center1 = None
        self.q_display_center2 = None
        self.q_sys_id_pregrasp = None
        self.q_postgrasp = None
        self.place_flipped1 = False
        self.place_flipped2 = False
        self.meshcat = meshcat
        self.plant = plant
        self._iiwa_controller_plant = controller_plant
        self.fake_plant, self.fake_plant_context = make_iiwa_plant()
        self.velocity_limits = 0.4 * np.ones(7)
        self.acceleration_limits = 0.4 * np.ones(7)
        self.display_velocity_limits = 0.1 * np.ones(7)
        self.rotate_velocity_limits = 0.1 * np.ones(7)
        self.rotate_velocity_limits[6] = 0.075
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
        self.edge_inflator, self.cci_objects = get_cci_edge_inflator()
        self.drm_planner = drm_planner(self.cci_objects)

        self.models_path = models_path
        self.savedir = dirstr
        self.done = False

        self.pcd0 = None
        self.pcd1 = None
        self.pcd2 = None
        self.pcd3 = None

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
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.PLAN_PICK)
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
            return
        if mode == PlannerState.PICK_GRASP:
            traj_pose = context.get_abstract_state(
                int(self._traj_X_G_index)
            ).get_value()
            if traj_pose.get_number_of_segments() > 0 and context.get_time() > traj_pose.end_time():
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.GO_HOME0)
                self.GoHome(context, state)
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
            input("Next: Pregrasp 1 (scs)") # pause for debugging
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
            return
        if mode == PlannerState.GO_HOME2:
            traj_q = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().trajectory
            start_time = context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().start_time_s
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.PLAN_SYS_ID)
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
                    print("Doing sysid")
                    self.DoSysID(context, state)
                else:
                    self.DoDisplay(context, state)
                # input("Next: DoDisplay (IK + Toppra)") # pause for debugging
        if pick_mode == PickState.DISPLAY:
            if context.get_time() > traj_q.end_time() + start_time:
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
        bin_pcd = cloud.Crop(lower_xyz=[-0.04, 0.41, -bin_depth], upper_xyz=[0.165, 0.725, 0.16])
        bin_pcd.EstimateNormals(radius=0.1, num_closest=30)
        bin_pcd.FlipNormalsTowardPoint(self._X_WC_bin.translation())
        bin_pcd = bin_pcd.VoxelizedDownSample(voxel_size=ONLINE_VOXEL_RADIUS)

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
        
        self.meshcat.SetObject("bin_cloud", bin_pcd, point_size=0.001, rgba=Rgba(1,0,1,1))

        # get end effector pose from grasp pose
        X_GE = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 0)), [0, 0, -self.eef_to_gripper_length])

        # pregrasp is negative z in the gripper frame
        X_GgraspGpregrasp = RigidTransform([0, 0.0, -self.pregrasp_dist])

        # Initialize DRM with current pcd
        self.drm_planner = drm_planner(self.cci_objects, bin_pcd.xyzs().T)

        cci_checker = CCICollisionChecker(self.cci_objects['cci_mplant'], 
                                        self.cci_objects['cci_plant'].getRobotGeometryIds(), 
                                        cci.Voxels(bin_pcd.xyzs()), 
                                        ONLINE_VOXEL_RADIUS)
        
        # Planning first grasping trajectory
        self.grasp_node.compute_candidate_grasps(
            bin_pcd,
            bin_pcd,
            candidate_num=1,
            num_samples=10,
            random_seed=np.random.randint(1000),
            grasp_type=GraspType.TOP
        )

        grasps = self.grasp_node.get_best_grasps()
        print("bin grasps:", len(grasps))
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        for i in range(len(grasps)):
            print("Grasp:", grasps[i])

            X_WG = grasps[i]
            X_WPregrasp = (RigidTransform(X_WG) @ X_GE) @ X_GgraspGpregrasp

            q_goal = solve_via_analytic_IK(
                pose=X_WPregrasp,
                current_config=q,
                ik_domain=self.ik_domain,
                checker=cci_checker
            )

            if q_goal is None:
                continue
            
            self.q_bin_pregrasp = q_goal
            X_WG_bin = RigidTransform(X_WG)
            break

        manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bin_pcd.xyzs().T))
        manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

        gripper_xyzs = self.grasp_node.hand_collision_model.to_pcd()
        gripper_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(ONLINE_VOXEL_RADIUS).transform((X_WG @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4()))
        gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])

        viz_geoms = [manipuland_cloud, gripper_cloud]
        # o3d.visualization.draw_plotly(viz_geoms)

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

        X_G, times = MakePickGripperFrames(X_G, 10.0)
        
        state.get_mutable_abstract_state(int(self._times_index_single_grasp)).set_value(
            times
        )
        state.get_mutable_abstract_state(int(self._gripper_pose_index_single_grasp)).set_value(
            X_G
        )

        state.get_mutable_abstract_state(
            int(self._mode_index)
        ).set_value(after_scan_state)
        self.PlanToPregrasp(context, state)
    
    def PlanSysIdPick(self, context, state, after_scan_state):
        self.GetPointCloud(context, state, after_scan_state)

        # get end effector pose from grasp pose
        X_GE = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 0)), [0, 0, -self.eef_to_gripper_length])

        # pregrasp is negative z in the gripper frame
        X_GgraspGpregrasp = RigidTransform([0, 0.0, -self.pregrasp_dist])

        # Initialize DRM with current pcd
        self.drm_planner = drm_planner(self.cci_objects, self.current_manipuland_pcd.xyzs().T)

        cci_checker = CCICollisionChecker(self.cci_objects['cci_mplant'], 
                                        self.cci_objects['cci_plant'].getRobotGeometryIds(), 
                                        cci.Voxels(self.current_manipuland_pcd.xyzs()), 
                                        ONLINE_VOXEL_RADIUS)
        
        # Planning first grasping trajectory
        self.grasp_node.compute_candidate_grasps(
            self.current_manipuland_pcd,
            self.current_scene_pcd,
            candidate_num=1,
            num_samples=10,
            random_seed=np.random.randint(1000),
            grasp_type=GraspType.STABLE
        )

        grasps = self.grasp_node.get_best_grasps()
        print("grasps:", len(grasps))
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        for i in range(len(grasps)):
            print("Grasp:", grasps[i])

            X_WG = grasps[i]
            X_WPregrasp = (RigidTransform(X_WG) @ X_GE) @ X_GgraspGpregrasp

            q_goal = solve_via_analytic_IK(
                pose=X_WPregrasp,
                current_config=q,
                ik_domain=self.ik_domain,
                checker=cci_checker
            )

            if q_goal is None:
                continue
            
            self.q_sys_id_pregrasp = q_goal
            X_WG_sys_id = RigidTransform(X_WG)
            break

        manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.current_manipuland_pcd.xyzs().T))
        manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

        gripper_xyzs = self.grasp_node.hand_collision_model.to_pcd()
        gripper_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(ONLINE_VOXEL_RADIUS).transform((X_WG @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4()))
        gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])

        viz_geoms = [manipuland_cloud, gripper_cloud]
        # o3d.visualization.draw_plotly(viz_geoms)

        # Solve for pick trajectory before moving
        X_WE = X_WG_sys_id.multiply(X_GE)
        X_G = {
            "pick": X_WE,
            "prepick": X_WE @ X_GgraspGpregrasp,
            "place": end_bin,
            "postplace": end_bin @ RigidTransform([0, 0.0, -0.1]),
        }

        X_G, times = MakePickGripperFrames(X_G)
        
        state.get_mutable_abstract_state(int(self._times_index_single_grasp)).set_value(
            times
        )
        state.get_mutable_abstract_state(int(self._gripper_pose_index_single_grasp)).set_value(
            X_G
        )

        q_home = context.get_discrete_state(self._q0_index).get_value().copy()
        self.q_postgrasp = solve_via_analytic_IK(
            pose=X_G["preplace"],
            current_config=q_home,
            ik_domain=self.ik_domain,
            checker=cci_checker
        )

        state.get_mutable_abstract_state(
            int(self._mode_index)
        ).set_value(after_scan_state)
        self.PlanToPregrasp(context, state)

    def PlanBinPickPoseTraj(self, context, state):
        current_time = context.get_time()

        X_G = context.get_abstract_state(int(self._gripper_pose_index_single_grasp)).get_value()
        times = context.get_abstract_state(int(self._times_index_single_grasp)).get_value()
            
        traj_wsg_command = MakeGripperCommandTrajectory(times, TrajType.BIN, current_time)
        traj_gripper_pose = MakeGripperPoseTrajectory(X_G, times, TrajType.BIN, current_time)
        
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
        pcd0 = cloud0.Crop(lower_xyz=[0.23, -0.17, platform_height], upper_xyz=[0.7, 0.17, 0.27])
        pcd0.EstimateNormals(radius=0.1, num_closest=30)
        pcd0.FlipNormalsTowardPoint(self._X_WC0.translation())

        cloud1 = self.GetInputPort("cloud_back_left").Eval(context)
        pcd1 = cloud1.Crop(lower_xyz=[0.23, -0.17, platform_height], upper_xyz=[0.7, 0.17, 0.27])
        pcd1.EstimateNormals(radius=0.1, num_closest=30)
        pcd1.FlipNormalsTowardPoint(self._X_WC1.translation())

        cloud2 = self.GetInputPort("cloud_back_right").Eval(context)
        pcd2 = cloud2.Crop(lower_xyz=[0.23, -0.17, platform_height], upper_xyz=[0.7, 0.17, 0.27])
        pcd2.EstimateNormals(radius=0.1, num_closest=30)
        pcd2.FlipNormalsTowardPoint(self._X_WC2.translation())

        merged_pcd = Concatenate([pcd0, pcd1, pcd2])
        down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=ONLINE_VOXEL_RADIUS)
        
        # remove outliers
        o3d_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(down_sampled_pcd.xyzs().T))
        try:
            cl, ind = o3d_cloud.remove_statistical_outlier(
                nb_neighbors=int(down_sampled_pcd.xyzs().shape[1] // 20), 
                std_ratio=0.8)
            filtered_pts = np.asarray(o3d_cloud.points)[ind]
            down_sampled_pcd.resize(filtered_pts.shape[0])
            down_sampled_pcd.mutable_xyzs()[:] = filtered_pts.T
        except:
            pass
        
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

            viz_geoms = [old_manipuland_cloud, old_gripper2_cloud, manipuland_cloud, gripper2_cloud]
            o3d.visualization.draw_plotly(viz_geoms)
            viz_geoms = [old_manipuland_cloud, transformed_manipuland_cloud, manipuland_cloud, numpy_transformed_manipuland_cloud]
            o3d.visualization.draw_plotly(viz_geoms)

            if translation_magnitude > 0.01 or rotation_magnitude > np.pi * 10.0/180.0:
                print("Transformation is too large, realigning grasp 2")

                self.X_WG2 = temp_X_WG2
                
                # get end effector pose from grasp pose
                X_GE = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 0)), [0, 0, -self.eef_to_gripper_length])

                # pregrasp is negative z in the gripper frame
                X_GgraspGpregrasp = RigidTransform([0, 0.0, -self.pregrasp_dist])
                X_WPregrasp2 = (RigidTransform(self.X_WG2) @ X_GE) @ X_GgraspGpregrasp


                cci_checker = CCICollisionChecker(self.cci_objects['cci_mplant'], 
                                                self.cci_objects['cci_plant'].getRobotGeometryIds(), 
                                                cci.Voxels(down_sampled_pcd.xyzs()), 
                                                ONLINE_VOXEL_RADIUS)
                attempts = 0
                while attempts < 50:
                    q_goal2 = solve_via_analytic_IK(
                        pose=X_WPregrasp2,
                        current_config=self.q_pregrasp2,
                        ik_domain=self.ik_domain,
                        checker=cci_checker
                    )
                    if q_goal2 is not None:
                        break
                    attempts += 1

                if q_goal2 is None:
                    print("Failed to solve IK for grasp 2")

                self.q_pregrasp2 = q_goal2
                self.PlanPickAndDisplay(context, state, skip_first=True)
                

        self.current_manipuland_pcd = down_sampled_pcd
        print("object pcd got in", time.time()-start, "seconds")

        start = time.time()
        # Get table pcd 
        cloud0 = self.GetInputPort("cloud_front").Eval(context)
        pcd0 = cloud0.Crop(lower_xyz=[0.1, -0.3, 0.0], upper_xyz=[0.85, 0.3, 0.07]).VoxelizedDownSample(voxel_size=0.01)
        pcd0.EstimateNormals(radius=0.1, num_closest=30)
        pcd0.FlipNormalsTowardPoint(self._X_WC0.translation())

        cloud1 = self.GetInputPort("cloud_back_left").Eval(context)
        pcd1 = cloud1.Crop(lower_xyz=[0.1, -0.3, 0.0], upper_xyz=[0.85, 0.3, 0.07]).VoxelizedDownSample(voxel_size=0.01)
        pcd1.EstimateNormals(radius=0.1, num_closest=30)
        pcd1.FlipNormalsTowardPoint(self._X_WC1.translation())

        cloud2 = self.GetInputPort("cloud_back_right").Eval(context)
        pcd2 = cloud2.Crop(lower_xyz=[0.1, -0.3, 0.0], upper_xyz=[0.85, 0.3, 0.07]).VoxelizedDownSample(voxel_size=0.01)
        pcd2.EstimateNormals(radius=0.1, num_closest=30)
        pcd2.FlipNormalsTowardPoint(self._X_WC2.translation())

        merged_pcd = Concatenate([pcd0, pcd1, pcd2])
        
        # remove outliers
        o3d_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(merged_pcd.xyzs().T))
        cl, ind = o3d_cloud.remove_statistical_outlier(
            nb_neighbors=int(merged_pcd.xyzs().shape[1] // 20), 
            std_ratio=0.8)
        filtered_pts = np.asarray(o3d_cloud.points)[ind]
        merged_pcd.resize(filtered_pts.shape[0])
        merged_pcd.mutable_xyzs()[:] = filtered_pts.T
        self.current_scene_pcd = merged_pcd
        print("table pcd got in", time.time()-start, "seconds")

        # Find grasp candidates
        if mode == PlannerState.SCANNING1:
            self.PlanGrasp(context, state)
        if mode == PlannerState.PLAN_SYS_ID:
            return
        
        state.get_mutable_abstract_state(
            int(self._mode_index)
        ).set_value(after_scan_state)
        self.PlanToPregrasp(context, state)

        return

    def GoHome(self, context, state):
        '''
        Reset to default home position to move arm out of the way of the camera
        '''

        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        q_goal = context.get_discrete_state(self._q0_index).get_value().copy() # initial pose

        try:
            obstacles_vox = np.hstack([ceiling_vox, camera_vox, bin_cam_vox, self.current_manipuland_pcd.xyzs()])

            traj = scs_trajopt(
                q, 
                q_goal, 
                self.drm_planner,
                self.cci_objects, 
                self.edge_inflator, 
                obstacles_vox, 
                self.velocity_limits, 
                self.acceleration_limits
            )
        except:
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
        if mode == PlannerState.SCANNING2:
            q_goal = self.q_pregrasp2
        if mode == PlannerState.PLAN_PICK:
            q_goal = self.q_bin_pregrasp
        if mode == PlannerState.PLAN_SYS_ID:
            q_goal = self.q_sys_id_pregrasp

        obstacles_vox = np.hstack([ceiling_vox, camera_vox, bin_cam_vox, self.current_manipuland_pcd.xyzs()])
    
        traj = scs_trajopt(
            q, 
            q_goal, 
            self.drm_planner,
            self.cci_objects, 
            self.edge_inflator, 
            obstacles_vox, 
            self.velocity_limits, 
            self.acceleration_limits
        )
        if traj is None:
            logging.error("Failed to find a path to the grasping start positions.")
            exit(1)

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
        self.drm_planner = drm_planner(self.cci_objects, self.current_manipuland_pcd.xyzs().T)

        cci_checker = CCICollisionChecker(self.cci_objects['cci_mplant'], 
                                        self.cci_objects['cci_plant'].getRobotGeometryIds(), 
                                        cci.Voxels(self.current_manipuland_pcd.xyzs()), 
                                        ONLINE_VOXEL_RADIUS)
        # while not grasps_found:
        # Planning first grasping trajectory
        self.grasp_node.compute_candidate_grasps(
            self.current_manipuland_pcd,
            pcd_with_background,
            candidate_num=1,
            num_samples=30,
            random_seed=np.random.randint(1000),
            grasp_type=GraspType.PAIR
        )

        grasp_pairs = self.grasp_node.get_best_grasps()
        print("grasp pairs:", len(grasp_pairs))
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        for i in range(len(grasp_pairs)):
            print("Grasp Pair:", grasp_pairs[i])

            X_WG1 = grasp_pairs[i][0]
            X_WG2 = grasp_pairs[i][1]
            X_WPregrasp1 = (RigidTransform(X_WG1) @ X_GE) @ X_GgraspGpregrasp
            X_WPregrasp2 = (RigidTransform(X_WG2) @ X_GE) @ X_GgraspGpregrasp

            q_goal1 = solve_via_analytic_IK(
                pose=X_WPregrasp1,
                current_config=q,
                ik_domain=self.ik_domain,
                checker=cci_checker
            )

            if q_goal1 is None:
                continue
                
            # check if configuration is in collision with scene
            if check_configuration_has_collisions(self.models_path, com, rot, dims, q_goal1):
                print("grasp 1 configuration has collision with scene")
                continue

            q_goal2 = solve_via_analytic_IK(
                pose=X_WPregrasp2,
                current_config=q,
                ik_domain=self.ik_domain,
                checker=cci_checker
            )

            if q_goal2 is None:
                continue

            # check if configuration is in collision with scene
            if check_configuration_has_collisions(self.models_path, com, rot, dims, q_goal2):
                print("grasp 2 configuration has collision with scene")
                continue
            
            self.q_pregrasp1 = q_goal1
            self.q_pregrasp2 = q_goal2
            self.X_WG1 = RigidTransform(X_WG1)
            self.X_WG2 = RigidTransform(X_WG2)
            break

        manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.current_manipuland_pcd.xyzs().T))
        manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

        gripper1_xyzs = self.grasp_node.hand_collision_model.to_pcd()
        gripper1_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper1_xyzs)).voxel_down_sample(ONLINE_VOXEL_RADIUS).transform((X_WG1 @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4()))
        gripper1_cloud.paint_uniform_color([1.0, 0.0, 0.0])

        gripper2_xyzs = self.grasp_node.hand_collision_model.to_pcd()
        gripper2_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper2_xyzs)).voxel_down_sample(ONLINE_VOXEL_RADIUS).transform((X_WG2 @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4()))
        gripper2_cloud.paint_uniform_color([0.0, 1.0, 0.0])

        viz_geoms = [manipuland_cloud, gripper1_cloud, gripper2_cloud]
        o3d.visualization.draw_plotly(viz_geoms)

        # Solve for pick and display trajectories before moving
        self.PlanPickAndDisplay(context, state)

    def PlanPickAndDisplay(self, context, state, skip_first=False):
        cci_checker = CCICollisionChecker(self.cci_objects['cci_mplant'], 
                                        self.cci_objects['cci_plant'].getRobotGeometryIds(), 
                                        cci.Voxels(), 
                                        ONLINE_VOXEL_RADIUS)
            
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

            X_G1["display_traj"] = yaw_display_traj
            X_G1, times1 = MakePickAndDisplayGripperFrames(X_G1, self.gripper_length, self.pregrasp_dist, self.place_flipped1, X_GE)

            display_traj1, self.q_display_center1 = MakeDisplayJointPositionsTrajectory(
                X_G1, 
                times1, 
                self.q_pregrasp1,
                self.ik_domain,
                cci_checker
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
        if self.q_postgrasp is not None:
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
            X_G2["display_traj"] = yaw_display_traj
            X_G2, times2 = MakePickAndDisplayGripperFrames(X_G2, self.gripper_length, self.pregrasp_dist, self.place_flipped2, X_GE)

            display_traj2, self.q_display_center2 = MakeDisplayJointPositionsTrajectory(
                X_G2, 
                times2, 
                self.q_pregrasp2,
                self.ik_domain,
                cci_checker)

            if self.place_flipped2:
                print("Solving for flipped place")
                
                self.q_postgrasp = solve_via_analytic_IK(
                    pose=X_G2["preplace"],
                    current_config=self.q_display_center2,
                    ik_domain=self.ik_domain,
                    checker=cci_checker
                )

                pick_gripper_xyzs = self.grasp_node.hand_collision_model.to_pcd()
                pick_gripper_cloud = (
                    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pick_gripper_xyzs))
                    .voxel_down_sample(ONLINE_VOXEL_RADIUS)
                    .transform(
                        (X_G2["pick_gripper_frame"].multiply(
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
                        (X_G2["place_gripper_frame"].multiply(
                            RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2), [0,0,0])
                        )).GetAsMatrix4()
                    )
                )
                place_gripper_cloud.paint_uniform_color([1.0, 0.0, 1.0])

                manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.current_manipuland_pcd.xyzs().T))
                manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

                viz_geoms = [manipuland_cloud, pick_gripper_cloud, place_gripper_cloud]
                o3d.visualization.draw_plotly(viz_geoms)

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
        elif mode == PlannerState.GO_TO_SYS_ID_PREGRASP or mode == PlannerState.SYS_ID_GRASP:
            X_G = context.get_abstract_state(int(self._gripper_pose_index_single_grasp)).get_value()
            times = context.get_abstract_state(int(self._times_index_single_grasp)).get_value()
        
        traj_type = TrajType.POSTDISPLAY
        if pick_mode == PickState.IDLE:
            traj_type = TrajType.PREDISPLAY
            
        traj_wsg_command = MakeGripperCommandTrajectory(times, traj_type, current_time)
        traj_gripper_pose = MakeGripperPoseTrajectory(X_G, times, traj_type, current_time)
        
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
                self.q_postgrasp = q
        else:
            q_goal = self.q_display_center2
            if not self.place_flipped2:
                self.q_postgrasp = q
        
        obstacles_vox = np.hstack([ceiling_vox, camera_vox, bin_cam_vox])

        traj = scs_trajopt(
            q, 
            q_goal, 
            self.drm_planner,
            self.cci_objects, 
            self.edge_inflator, 
            obstacles_vox, 
            self.display_velocity_limits, 
            self.display_acceleration_limits
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
        else:
            display_traj = context.get_abstract_state(int(self._display_traj_index2)).get_value()
        
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
        obstacles_vox = np.hstack([camera_vox, bin_cam_vox])

        traj = scs_trajopt(
            q, 
            q_goal, 
            self.drm_planner,
            self.cci_objects, 
            self.edge_inflator, 
            obstacles_vox, 
            self.display_velocity_limits, 
            self.display_acceleration_limits
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
                start_time_s=current_time+1.0,
            )
        )
    
    def GoToPlace(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        q_goal = self.q_postgrasp
        
        if mode == PlannerState.SYS_ID_GRASP:
            print("going to bin place")
            obstacles_vox = np.hstack([camera_vox, bin_cam_vox])
        else:
            obstacles_vox = np.hstack([ceiling_vox, camera_vox, bin_cam_vox])
        traj = scs_trajopt(
            q, 
            q_goal, 
            self.drm_planner,
            self.cci_objects, 
            self.edge_inflator, 
            obstacles_vox, 
            self.velocity_limits, 
            self.acceleration_limits
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
