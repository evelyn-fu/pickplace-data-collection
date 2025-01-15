import numpy as np
import logging
import pickle
import datetime
import time
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as R
from perception.icp import icp
from planning.grasp import GraspListener
from planning.toppra import reparameterize_with_toppra
from planning.trajectories import (
    MakePickAndDisplayGripperFrames,
    MakePickAndDisplayJointPositionsTrajectory,
)
from planning.trajectory_sources import TrajectoryWithTimingInformationSource
from planning.gcs import plan_unconstrained_gcs_path_start_to_goal
from planning.inverse_kinematics import solve_global_inverse_kinematics
from planning.rrt import get_collision_checker, rrt_planning
from iiwa_setup_dataclasses.trajectories import TrajectoryWithTimingInformation
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

import pydrake.planning as mut
from pydrake.common import RandomGenerator, use_native_cpp_logging
from pydrake.planning import (RobotDiagramBuilder,
                              SceneGraphCollisionChecker)
from pydrake.solvers import MosekSolver, GurobiSolver
from pydrake.geometry.optimization import IrisOptions, IrisInConfigurationSpace
from pydrake.all import IrisZo, IrisZoOptions, Hyperellipsoid, HPolyhedron
from pydrake.geometry import Rgba

from manipulation.meshcat_utils import AddMeshcatTriad
from enum import Enum

import sys
# append path to pycuci to system path
PYCUCI_ROOT = os.path.dirname(__file__) + "/../../" + "cuciv0" 
sys.path.append(PYCUCI_ROOT+'/bazel-bin/cuci/src/pybind/pycuci')
import pycuci as cci
PETE_ASSETS =  os.path.dirname(__file__)+"/../pete_assets/"

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

class PlannerState(Enum):
    WAIT_FOR_OBJECTS_TO_SETTLE = 1
    START = 2
    SCANNING1 = 3
    GO_TO_PREGRASP1 = 4
    GRASP1 = 5
    GO_HOME1 = 6
    SCANNING2 = 7
    GO_TO_PREGRASP2 = 8
    GRASP2 = 9
    GO_HOME2 = 10
    DONE = 11

class PickState(Enum):
    IDLE = 1
    PREPICK = 2
    CLOSING = 3
    POSTPICK = 4
    TO_DISPLAY = 5
    DISPLAY = 6
    TO_PREPLACE = 7
    PLACE = 8
    OPENING = 9
    POSTPLACE = 10

class ScanState(Enum):
    IDLE = 1
    GO_TO_1 = 2
    GO_TO_2 = 3
    GO_TO_3 = 4
    GO_HOME = 5
    DONE = 6

yaw_display_traj = []

yaw_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, 0)), [0.4, 0.0, 0.6]))
yaw_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, -np.pi/4)), [0.4, 0.0, 0.6]))
yaw_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, -np.pi/2)), [0.4, 0.0, 0.6]))
yaw_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, -3*np.pi/4)), [0.4, 0.0, 0.6]))
yaw_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, -np.pi)), [0.4, 0.0, 0.6]))
yaw_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, -5*np.pi)), [0.4, 0.0, 0.6]))
yaw_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, -3*np.pi/2)), [0.4, 0.0, 0.6]))

q_home = [0.3, 0.4, 0.0, -1.2, 0.0, 1.0, -1.57]

class TwoGraspPlanner(LeafSystem):
    def __init__(
            self, 
            plant,
            controller_plant,
            X_WC0,
            X_WC1,
            X_WC2,
            scanning_traj_dir,
            meshcat, 
            dirstr,
            regions1=None,
            regions2=None,
            traj_dir=None,
            models_path=None,
            no_obstacles=False,
            gripper_model_path=None,
            default_home=q_home,
            gripper_length=0.12,
            pregrasp_dist=0.18,
            eef_to_gripper_length=0.1,
        ):
        LeafSystem.__init__(self)

        # For grasp planner
        self.current_scene_pcd = PointCloud(0)
        self.current_manipuland_pcd = PointCloud(0)
        self.DeclareAbstractInputPort("cloud_front", AbstractValue.Make(PointCloud(0)))
        self.DeclareAbstractInputPort("cloud_back_right", AbstractValue.Make(PointCloud(0)))
        self.DeclareAbstractInputPort("cloud_back_left", AbstractValue.Make(PointCloud(0)))
        self._X_WC0 = X_WC0
        self._X_WC1 = X_WC1
        self._X_WC2 = X_WC2
        self._scanning_traj_dir = scanning_traj_dir

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
        self._scan_mode_index = self.DeclareAbstractState(
            AbstractValue.Make(ScanState.IDLE)
        )

        # Store last calculated grasp pose
        self._grasp_X_G_index = self.DeclareAbstractState(
            AbstractValue.Make(RigidTransform())
        )

        # Store the path parameterized display trajectories
        self._to_postpick_traj_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )
        self._display_traj_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )
        self._to_place_traj_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )
        self._place_traj_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )

        self._gripper_traj_end_time = None

        # Store planned grasp trajectories
        self._traj_X_G_index = self.DeclareAbstractState( # pose traj
            AbstractValue.Make(PiecewisePose())
        )
        self._traj_wsg_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )
        self._times_index = self.DeclareAbstractState(
            AbstractValue.Make({"initial": 0.0})
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
        self.DeclareVectorOutputPort("wsg_position", 1, self.CalcWsgPosition)

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

        self._q0_index = self.DeclareDiscreteState(num_positions)  # for q0
        self.default_home = default_home
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)

        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)

        self.grasp_node = GraspListener(gripper_model_path=gripper_model_path)
        self.q_pregrasp1 = None
        self.q_pregrasp2 = None
        self.q_display_center = None
        self.q_preplace = None
        self.X_WG1 = None
        self.X_WG2 = None
        self.meshcat = meshcat
        self.plant = plant
        self._iiwa_controller_plant = controller_plant
        self.velocity_limits = 0.4 * np.ones(7)
        self.acceleration_limits = 0.4 * np.ones(7)
        self.display_velocity_limits = 0.2 * np.ones(7)
        # self.display_velocity_limits[6] = 0.05
        self.display_acceleration_limits = 0.2 * np.ones(7)
        self.regions = None #regions
        self.object_com = None
        self.object_dims = None
        self.object_rot = None
        self.regions1 = regions1
        self.regions2 = regions2
        self.traj_dir = traj_dir
        self.use_offline_regions1 = False if regions1 is None else True
        self.use_offline_regions2 = False if regions1 is None else True
        self.gripper_length = gripper_length
        self.pregrasp_dist = pregrasp_dist
        self.eef_to_gripper_length = eef_to_gripper_length
        self.joint_limits = np.zeros((7, 2))
        for i in range(7):
            joint = controller_plant.GetJointByName("iiwa_joint_%i" % (i + 1))
            self.joint_limits[i, 0] = joint.position_lower_limits()
            self.joint_limits[i, 1] = joint.position_upper_limits()

        if not self.use_offline_regions1 or not self.use_offline_regions1:
            if models_path == None:
                raise Exception("Must include models path to generate regions online")
            
        self.models_path = models_path
        self.savedir = dirstr
        self.no_obstacles = no_obstacles
        self.done = False

        self.pcd0 = None
        self.pcd1 = None
        self.pcd2 = None
        self.pcd3 = None

    def Update(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        current_time = context.get_time()
        times = context.get_abstract_state(int(self._times_index)).get_value()

        if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            if current_time - times["initial"] > 1.0:
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
                ).set_value(PlannerState.SCANNING1)
                # Update scanning state
                state.get_mutable_abstract_state(
                    int(self._scan_mode_index)
                ).set_value(ScanState.IDLE)
            return
        if mode == PlannerState.SCANNING1:
            self.GetPointCloud(context, state, PlannerState.GO_TO_PREGRASP1)
            input("Next: Pregrasp 1 (gcs)") # pause for debugging
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
                ).set_value(PickState.PREPICK)
                self.PlanPickAndDisplay(context, state)
                input("Next: Pick 1 (IK + toppra)") # pause for debugging
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
                # Update scanning state
                state.get_mutable_abstract_state(
                    int(self._scan_mode_index)
                ).set_value(ScanState.IDLE)
            return
        if mode == PlannerState.SCANNING2:
            self.GetPointCloud(context, state, PlannerState.GO_TO_PREGRASP2)
            input("Next: Pregrasp 2 (gcs)") # pause for debugging
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
                ).set_value(PickState.PREPICK)
                self.PlanPickAndDisplay(context, state)
                input("Next: Pick 1 (IK + toppra)") # pause for debugging
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
                ).set_value(PlannerState.DONE)
                self.done = True
            return
        
    def UpdateInGrasp(self, context, state, after_grasp_state):
        pick_mode = context.get_abstract_state(int(self._pick_mode_index)).get_value()
        traj_q = context.get_abstract_state(
            int(self._current_joint_traj_idx)
        ).get_value().trajectory
        start_time = context.get_abstract_state(
            int(self._current_joint_traj_idx)
        ).get_value().start_time_s

        if pick_mode == PickState.PREPICK:
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._pick_mode_index)
                ).set_value(PickState.CLOSING)
                self.PlanGripper(context, state, "close")
        if pick_mode == PickState.CLOSING:
            if context.get_time() > self._gripper_traj_end_time:
                state.get_mutable_abstract_state(
                    int(self._pick_mode_index)
                ).set_value(PickState.POSTPICK)
                self.GoToPostpick(context, state)
                input("Next: GoToPostpick (IK + toppra)") # pause for debugging
        if pick_mode == PickState.POSTPICK:
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._pick_mode_index)
                ).set_value(PickState.TO_DISPLAY)
                self.GoToDisplay(context, state)
                input("Next: GoToDisplay (gcs)") # pause for debugging
        if pick_mode == PickState.TO_DISPLAY:
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._pick_mode_index)
                ).set_value(PickState.DISPLAY)
                self.DoDisplay(context, state)
                input("Next: Display (IK + toppra)") # pause for debugging
        if pick_mode == PickState.DISPLAY:
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._pick_mode_index)
                ).set_value(PickState.TO_PREPLACE)
                self.GoToPreplace(context, state)
                input("Next: GoToPreplace (gcs)") # pause for debugging
        if pick_mode == PickState.TO_PREPLACE:
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._pick_mode_index)
                ).set_value(PickState.PLACE)
                self.GoToPlace(context, state)
                input("Next: GoToPlace (IK + toppra)") # pause for debugging
        if pick_mode == PickState.PLACE:
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._pick_mode_index)
                ).set_value(PickState.OPENING)
                self.PlanGripper(context, state, "open")
        if pick_mode == PickState.OPENING:
            if context.get_time() > self._gripper_traj_end_time:
                state.get_mutable_abstract_state(
                    int(self._pick_mode_index)
                ).set_value(PickState.POSTPLACE)
                self.DoPlace(context, state)
                input("Next: Place (IK + toppra)") # pause for debugging
        if pick_mode == PickState.POSTPLACE:
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._pick_mode_index)
                ).set_value(PickState.IDLE)
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(after_grasp_state)
                self.GoHome(context, state)
                input("Next: Postgrasp (gcs)") # pause for debugging
        return


    def GetPointCloud(self, context, state, after_scan_state):
        start = time.time()
        # Get manipuland pcd 
        cloud0 = self.GetInputPort("cloud_front").Eval(context)
        pcd0 = cloud0.Crop(lower_xyz=[0.23, -0.17, 0.071], upper_xyz=[0.57, 0.17, 0.27])
        pcd0.EstimateNormals(radius=0.1, num_closest=30)
        pcd0.FlipNormalsTowardPoint(self._X_WC0.translation())

        cloud1 = self.GetInputPort("cloud_back_left").Eval(context)
        pcd1 = cloud1.Crop(lower_xyz=[0.23, -0.17, 0.071], upper_xyz=[0.57, 0.17, 0.27])
        pcd1.EstimateNormals(radius=0.1, num_closest=30)
        pcd1.FlipNormalsTowardPoint(self._X_WC1.translation())

        cloud2 = self.GetInputPort("cloud_back_right").Eval(context)
        pcd2 = cloud2.Crop(lower_xyz=[0.23, -0.17, 0.071], upper_xyz=[0.57, 0.17, 0.27])
        pcd2.EstimateNormals(radius=0.1, num_closest=30)
        pcd2.FlipNormalsTowardPoint(self._X_WC2.translation())

        merged_pcd = Concatenate([pcd0, pcd1, pcd2])
        down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)
        
        # remove outliers
        o3d_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(down_sampled_pcd.xyzs().T))
        cl, ind = o3d_cloud.remove_statistical_outlier(
            nb_neighbors=int(down_sampled_pcd.xyzs().shape[1] // 10), 
            std_ratio=1.0)
        filtered_pts = np.asarray(o3d_cloud.points)[ind]
        down_sampled_pcd.resize(filtered_pts.shape[0])
        down_sampled_pcd.mutable_xyzs()[:] = filtered_pts.T
        self.current_manipuland_pcd = down_sampled_pcd
        print("object pcd got in", time.time()-start, "seconds")

        start =time.time()
        # Get table pcd 
        cloud0 = self.GetInputPort("cloud_front").Eval(context)
        pcd0 = cloud0.Crop(lower_xyz=[0.1, -0.3, 0.0], upper_xyz=[0.7, 0.3, 0.07]).VoxelizedDownSample(voxel_size=0.01)
        pcd0.EstimateNormals(radius=0.1, num_closest=30)
        pcd0.FlipNormalsTowardPoint(self._X_WC0.translation())

        cloud1 = self.GetInputPort("cloud_back_left").Eval(context)
        pcd1 = cloud1.Crop(lower_xyz=[0.1, -0.3, 0.0], upper_xyz=[0.7, 0.3, 0.07]).VoxelizedDownSample(voxel_size=0.01)
        pcd1.EstimateNormals(radius=0.1, num_closest=30)
        pcd1.FlipNormalsTowardPoint(self._X_WC1.translation())

        cloud2 = self.GetInputPort("cloud_back_right").Eval(context)
        pcd2 = cloud2.Crop(lower_xyz=[0.1, -0.3, 0.0], upper_xyz=[0.7, 0.3, 0.07]).VoxelizedDownSample(voxel_size=0.01)
        pcd2.EstimateNormals(radius=0.1, num_closest=30)
        pcd2.FlipNormalsTowardPoint(self._X_WC2.translation())

        merged_pcd = Concatenate([pcd0, pcd1, pcd2])
        
        # remove outliers
        o3d_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(merged_pcd.xyzs().T))
        cl, ind = o3d_cloud.remove_statistical_outlier(
            nb_neighbors=int(merged_pcd.xyzs().shape[1] // 10), 
            std_ratio=1.0)
        filtered_pts = np.asarray(o3d_cloud.points)[ind]
        merged_pcd.resize(filtered_pts.shape[0])
        merged_pcd.mutable_xyzs()[:] = filtered_pts.T
        self.current_scene_pcd = merged_pcd
        print("table pcd got in", time.time()-start, "seconds")
        
        state.get_mutable_abstract_state(
            int(self._mode_index)
        ).set_value(after_scan_state)
        self.PlanToPregrasp(context, state)

        return

    def GoHome(self, context, state):
        '''
        Reset to default home position to move arm out of the way of the camera
        '''

        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        q_goal = context.get_discrete_state(self._q0_index).get_value().copy() # initial pose

        loaded_traj = False
        if mode != PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE and self.traj_dir is not None:
            if mode == PlannerState.GRASP1 and os.path.exists(os.path.join(self.traj_dir, "grasp1_gohome_traj")):
                traj = CompositeBezierCurveTrajectoryAttributes.load(self.traj_dir + "/grasp1_gohome_traj/").to_composite_bezier_curve_trajectory()
                loaded_traj = True
            if mode == PlannerState.GRASP2 and os.path.exists(os.path.join(self.traj_dir, "grasp2_gohome_traj")):
                traj = CompositeBezierCurveTrajectoryAttributes.load(self.traj_dir + "/grasp2_gohome_traj/").to_composite_bezier_curve_trajectory()
                loaded_traj = True

        if not loaded_traj:
            if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
                traj = plan_unconstrained_gcs_path_start_to_goal(
                    plant=self._iiwa_controller_plant, q_start=q, q_goal=q_goal, regions=None, no_obstacles=True
                )
            else:
                traj = plan_unconstrained_gcs_path_start_to_goal(
                    plant=self._iiwa_controller_plant, q_start=q, q_goal=q_goal, regions=self.regions, no_obstacles=self.no_obstacles
                )

            if traj is None:
                logging.error("Failed to find a path to the home positions.")
                exit(1)
        
        if mode == PlannerState.GRASP1:
            make_trajectory_save_dirs(self.savedir, "grasp1_gohome_traj")
            traj_attr = CompositeBezierCurveTrajectoryAttributes.from_composite_bezier_curve_trajectory(traj)
            traj_attr.log(self.savedir + "/grasp1_gohome_traj/")
        elif mode == PlannerState.GRASP2:
            make_trajectory_save_dirs(self.savedir, "grasp2_gohome_traj")
            traj_attr = CompositeBezierCurveTrajectoryAttributes.from_composite_bezier_curve_trajectory(traj)
            traj_attr.log(self.savedir + "/grasp2_gohome_traj/")


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

        # Find grasp candidate
        self.PlanGrasp(context, state)

        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        if mode == PlannerState.SCANNING1:
            q_goal = self.q_pregrasp1
        if mode == PlannerState.SCANNING2:
            q_goal = self.q_pregrasp2

        loaded_traj = False
        if self.traj_dir is not None:
            if mode == PlannerState.SCANNING1 and os.path.exists(os.path.join(self.traj_dir, "grasp1_pregrasp_traj")):
                traj = CompositeBezierCurveTrajectoryAttributes.load(self.traj_dir + "/grasp1_pregrasp_traj/").to_composite_bezier_curve_trajectory()
                loaded_traj = True
            if mode == PlannerState.SCANNING2 and os.path.exists(os.path.join(self.traj_dir, "grasp2_pregrasp_traj")):
                traj = CompositeBezierCurveTrajectoryAttributes.load(self.traj_dir + "/grasp2_pregrasp_traj/").to_composite_bezier_curve_trajectory()
                loaded_traj = True

        # Set gcs regions or generate if not given or not ignoring obstacles/loading trajectories
        if mode == PlannerState.SCANNING1:
            if not self.use_offline_regions1 and not self.no_obstacles and not loaded_traj:
                collision_checker = get_collision_checker(self.models_path, self.object_com, self.object_rot, self.object_dims)
                start = time.time()
                rrt_path = rrt_planning(q, q_goal, self.joint_limits, collision_checker)
                print("rrt_time", time.time() - start)
                start = time.time()
                if rrt_path is not None:
                    self.regions = get_regions_cci(rrt_path, self.current_manipuland_pcd.xyzs(), 0.005)
                    print("regions cci time", time.time() - start)
                else:
                    self.regions = []

                # make sure the start and pregrasp positions are in the regions
                q_in_regions = False
                q_goal_in_regions = False

                for region in self.regions:
                    if region.PointInSet(q_goal):
                        q_goal_in_regions = True

                if not q_goal_in_regions:
                    print("getting seeded region for q goal")
                    self.regions.append(get_seeded_region(self.models_path, self.object_com, self.object_rot, self.object_dims, q_goal))

                for region in self.regions:
                    if region.PointInSet(q):
                        q_in_regions = True

                if not q_in_regions:
                    print("getting seeded region for q")
                    self.regions.append(get_seeded_region(self.models_path, self.object_com, self.object_rot, self.object_dims, q))
               
                save_regions_pkl(self.regions, self.models_path, self.savedir, "regions_1")
            else:
                self.regions = self.regions1
        else:
            if not self.use_offline_regions2 and not self.no_obstacles and not loaded_traj:
                collision_checker = get_collision_checker(self.models_path, self.object_com, self.object_rot, self.object_dims)
                start = time.time()
                rrt_path = rrt_planning(q, q_goal, self.joint_limits, collision_checker)
                print("rrt_time", time.time() - start)
                start = time.time()
                if rrt_path is not None:
                    self.regions = get_regions_cci(rrt_path, self.current_manipuland_pcd.xyzs(), 0.005)
                    print("regions cci time", time.time() - start)
                else:
                    self.regions = []

                # make sure the start and pregrasp positions are in the regions
                q_in_regions = False
                q_goal_in_regions = False
                
                if self.regions is None:
                    self.regions = []
                    
                for region in self.regions:
                    if region.PointInSet(q):
                        q_in_regions = True
                
                if not q_in_regions:
                    print("getting seeded region for q")
                    self.regions.append(get_seeded_region(self.models_path, self.object_com, self.object_rot, self.object_dims, q))
                
                for region in self.regions:
                    if region.PointInSet(q_goal):
                        q_goal_in_regions = True

                if not q_goal_in_regions:
                    print("getting seeded region for q goal")
                    self.regions.append(get_seeded_region(self.models_path, self.object_com, self.object_rot, self.object_dims, q_goal))

                save_regions_pkl(self.regions, self.models_path, self.savedir, "regions_2")
            else:
                self.regions = self.regions2

        # input("regions generated, press [ENTER] to continue")

        if not loaded_traj:
            traj = plan_unconstrained_gcs_path_start_to_goal(
                plant=self._iiwa_controller_plant, q_start=q, q_goal=q_goal, regions=self.regions, no_obstacles=self.no_obstacles
            )
            if traj is None:
                logging.error("Failed to find a path to the grasping start positions.")
                exit(1)

        if mode == PlannerState.SCANNING1:
            make_trajectory_save_dirs(self.savedir, "grasp1_pregrasp_traj")
            traj_attr = CompositeBezierCurveTrajectoryAttributes.from_composite_bezier_curve_trajectory(traj)
            traj_attr.log(self.savedir + "/grasp1_pregrasp_traj/")
        else:
            make_trajectory_save_dirs(self.savedir, "grasp2_pregrasp_traj")
            traj_attr = CompositeBezierCurveTrajectoryAttributes.from_composite_bezier_curve_trajectory(traj)
            traj_attr.log(self.savedir + "/grasp2_pregrasp_traj/")

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
        
        if mode == PlannerState.SCANNING1:
            # if self.pcd0:
            #     self.meshcat.SetObject("cloud0", self.pcd0, point_size=0.0001, rgba=Rgba(1,0,0,1))
            # if self.pcd1:
            #     self.meshcat.SetObject("cloud1", self.pcd1, point_size=0.0001, rgba=Rgba(1,1,0,1))
            # if self.pcd2:
            #     self.meshcat.SetObject("cloud2", self.pcd2, point_size=0.0001, rgba=Rgba(0,1,0,1))
            # if self.pcd3:
            #     self.meshcat.SetObject("cloud3", self.pcd3, point_size=0.0001, rgba=Rgba(0,0,1,1))
            self.meshcat.SetObject("cloud", self.current_manipuland_pcd, point_size=0.001, rgba=Rgba(1,1,0,1))
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
            self.object_com = com
            pcd_points_axis_aligned = pcd_points @ rot_principal_component_to_axes.as_matrix().T
            dims = np.max(pcd_points_axis_aligned, axis=0) - np.min(pcd_points_axis_aligned, axis=0)
            self.object_dims = dims
            rot = RotationMatrix(rot_principal_component_to_axes.as_matrix().T)
            self.object_rot = rot
            AddMeshcatTriad(self.meshcat, "principal axis", 
                            X_PT=RigidTransform(rot,
                            [com[0], com[1], com[2]]))
            
            grasps_found = False
            while not grasps_found:
                # Planning first grasping trajectory
                self.grasp_node.compute_candidate_grasps(
                    self.current_manipuland_pcd,
                    pcd_with_background,
                    candidate_num=1,
                    num_samples=15,
                    random_seed=np.random.randint(1000),
                )

                grasp_pairs = self.grasp_node.get_best_grasps()
                print("grasp pairs:", len(grasp_pairs))
                for i in range(len(grasp_pairs)):
                    print("Grasp Pair:", grasp_pairs[i])

                    X_WG1 = grasp_pairs[i][0]
                    X_WG2 = grasp_pairs[i][1]
                    X_WPregrasp1 = (RigidTransform(X_WG1) @ X_GE) @ X_GgraspGpregrasp
                    X_WPregrasp2 = (RigidTransform(X_WG2) @ X_GE) @ X_GgraspGpregrasp

                    # Check that IK passes
                    q = self.get_input_port(self._iiwa_position_index).Eval(context)
                    q_goal1 = solve_global_inverse_kinematics(
                        plant=self._iiwa_controller_plant,
                        X_G=X_WPregrasp1,
                        initial_guess=q,
                        position_tolerance=0.0,
                        orientation_tolerance=0.0,
                        gripper_frame_name="iiwa_link_7",
                        joint_limits=self.joint_limits
                    )
                    attempts = 0
                    while q_goal1 is None and attempts < 10:
                        print("trying global inverse kinematics with new initial guess randomized around q")
                        q_goal1 = solve_global_inverse_kinematics(
                            plant=self._iiwa_controller_plant,
                            X_G=X_WPregrasp1,
                            initial_guess=q + np.random.normal(0, np.pi/4, 7),
                            position_tolerance=0.0,
                            orientation_tolerance=0.0,
                            gripper_frame_name="iiwa_link_7",
                            joint_limits=self.joint_limits
                        )
                        attempts += 1

                    if q_goal1 is None:
                        continue
                        
                    # check if configuration is in collision with scene
                    if check_configuration_has_collisions(self.models_path, com, rot, dims, q_goal1):
                        print("grasp 1 configuration has collision with scene")
                        continue

                    q = self.get_input_port(self._iiwa_position_index).Eval(context)
                    q_goal2 = solve_global_inverse_kinematics(
                        plant=self._iiwa_controller_plant,
                        X_G=X_WPregrasp2,
                        initial_guess=q,
                        position_tolerance=0.0,
                        orientation_tolerance=0.0,
                        gripper_frame_name="iiwa_link_7",
                        joint_limits=self.joint_limits
                    )
                    attempts = 0
                    while q_goal2 is None and attempts < 10:
                        print("trying global inverse kinematics with new initial guess randomized around q")
                        q_goal2 = solve_global_inverse_kinematics(
                            plant=self._iiwa_controller_plant,
                            X_G=X_WPregrasp2,
                            initial_guess=q + np.random.normal(0, np.pi/4, 7),
                            position_tolerance=0.0,
                            orientation_tolerance=0.0,
                            gripper_frame_name="iiwa_link_7",
                            joint_limits=self.joint_limits
                        )
                        attempts += 1

                    if q_goal2 is None:
                        continue

                    # check if configuration is in collision with scene
                    if check_configuration_has_collisions(self.models_path, com, rot, dims, q_goal2):
                        print("grasp 2 configuration has collision with scene")
                        continue
                    
                    self.q_pregrasp1 = q_goal1
                    self.q_pregrasp2 = q_goal2
                    self.X_WG1 = X_WG1
                    self.X_WG2 = X_WG2
                    break
                
                if not(self.q_pregrasp1 is None or self.q_pregrasp2 is None):
                    grasps_found = True

            X_WG = self.X_WG1

            manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.current_manipuland_pcd.xyzs().T))
            manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

            gripper1_xyzs = self.grasp_node.hand_collision_model.to_pcd()
            gripper1_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper1_xyzs)).voxel_down_sample(0.005).transform((self.X_WG1 @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4()))
            gripper1_cloud.paint_uniform_color([1.0, 0.0, 0.0])

            gripper2_xyzs = self.grasp_node.hand_collision_model.to_pcd()
            gripper2_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper2_xyzs)).voxel_down_sample(0.005).transform((self.X_WG2 @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0]).GetAsMatrix4()))
            gripper2_cloud.paint_uniform_color([0.0, 1.0, 0.0])

            viz_geoms = [manipuland_cloud, gripper1_cloud, gripper2_cloud]
            o3d.visualization.draw_plotly(viz_geoms)
        else:
            X_WG = self.X_WG2

        X_WG = RigidTransform(X_WG)
        print(X_WG)

        X_WE = X_WG.multiply(X_GE)

        # Store grasp pose to use later when making pick + display trajectory
        state.get_mutable_abstract_state(self._grasp_X_G_index).set_value(X_WE)

        # # visualize grasp in o3d
        # manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.current_manipuland_pcd.xyzs().T))
        # manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])

        # gripper_xyzs = self.grasp_node.hand_collision_model.to_pcd()
        # gripper_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gripper_xyzs)).voxel_down_sample(0.005).transform((X_WG @ RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2),[0,0,0])).GetAsMatrix4())
        # gripper_cloud.paint_uniform_color([1.0, 0.0, 0.0])

        # viz_geoms = [manipuland_cloud, gripper_cloud]
        # o3d.visualization.draw_plotly(viz_geoms)

        return X_WE

    def PlanPickAndDisplay(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        X_G_pick = context.get_abstract_state(
            int(self._grasp_X_G_index)
        ).get_value()
        X_G_prepick = self.GetInputPort("body_poses").Eval(context)[int(self._ee_index)]

        X_G = {
            "pick": X_G_pick,
            "prepick": X_G_prepick
        }

        X_G["display_traj"] = yaw_display_traj
        place_flipped = False #(mode == PlannerState.GO_TO_PREGRASP1)
        X_G, times = MakePickAndDisplayGripperFrames(X_G, self.gripper_length, self.pregrasp_dist, place_flipped)

        state.get_mutable_abstract_state(int(self._times_index)).set_value(
            times
        )

        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        traj_q1, traj_q2, traj_q3, traj_q4, traj_q5, self.q_display_center, self.q_preplace = MakePickAndDisplayJointPositionsTrajectory(X_G, times, self._iiwa_controller_plant, q, self.q_pregrasp1, place_flipped, 8, joint_limits=self.joint_limits)
        
        toppra_traj_pick = reparameterize_with_toppra(
            trajectory=traj_q1,
            plant=self._iiwa_controller_plant,
            velocity_limits=self.velocity_limits,
            acceleration_limits=self.acceleration_limits,
            num_grid_points=100,
            is_pl=True,
        )

        # start pick traj
        current_time = context.get_time()
        state.get_mutable_abstract_state(self._current_joint_traj_idx).set_value(
            TrajectoryWithTimingInformation(
                trajectory=toppra_traj_pick,
                start_time_s=current_time,
            )
        )

        # Store the display traj for later
        state.get_mutable_abstract_state(self._to_postpick_traj_index).set_value(
            traj_q2
        )
        state.get_mutable_abstract_state(self._display_traj_index).set_value(
            traj_q3
        )
        state.get_mutable_abstract_state(self._to_place_traj_index).set_value(
            traj_q4
        )
        state.get_mutable_abstract_state(self._place_traj_index).set_value(
            traj_q5
        )
    
    def GoToPostpick(self, context, state):
        current_time = context.get_time()
        display_traj = context.get_abstract_state(int(self._to_postpick_traj_index)).get_value()
        toppra_traj = reparameterize_with_toppra(
            trajectory=display_traj,
            plant=self._iiwa_controller_plant,
            velocity_limits=self.display_velocity_limits,
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

    def GoToDisplay(self, context, state):
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        traj = plan_unconstrained_gcs_path_start_to_goal(
            plant=self._iiwa_controller_plant, q_start=q, q_goal=self.q_display_center, regions=None, no_obstacles=True
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

    def DoDisplay(self, context, state):
        current_time = context.get_time()
        display_traj = context.get_abstract_state(int(self._display_traj_index)).get_value()
        toppra_traj = reparameterize_with_toppra(
            trajectory=display_traj,
            plant=self._iiwa_controller_plant,
            velocity_limits=self.display_velocity_limits,
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
    
    def GoToPreplace(self, context, state):
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        traj = plan_unconstrained_gcs_path_start_to_goal(
            plant=self._iiwa_controller_plant, q_start=q, q_goal=self.q_preplace, regions=None, no_obstacles=True
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

    def GoToPlace(self, context, state):
        current_time = context.get_time()
        display_traj = context.get_abstract_state(int(self._to_place_traj_index)).get_value()
        toppra_traj = reparameterize_with_toppra(
            trajectory=display_traj,
            plant=self._iiwa_controller_plant,
            velocity_limits=self.display_velocity_limits,
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

    def DoPlace(self, context, state):
        current_time = context.get_time()
        place_traj = context.get_abstract_state(int(self._place_traj_index)).get_value()
        toppra_traj = reparameterize_with_toppra(
            trajectory=place_traj,
            plant=self._iiwa_controller_plant,
            velocity_limits=self.velocity_limits,
            acceleration_limits=self.acceleration_limits,
            num_grid_points=100,
            is_pl=True,
        )

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

        self._gripper_traj_end_time = current_time + 3.0

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
                context.get_abstract_state(int(self._traj_X_G_index))
                .get_value()
                .GetPose(context.get_time())
            )
            return

        # Command the current position (note: this is not particularly good if the velocity is non-zero)
        output.set_value(
            self.GetInputPort("body_poses")[int(self._ee_index)]
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
        
        # keep closed if displaying
        if (pick_mode == PickState.POSTPICK or 
            pick_mode == PickState.TO_DISPLAY or 
            pick_mode == PickState.DISPLAY or 
            pick_mode == PickState.TO_PREPLACE or 
            pick_mode == PickState.PLACE or 
            pick_mode == PickState.CLOSING):
            output.SetFromVector([closed])
            return

        # Command the open position
        output.SetFromVector([opened])
    
    def GetState(self, context, output):
        state = context.get_abstract_state(int(self._mode_index)).get_value()
        output.set_value(state)

    def Initialize(self, context, discrete_state):
        discrete_state.set_value(
            int(self._q0_index),
            self.default_home
        )
