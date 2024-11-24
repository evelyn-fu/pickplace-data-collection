import numpy as np
import logging
import pickle
import datetime
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
from pydrake.geometry import Rgba

from manipulation.meshcat_utils import AddMeshcatTriad
from enum import Enum

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
    builder = RobotDiagramBuilder()
    plant = builder.plant()
    builder.parser().AddModels(models_path)
    # builder.parser().AddModelsFromString(bounding_box_urdf, "urdf")
    diagram = builder.Build()

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, q_nominal)

    # iris_options = IrisOptions(require_sample_point_is_contained=True)
    iris_options = IrisOptions()
    iris_options.iteration_limit = 10
    # increase num_collision_infeasible_samples to improve the (probabilistic)
    # certificate of having no collisions.
    iris_options.num_collision_infeasible_samples = 3
    iris_options.require_sample_point_is_contained = True
    iris_options.relative_termination_threshold = 0.01
    iris_options.termination_threshold = -1
    region = IrisInConfigurationSpace(plant, plant_context, iris_options)
    print("region:", region)

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
    MOVE = 4
    OPENING = 5
    POSTPLACE = 6

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

q_home = [0.0, 0.4, 0.0, -1.2, 0.0, 1.0, -1.57]

class TwoGraspPlanner(LeafSystem):
    def __init__(
            self, 
            plant,
            controller_plant, 
            eef_body_index,
            X_EefC,
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
            eef_to_gripper_length=0.19,
        ):
        LeafSystem.__init__(self)

        # For grasp planner
        self.current_pcd = PointCloud(0)
        self.DeclareAbstractInputPort("cloud_handeye", AbstractValue.Make(PointCloud(0)))
        self.DeclareAbstractInputPort("cloud_stationary", AbstractValue.Make(PointCloud(0)))
        self._eef_body_index = eef_body_index
        self._X_EefC = X_EefC
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

        # Store the path parameterized display trajectory
        self._display_traj_index = self.DeclareAbstractState(
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
        self.X_WG1 = None
        self.X_WG2 = None
        self.meshcat = meshcat
        self.plant = plant
        self._iiwa_controller_plant = controller_plant
        self.velocity_limits = 0.4 * np.ones(7)
        self.acceleration_limits = 0.4 * np.ones(7)
        self.display_velocity_limits = 0.2 * np.ones(7)
        self.display_velocity_limits[6] = 0.05
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
            traj_q= context.get_abstract_state(
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
            # self.PlanToPregrasp(context, state)
            # state.get_mutable_abstract_state(
            #     int(self._mode_index)
            # ).set_value(PlannerState.GO_TO_PREGRASP1)
            self.GetPointCloud(context, state, PlannerState.GO_TO_PREGRASP1)
            return
        if mode == PlannerState.GO_TO_PREGRASP1:
            traj_q= context.get_abstract_state(
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
            return
        if mode == PlannerState.GRASP1:
            self.UpdateInGrasp(context, state, PlannerState.GO_HOME1)
            return
        if mode == PlannerState.GO_HOME1:
            traj_q= context.get_abstract_state(
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
            state.get_mutable_abstract_state(
                int(self._mode_index)
            ).set_value(PlannerState.GO_TO_PREGRASP2)
            self.PlanToPregrasp(context, state)
            # self.GetPointCloud(context, state, PlannerState.GO_TO_PREGRASP2)
            return
        if mode == PlannerState.GO_TO_PREGRASP2:
            traj_q= context.get_abstract_state(
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
            return
        if mode == PlannerState.GRASP2:
            self.UpdateInGrasp(context, state, PlannerState.GO_HOME2)
            return
        if mode == PlannerState.GO_HOME2:
            traj_q= context.get_abstract_state(
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
                ).set_value(PickState.MOVE)
                self.DoDisplay(context, state)
        if pick_mode == PickState.MOVE:
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
        if pick_mode == PickState.POSTPLACE:
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._pick_mode_index)
                ).set_value(PickState.IDLE)
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(after_grasp_state)
                self.GoHome(context, state)
        return


    def GetPointCloud(self, context, state, after_scan_state):
        scan_mode = context.get_abstract_state(int(self._scan_mode_index)).get_value()
        traj_q = context.get_abstract_state(
            int(self._current_joint_traj_idx)
        ).get_value().trajectory
        start_time = context.get_abstract_state(
            int(self._current_joint_traj_idx)
        ).get_value().start_time_s

        body_poses = self.GetInputPort("body_poses").Eval(context)
        X_WC = body_poses[self._eef_body_index] @ self._X_EefC 
        AddMeshcatTriad(self.meshcat, "X_WC", 
                        X_PT=X_WC)

        def set_traj(scan_traj):
            breaks = np.linspace(0, scan_traj.end_time(), int(1e3), endpoint=False)
            knots = scan_traj.vector_values(breaks)

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
        
        def get_pcd(start_new_pcd = False, ind=0):
            body_poses = self.GetInputPort("body_poses").Eval(context)
            cloud = self.GetInputPort("cloud_handeye").Eval(context)
            new_pcd = cloud.Crop(lower_xyz=[0.23, -0.17, 0.055], upper_xyz=[0.57, 0.17, 0.27])
            new_pcd.EstimateNormals(radius=0.1, num_closest=30)
            X_WC = body_poses[self._eef_body_index] @ self._X_EefC 
            new_pcd.FlipNormalsTowardPoint(X_WC.translation())
            if start_new_pcd:
                return new_pcd
            
            if ind == 0:
                self.pcd0 = new_pcd
            if ind == 1:
                self.pcd1 = new_pcd
            if ind == 2:
                self.pcd2 = new_pcd
            if ind == 3:
                self.pcd3 = new_pcd

        if scan_mode == ScanState.IDLE:
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._scan_mode_index)
                ).set_value(ScanState.GO_TO_1)

                # start pcd with front camera view
                body_poses = self.GetInputPort("body_poses").Eval(context)
                cloud = self.GetInputPort("cloud_stationary").Eval(context)
                new_pcd = cloud.Crop(lower_xyz=[0.23, -0.17, 0.055], upper_xyz=[0.57, 0.17, 0.27])
                new_pcd.EstimateNormals(radius=0.1, num_closest=30)
                X_WC = body_poses[self._eef_body_index] @ self._X_EefC 
                new_pcd.FlipNormalsTowardPoint(X_WC.translation())
                self.current_pcd = new_pcd
                self.pcd0 = new_pcd

                # load trajectory to first camera view
                traj = CompositeBezierCurveTrajectoryAttributes.load(self._scanning_traj_dir + "/to1/").to_composite_bezier_curve_trajectory()
                set_traj(traj)
                return
        if scan_mode == ScanState.GO_TO_1:
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._scan_mode_index)
                ).set_value(ScanState.GO_TO_2)

                # update pcd (skip for now, too close to object)
                get_pcd(ind=1)

                # load trajectory to next camera view
                traj = CompositeBezierCurveTrajectoryAttributes.load(self._scanning_traj_dir + "/to2/").to_composite_bezier_curve_trajectory()
                set_traj(traj)
                return
        if scan_mode == ScanState.GO_TO_2:
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._scan_mode_index)
                ).set_value(ScanState.GO_TO_3)

                # update pcd
                get_pcd(ind=2)

                # load trajectory to next camera view
                traj = CompositeBezierCurveTrajectoryAttributes.load(self._scanning_traj_dir + "/to3/").to_composite_bezier_curve_trajectory()
                set_traj(traj)
                return
        if scan_mode == ScanState.GO_TO_3:
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._scan_mode_index)
                ).set_value(ScanState.GO_HOME)

                # update pcd
                get_pcd(ind=3)

                # merge all and downsample
                # note: camera calibration is bad, manual tuning is added here
                pcd_components = []
                if self.pcd0:
                    X_adjust = RigidTransform(RotationMatrix(),[0.0, 0.0, -0.0])
                    transformed_xyzs = X_adjust @ self.pcd0.xyzs()
                    self.pcd0.mutable_xyzs()[:] = transformed_xyzs
                    pcd_components.append(self.pcd0)
                if self.pcd1:
                    X_adjust = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 0)),[0.0, -0.005, 0.0])
                    transformed_xyzs = X_adjust @ self.pcd1.xyzs()
                    self.pcd1.mutable_xyzs()[:] = transformed_xyzs
                    pcd_components.append(self.pcd1)
                if self.pcd2:
                    X_adjust = RigidTransform(RotationMatrix(RollPitchYaw(0.0, -0.01, 0.03)),[-0.01, -0.01, 0.0])
                    transformed_xyzs = X_adjust @ self.pcd2.xyzs()
                    self.pcd2.mutable_xyzs()[:] = transformed_xyzs
                    pcd_components.append(self.pcd2)
                if self.pcd3:
                    X_adjust = RigidTransform(RotationMatrix(RollPitchYaw(0.0, 0.0, 0.0)),[0.0, 0.005, 0.0])
                    transformed_xyzs = X_adjust @ self.pcd3.xyzs()
                    self.pcd3.mutable_xyzs()[:] = transformed_xyzs
                    pcd_components.append(self.pcd3)

                # merge
                merged_pcd = Concatenate(pcd_components)
                
                # downsample
                down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)
                
                # remove outliers
                o3d_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(down_sampled_pcd.xyzs().T))
                cl, ind = o3d_cloud.remove_statistical_outlier(
                    nb_neighbors=int(down_sampled_pcd.xyzs().shape[1] // 10), 
                    std_ratio=1.0)
                filtered_pts = np.asarray(o3d_cloud.points)[ind]
                down_sampled_pcd.resize(filtered_pts.shape[0])
                down_sampled_pcd.mutable_xyzs()[:] = filtered_pts.T
                self.current_pcd = down_sampled_pcd

                # load trajectory to return home
                traj = CompositeBezierCurveTrajectoryAttributes.load(self._scanning_traj_dir + "/to_home/").to_composite_bezier_curve_trajectory()
                set_traj(traj)
                return
        if scan_mode == ScanState.GO_HOME:
            if context.get_time() > traj_q.end_time() + start_time:
                state.get_mutable_abstract_state(
                    int(self._scan_mode_index)
                ).set_value(ScanState.DONE)
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(after_scan_state)
                self.PlanToPregrasp(context, state)
                return

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
                # self.regions = get_regions(self.models_path, self.object_com, self.object_rot, self.object_dims)
                q_in_regions = False
                q_goal_in_regions = False

                if self.regions is None:
                    self.regions = []

                for region in self.regions:
                    if region.PointInSet(q_goal):
                        q_goal_in_regions = True

                if not q_goal_in_regions:
                    print("getting seeded region for q goal")
                    self.regions.append(get_seeded_region(self.models_path, self.object_com, self.object_rot, self.object_dims, q_goal))

                for region in self.regions:
                    if region.PointInSet(q):
                        q_in_regions = True

                # make sure the start and pregrasp positions are in the regions
                if not q_in_regions:
                    print("getting seeded region for q")
                    self.regions.append(get_seeded_region(self.models_path, self.object_com, self.object_rot, self.object_dims, q))
                
                
                # if not q_in_regions:
                #     raise Exception("q not in regions?")
                
                # if not q_goal_in_regions:
                #     raise Exception("q_goal not in regions?")

                save_regions_pkl(self.regions, self.models_path, self.savedir, "regions_1")
            else:
                self.regions = self.regions1
        else:
            if not self.use_offline_regions2 and not self.no_obstacles and not loaded_traj:
                # self.regions = get_regions(self.models_path, self.object_com, self.object_rot, self.object_dims)

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

        input("regions generated, press [ENTER] to continue")

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
        
        pcd_with_floor = self.current_pcd # Includes some floor on purpose
        object_pcd = pcd_with_floor.Crop(lower_xyz=[0.23, -0.17, 0.063], upper_xyz=[0.57, 0.17, 0.27]) # just object
        if self.pcd0:
            self.meshcat.SetObject("cloud0", self.pcd0, point_size=0.0001, rgba=Rgba(1,0,0,1))
        if self.pcd1:
            self.meshcat.SetObject("cloud1", self.pcd1, point_size=0.0001, rgba=Rgba(1,1,0,1))
        if self.pcd2:
            self.meshcat.SetObject("cloud2", self.pcd2, point_size=0.0001, rgba=Rgba(0,1,0,1))
        if self.pcd3:
            self.meshcat.SetObject("cloud3", self.pcd3, point_size=0.0001, rgba=Rgba(0,0,1,1))
        self.meshcat.SetObject("cloud", object_pcd, point_size=0.001)
        # self.meshcat.SetObject("cloud_w_floor", pcd_with_floor, point_size=0.001, rgba=Rgba(1,1,0,1))

        pcd_points = object_pcd.xyzs().T
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

        # get end effector pose from grasp pose
        X_GE = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 0)), [0, 0, -self.eef_to_gripper_length])

        # pregrasp is negative z in the gripper frame
        X_GgraspGpregrasp = RigidTransform([0, 0.0, -self.pregrasp_dist])
        
        if mode == PlannerState.SCANNING1:
            # Planning first grasping trajectory
            self.grasp_node.compute_candidate_grasps(
                object_pcd,
                pcd_with_floor,
                candidate_num=1,
                num_samples=15,
                random_seed=5,
            )

            grasp_pairs = self.grasp_node.get_best_grasps()
            print("grasp pairs:", len(grasp_pairs))
            X_ee = None
            X_pregrasp = None
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
                    )
                    attempts += 1

                if q_goal1 is None:
                    continue
                    
                # check if configuration is in collision with scene
                if check_configuration_has_collisions(self.models_path, com, rot, dims, q_goal1):
                    continue

                q = self.get_input_port(self._iiwa_position_index).Eval(context)
                q_goal2 = solve_global_inverse_kinematics(
                    plant=self._iiwa_controller_plant,
                    X_G=X_WPregrasp2,
                    initial_guess=q,
                    position_tolerance=0.0,
                    orientation_tolerance=0.0,
                    gripper_frame_name="iiwa_link_7",
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
                    )
                    attempts += 1

                if q_goal2 is None:
                    continue

                # check if configuration is in collision with scene
                if check_configuration_has_collisions(self.models_path, com, rot, dims, q_goal2):
                    continue
                
                self.q_pregrasp1 = q_goal1
                self.q_pregrasp2 = q_goal2
                self.X_WG1 = X_WG1
                self.X_WG2 = X_WG2
                X_ee = RigidTransform(X_WG1) @ X_GE
                X_pregrasp = (RigidTransform(X_WG1) @ X_GE) @ X_GgraspGpregrasp
                break
            
            if self.q_pregrasp1 is None or self.q_pregrasp2 is None:
                logging.error(
                    "Failed to solve inverse kinematics for any pair of grasps (cry)"
                )
                exit(1)

            X_WG = self.X_WG1
            manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(object_pcd.xyzs().T))
            manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])
            viz_geoms = [manipuland_cloud]
            viz_geoms.append(self.grasp_node.make_gripper_line_set(self.X_WG1, [0.0, 1.0, 0.0]))
            viz_geoms.append(self.grasp_node.make_gripper_line_set(self.X_WG2, [1.0, 0.0, 0.0]))
            viz_geoms.append(self.grasp_node.make_triad_line_set(RigidTransform().GetAsMatrix4()))
            o3d.visualization.draw_geometries(viz_geoms)
        else:
            X_WG = self.X_WG2

        X_WG = RigidTransform(X_WG)
        print(X_WG)

        X_WE = X_WG.multiply(X_GE)

        # Store grasp pose to use later when making pick + display trajectory
        state.get_mutable_abstract_state(self._grasp_X_G_index).set_value(X_WE)

        # # visualize grasp in o3d
        # manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(object_pcd.xyzs().T))
        # manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])
        # viz_geoms = [manipuland_cloud]
        # viz_geoms.append(self.grasp_node.make_gripper_line_set(X_WG.GetAsMatrix4(), [0.0, 1.0, 0.0]))
        # viz_geoms.append(self.grasp_node.make_gripper_line_set(X_WE.GetAsMatrix4(), [1.0, 0.0, 0.0]))
        # o3d.visualization.draw_geometries(viz_geoms)

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
        traj_q1, traj_q2, traj_q3 = MakePickAndDisplayJointPositionsTrajectory(X_G, times, self._iiwa_controller_plant, q, 8)
        
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
        state.get_mutable_abstract_state(self._display_traj_index).set_value(
            traj_q2
        )
        state.get_mutable_abstract_state(self._place_traj_index).set_value(
            traj_q3
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
        closed = np.array([0.02])
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
        closed = np.array([0.02])

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
        if pick_mode == PickState.MOVE or pick_mode == PickState.CLOSING:
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
