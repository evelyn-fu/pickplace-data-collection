import numpy as np
import logging
import pickle
import datetime
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as R
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
    builder.parser().AddModelsFromString(bounding_box_urdf, "urdf")
    diagram = builder.Build()

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, q_nominal)

    iris_options = IrisOptions(require_sample_point_is_contained=True)
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

class PlannerState(Enum):
    WAIT_FOR_OBJECTS_TO_SETTLE = 1
    GO_TO_PREGRASP1 = 2
    GRASP1 = 3
    GO_HOME1 = 4 # Reset arm out of the way of cameras in order to plan for next grasp
    GO_TO_PREGRASP2 = 5
    GRASP2 = 6
    GO_HOME2 = 7
    DONE = 8

class PickState(Enum):
    IDLE = 1
    PREPICK = 2
    CLOSING = 3
    MOVE = 4
    OPENING = 5
    POSTPLACE = 6

default_home_pose = RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, 0)), [0.5, 0.0, 0.5]) # arm out of the way of depth cameras

# pregrasp is negative z in the gripper frame
X_GgraspGpregrasp = RigidTransform([0, 0.0, -0.15])

yaw_display_traj_negative = []

yaw_display_traj_negative.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, 0)), [0.6, 0.0, 0.54]))
yaw_display_traj_negative.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, -np.pi/4)), [0.6, 0.0, 0.54]))
yaw_display_traj_negative.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, -np.pi/2)), [0.6, 0.0, 0.54]))
yaw_display_traj_negative.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, -3*np.pi/4)), [0.6, 0.0, 0.54]))
yaw_display_traj_negative.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, -np.pi)), [0.6, 0.0, 0.54]))
yaw_display_traj_negative.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, -5*np.pi)), [0.6, 0.0, 0.54]))
yaw_display_traj_negative.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, -3*np.pi/2)), [0.6, 0.0, 0.54]))

yaw_display_traj_positive = []

yaw_display_traj_positive.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, 0)), [0.6, 0.0, 0.54]))
yaw_display_traj_positive.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, np.pi/4)), [0.6, 0.0, 0.54]))
yaw_display_traj_positive.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, np.pi/2)), [0.6, 0.0, 0.54]))
yaw_display_traj_positive.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, 3*np.pi/4)), [0.6, 0.0, 0.54]))
yaw_display_traj_positive.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, np.pi)), [0.6, 0.0, 0.54]))
yaw_display_traj_positive.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, 5*np.pi)), [0.6, 0.0, 0.54]))
yaw_display_traj_positive.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, 3*np.pi/2)), [0.6, 0.0, 0.54]))



class TwoGraspPlanner(LeafSystem):
    def __init__(
            self, 
            plant,
            controller_plant, 
            camera_body_indices,
            meshcat, 
            regions1,
            regions2,
            traj_dir,
            models_path,
            dirstr,
            no_obstacles,
            gripper_model_path
        ):
        LeafSystem.__init__(self)

        # For grasp planner
        model_point_cloud = AbstractValue.Make(PointCloud(0))
        self.DeclareAbstractInputPort("cloud0_W", model_point_cloud)
        self.DeclareAbstractInputPort("cloud1_W", model_point_cloud)
        self.DeclareAbstractInputPort("cloud2_W", model_point_cloud)
        self._camera_body_indices = camera_body_indices

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

        # To get iiwa position
        num_positions = 7
        self._iiwa_position_index = self.DeclareVectorInputPort(
            "iiwa_position", num_positions
        ).get_index()

        self._q0_index = self.DeclareDiscreteState(num_positions)  # for q0
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)

        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)

        self.grasp_node = GraspListener(gripper_model_path=gripper_model_path)
        self.meshcat = meshcat
        self.plant = plant
        self._iiwa_controller_plant = controller_plant
        self.velocity_limits = 0.1 * np.ones(7)
        self.acceleration_limits = 0.1 * np.ones(7)
        self.regions = None #regions
        self.object_com = None
        self.object_dims = None
        self.object_rot = None
        self.regions1 = regions1
        self.regions2 = regions2
        self.traj_dir = traj_dir
        self.use_offline_regions1 = False if regions1 is None else True
        self.use_offline_regions2 = False if regions1 is None else True
        self.models_path = models_path
        self.savedir = dirstr
        self.no_obstacles = no_obstacles
        self.done = False

    def Update(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        current_time = context.get_time()
        times = context.get_abstract_state(int(self._times_index)).get_value()

        if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            if current_time - times["initial"] > 1.0:
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.GO_TO_PREGRASP1)
                self.PlanToPregrasp(context, state)
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
                ).set_value(PlannerState.GO_TO_PREGRASP2)
                self.PlanToPregrasp(context, state)
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


    def GoHome(self, context, state):
        '''
        Reset to default home position to move arm out of the way of the camera
        '''

        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        q_goal = context.get_discrete_state(self._q0_index).get_value().copy() # initial pose

        if self.traj_dir is not None:
            if mode == PlannerState.GRASP1:
                traj = CompositeBezierCurveTrajectoryAttributes.load(self.traj_dir + "/grasp1_gohome_traj/").to_composite_bezier_curve_trajectory()
            else:
                traj = CompositeBezierCurveTrajectoryAttributes.load(self.traj_dir + "/grasp2_gohome_traj/").to_composite_bezier_curve_trajectory()
        else:
            traj = plan_unconstrained_gcs_path_start_to_goal(
                plant=self._iiwa_controller_plant, q_start=q, q_goal=q_goal, regions=self.regions, no_obstacles=self.no_obstacles
            )
            if traj is None:
                logging.error("Failed to find a path to the home positions.")
                exit(1)

            mode = context.get_abstract_state(int(self._mode_index)).get_value()
        
        if mode == PlannerState.GRASP1:
            make_trajectory_save_dirs(self.savedir, "grasp1_gohome_traj")
            traj_attr = CompositeBezierCurveTrajectoryAttributes.from_composite_bezier_curve_trajectory(traj)
            traj_attr.log(self.savedir + "/grasp1_gohome_traj/")
        else:
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
        X_G_pick = self.PlanGrasp(context, state)
        
        X_G_prepick = X_G_pick @ X_GgraspGpregrasp

        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        q_goal = solve_global_inverse_kinematics(
            plant=self._iiwa_controller_plant,
            X_G=X_G_prepick,
            initial_guess=q,
            position_tolerance=0.01,
            orientation_tolerance=0.01,
            gripper_frame_name="iiwa_link_7",
        )
        print(q_goal)
        if q_goal is None:
            logging.error(
                "Failed to solve inverse kinematics for the grasping start pose."
            )
            exit(1)

        # Set gcs regions or generate if not given or not ignoring obstacles
        if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            if not self.use_offline_regions1 and not self.no_obstacles:
                self.regions = get_regions(self.models_path, self.object_com, self.object_rot, self.object_dims)

                # make sure the start and pregrasp positions are in the regions
                print("getting seeded region for q")
                self.regions.append(get_seeded_region(self.models_path, self.object_com, self.object_rot, self.object_dims, q))
                print("getting seeded region for q goal")
                self.regions.append(get_seeded_region(self.models_path, self.object_com, self.object_rot, self.object_dims, q_goal))

                q_in_regions = False
                q_goal_in_regions = False
                for region in self.regions:
                    if region.PointInSet(q):
                        q_in_regions = True
                    if region.PointInSet(q_goal):
                        q_goal_in_regions = True
                
                if not q_in_regions:
                    raise Exception("q not in regions?")
                
                if not q_goal_in_regions:
                    raise Exception("q_goal not in regions?")

                save_regions_pkl(self.regions, self.models_path, self.savedir, "regions_1")
            else:
                self.regions = self.regions1
        else:
            if not self.use_offline_regions2 and not self.no_obstacles:
                self.regions = get_regions(self.models_path, self.object_com, self.object_rot, self.object_dims)

                q_in_regions = False
                q_goal_in_regions = False
                for region in self.regions:
                    if region.PointInSet(q):
                        q_in_regions = True
                    if region.PointInSet(q_goal):
                        q_goal_in_regions = True
                
                if not q_in_regions:
                    print("getting seeded region for q")
                    self.regions.append(get_seeded_region(self.models_path, self.object_com, self.object_rot, self.object_dims, q))
                
                if not q_goal_in_regions:
                    print("getting seeded region for q goal")
                    self.regions.append(get_seeded_region(self.models_path, self.object_com, self.object_rot, self.object_dims, q_goal))

                save_regions_pkl(self.regions, self.models_path, self.savedir, "regions_2")
            else:
                self.regions = self.regions2

        if self.traj_dir is not None:
            if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
                traj = CompositeBezierCurveTrajectoryAttributes.load(self.traj_dir + "/grasp1_pregrasp_traj/").to_composite_bezier_curve_trajectory()
            else:
                traj = CompositeBezierCurveTrajectoryAttributes.load(self.traj_dir + "/grasp2_pregrasp_traj/").to_composite_bezier_curve_trajectory()
        else:
            traj = plan_unconstrained_gcs_path_start_to_goal(
                plant=self._iiwa_controller_plant, q_start=q, q_goal=q_goal, regions=self.regions, no_obstacles=self.no_obstacles
            )
            if traj is None:
                logging.error("Failed to find a path to the grasping start positions.")
                exit(1)

        if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
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

        # Get pcd and select grasp
        body_poses = self.get_input_port(3).Eval(context)
        pcd = []
        for i in range(3):
            cloud = self.get_input_port(i).Eval(context)

            # Crop to region of interest.
            pcd.append(cloud.Crop(lower_xyz=[0.3, -0.5, 0.121], upper_xyz=[1.0, 0.5, 0.32]))
            # Estimate normals
            pcd[i].EstimateNormals(radius=0.1, num_closest=30)

            # Flip normals toward camera
            X_WC = body_poses[self._camera_body_indices[i]]
            pcd[i].FlipNormalsTowardPoint(X_WC.translation())
        merged_pcd = Concatenate(pcd)

        down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)
        self.meshcat.SetObject("cloud", down_sampled_pcd, point_size=0.001)

        pcd_points = down_sampled_pcd.xyzs().T
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

        if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            # Planning first grasping trajectory
            print("align grasp axis", secondary_component)
            print("align minor axis", minor_component)
            # self.grasp_node.compute_candidate_grasps(
            #     down_sampled_pcd, 
            #     num_samples=5,
            #     random_seed=5, 
            #     align_grasp_axis=principal_component,  
            #     align_minor_axis=minor_component,
            #     split_axis=1, 
            #     minor_split_axis=0
            # )
            # grasps = [RigidTransform(
            #     R=RotationMatrix([
            #         [-0.26319746721208853, -0.9256698903407268, 0.2717762818380754],
            #         [-0.9566819996188846, 0.28676669060812554, 0.050243574344157924],
            #         [-0.12444534888196886, -0.24677949524677076, -0.961047988327011],
            #     ]),
            #     p=[0.4956561905765808, -0.008295454081149959, 0.3428158221666631],
            # )] # 1 
            grasps = [RigidTransform(
                R=RotationMatrix([
                    [-0.20844050647336543, -0.9779921178608655, 0.009163659920858486],
                    [-0.9669276384501052, 0.2074722940685329, 0.14834483204764537],
                    [-0.1469812820138356, 0.022060475877881697, -0.9888932390008593],
                ]),
                p=[0.6073802571650899, -0.016139619528128844, 0.351559001325315 + 0.01],
            )]
        else:
            # grasps = [RigidTransform(
            # R=RotationMatrix([
            #     [0.0834121306727252, -0.8851583536076943, -0.4577522315567158],
            #     [0.24095232073015768, 0.46363766580107135, -0.8526324495263393],
            #     [0.9669459114094168, -0.03917657321730244, 0.25195396508005563],
            # ]),
            # p=[0.5705643119928675, 0.06772450226931172, 0.1909821558295339],
            # )] # 1
            grasps = [RigidTransform(
            R=RotationMatrix([
                [-0.03442034895124868, 0.9994073003679098, 0.0005362363300874173],
                [-0.04829586151222898, -0.002199273198199581, 0.9988306527926499],
                [0.9982398255624083, 0.0343542016167908, 0.04834293632380032],
            ]) @ RotationMatrix(RollPitchYaw(0, -0.4, 0)),
            p=[0.5857679659224048, -0.10382563627445854 - 0.03, 0.24689332681875962],
            )]
            # Planning second grasping trajectory
            # self.grasp_node.compute_candidate_grasps(
            #     down_sampled_pcd, 
            #     num_samples=5,
            #     random_seed=1, 
            #     align_grasp_axis=secondary_component, 
            #     align_minor_axis=minor_component, 
            #     split_axis=2, 
            #     minor_split_axis=0
            # )
        
        # grasps = self.grasp_node.get_best_grasps(candidate_num=1)

        print(grasps)
        
        # get end effector pose from grasp pose
        X_GE = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 0)), [0, 0, -0.09])

        ee_grasps = [X_WG.multiply(X_GE) for X_WG in grasps]

        # Store grasp pose to use later when making pick + display trajectory
        state.get_mutable_abstract_state(self._grasp_X_G_index).set_value(ee_grasps[0])

        # visualize grasp in o3d
        manipuland_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(down_sampled_pcd.xyzs().T))
        manipuland_cloud.paint_uniform_color([0.0, 0.0, 1.0])
        viz_geoms = [manipuland_cloud]
        viz_geoms.append(self.grasp_node.make_gripper_line_set(grasps[0].GetAsMatrix4(), [0.0, 1.0, 0.0]))
        o3d.visualization.draw_geometries(viz_geoms)

        return ee_grasps[0]

    def PlanPickAndDisplay(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        X_G_pick = context.get_abstract_state(
            int(self._grasp_X_G_index)
        ).get_value()
        X_G_prepick = self.get_input_port(3).Eval(context)[int(self._ee_index)]

        X_G = {
            "pick": X_G_pick,
            "prepick": X_G_prepick
        }

        X_G["display_traj"] = (yaw_display_traj_negative, yaw_display_traj_positive)
        X_G, times = MakePickAndDisplayGripperFrames(X_G)

        state.get_mutable_abstract_state(int(self._times_index)).set_value(
            times
        )

        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        traj_q1, traj_q2, traj_q3 = MakePickAndDisplayJointPositionsTrajectory(X_G, times, self._iiwa_controller_plant, q)
        
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
        closed = np.array([0.0])
        current_time = context.get_time()

        traj_wsg_command = PiecewisePolynomial.FirstOrderHold(
            [current_time, current_time+3.0],
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
            self.get_input_port(3).Eval(context)[int(self._ee_index)]
        )
    
    def GetCurrentJointPositionTrajectory(self, context, output):
        current_joint_traj = context.get_abstract_state(
            self._current_joint_traj_idx
        ).get_value()
        output.set_value(current_joint_traj)

    def CalcWsgPosition(self, context, output):
        pick_mode = context.get_abstract_state(int(self._pick_mode_index)).get_value()
        opened = np.array([0.107])
        closed = np.array([0.0])

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

    def Initialize(self, context, discrete_state):
        discrete_state.set_value(
            int(self._q0_index),
            self.get_input_port(int(self._iiwa_position_index)).Eval(context),
        )