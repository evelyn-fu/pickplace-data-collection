import numpy as np
import logging
from scipy.spatial.transform import Rotation as R
from planning.grasp import GraspListener
from planning.toppra import reparameterize_with_toppra
from planning.trajectories import (
    MakeGripperCommandTrajectory,
    MakeGripperFrames,
    MakeGripperPoseTrajectory,
)
from planning.trajectory_sources import TrajectoryWithTimingInformationSource
from planning.gcs import plan_unconstrained_gcs_path_start_to_goal
from planning.inverse_kinematics import solve_global_inverse_kinematics
from iiwa_setup_dataclasses.trajectories import TrajectoryWithTimingInformation
from pydrake.systems.framework import LeafSystem, InputPortIndex
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
    PiecewisePolynomial
)

from manipulation.meshcat_utils import AddMeshcatTriad
from enum import Enum


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
    GO_TO_START1 = 2
    GRASP1 = 3
    GO_HOME = 4 # Reset arm out of the way of cameras in order to plan for next grasp
    GO_TO_START2 = 5
    GRASP2 = 6
    DONE = 7

default_home_pose = RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, np.pi/2)), [0.0, -0.5, 0.5]) # arm out of the way of depth cameras

default_display_traj = []

default_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, np.pi/2)), [0.0, -0.5, 0.5]))
default_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, np.pi/4)), [0.0, -0.5, 0.5]))
default_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, np.pi/2)), [0.0, -0.5, 0.5]))
default_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, np.pi)), [0.0, -0.5, 0.5]))
default_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, 3 * np.pi / 2)), [0.0, -0.5, 0.5]))
default_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, 7 * np.pi / 4)), [0.0, -0.5, 0.5]))
default_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, 3 * np.pi / 2)), [0.0, -0.5, 0.5]))
default_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, np.pi)), [0.0, -0.5, 0.5]))
default_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, np.pi/2)), [0.0, -0.5, 0.5]))


class TwoGraspPlanner(LeafSystem):
    def __init__(
            self, 
            plant,
            controller_plant, 
            camera_body_indices,
            meshcat
        ):
        LeafSystem.__init__(self)

        # For grasp planner
        model_point_cloud = AbstractValue.Make(PointCloud(0))
        self.DeclareAbstractInputPort("cloud0_W", model_point_cloud)
        self.DeclareAbstractInputPort("cloud1_W", model_point_cloud)
        self.DeclareAbstractInputPort("cloud2_W", model_point_cloud)
        self._camera_body_indices = camera_body_indices

        # for setting default iiwa position values
        self._gripper_body_index = plant.GetBodyByName("body").index()
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )

        # FSM state
        self._mode_index = self.DeclareAbstractState(
            AbstractValue.Make(PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE)
        )

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

        # output pose
        self.DeclareAbstractOutputPort(
            "X_WG",
            lambda: AbstractValue.Make(RigidTransform()),
            self.CalcGripperPose,
        )
        self.DeclareVectorOutputPort("wsg_position", 1, self.CalcWsgPosition)

        # output entire joint position trajectory
        self.DeclareAbstractOutputPort(
            "joint_position_trajectory",
            lambda: AbstractValue.Make(TrajectoryWithTimingInformation()),
            self.GetCurrentJointPositionTrajectory,
        )

        # For iiwa position control modes.
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
        self._X_G_init_index = self.DeclareAbstractState(
            AbstractValue.Make(RigidTransform())
        )
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)

        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)

        self.grasp_node = GraspListener()
        self.meshcat = meshcat
        self.plant = plant
        self._iiwa_controller_plant = controller_plant
        self.velocity_limits = 0.1 * np.ones(7)
        self.acceleration_limits = 0.1 * np.ones(7)

    def Update(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        current_time = context.get_time()
        times = context.get_abstract_state(int(self._times_index)).get_value()

        if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            if current_time - times["initial"] > 1.0:
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.GO_TO_START1)
                self.PlanToStart(context, state)
            return
        if mode == PlannerState.GO_TO_START1:
            traj_q= context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().trajectory
            if context.get_time() > traj_q.end_time():
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.GRASP1)
                self.Plan(context, state)
        if mode == PlannerState.GRASP1:
            traj_X_G = context.get_abstract_state(
                int(self._traj_X_G_index)
            ).get_value()
            if traj_X_G.get_number_of_segments() > 0 and (not traj_X_G.is_time_in_range(context.get_time())):
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.GO_HOME)
                self.GoHome(context, state)
            return
        if mode == PlannerState.GO_HOME:
            traj_q= context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().trajectory
            if context.get_time() > traj_q.end_time():
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.GO_TO_START2)
                self.PlanToStart(context, state)
        if mode == PlannerState.GO_TO_START2:
            traj_q= context.get_abstract_state(
                int(self._current_joint_traj_idx)
            ).get_value().trajectory
            if context.get_time() > traj_q.end_time():
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.GRASP2)
                self.Plan(context, state)
        if mode == PlannerState.GRASP2:
            traj_X_G = context.get_abstract_state(
                int(self._traj_X_G_index)
            ).get_value()
            if traj_X_G.get_number_of_segments() > 0 and (not traj_X_G.is_time_in_range(context.get_time())):
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.DONE)
            return


    def GoHome(self, context, state):
        '''
        Reset to original joint positions to avoid starting on edge of config space
        '''

        state.get_mutable_abstract_state(int(self._mode_index)).set_value(
            PlannerState.GO_HOME
        )
        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        q0 = context.get_discrete_state(self._q0_index).get_value().copy()
        # q0[0] = q[0]  # Safer to not reset the first joint.

        traj = plan_unconstrained_gcs_path_start_to_goal(
            plant=self._iiwa_controller_plant, q_start=q, q_goal=q0
        )
        if traj is None:
            logging.error("Failed to find a path to the home positions.")
            exit(1)

        toppra_traj = reparameterize_with_toppra(
            trajectory=traj,
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

    def PlanToStart(self, context, state):
        # Save start position for grasp trajectory planner
        X_G_init = self.get_input_port(3).Eval(context)[int(self._gripper_body_index)]
        state.get_mutable_abstract_state(int(self._X_G_init_index)).set_value(
            X_G_init
        )

        q = self.get_input_port(self._iiwa_position_index).Eval(context)
        q_goal = solve_global_inverse_kinematics(
            plant=self._iiwa_controller_plant,
            X_G=X_G_init,
            initial_guess=q,
            position_tolerance=0.0,
            orientation_tolerance=0.0,
            gripper_frame_name="body",
        )
        if q_goal is None:
            logging.error(
                "Failed to solve inverse kinematics for the grasping start pose."
            )
            exit(1)

        # Plan trajectory to saved start position to ensure we start grasp traj at consistent position
        traj = plan_unconstrained_gcs_path_start_to_goal(
            plant=self._iiwa_controller_plant, q_start=q, q_goal=q_goal
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


    def Plan(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        X_G_init= context.get_abstract_state(
            int(self._X_G_init_index)
        ).get_value()

        X_G = {
            "initial": X_G_init,
            "end": default_home_pose
        }

        # Get pcd and select grasp
        body_poses = self.get_input_port(3).Eval(context)
        pcd = []
        for i in range(3):
            cloud = self.get_input_port(i).Eval(context)

            # Crop to region of interest.
            pcd.append(cloud.Crop(lower_xyz=[-0.5, -1.0, 0.051], upper_xyz=[0.5, -0.3, 0.25]))
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
        AddMeshcatTriad(self.meshcat, "principal axis", 
                        X_PT=RigidTransform(RotationMatrix(rot_principal_component_to_axes.as_matrix().T),
                        [com[0], com[1], com[2]]))

        if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            # Planning first grasping trajectory
            # self.grasp_node.compute_candidate_grasps(down_sampled_pcd, random_seed=5, align_grasp_axis=secondary_component, split_axis=2, minor_split_axis=0)
            self.grasp_node.compute_candidate_grasps(down_sampled_pcd, random_seed=5, align_grasp_axis=principal_component, split_axis=1, minor_split_axis=0)
        else:
            # Planning second grasping trajectory
            self.grasp_node.compute_candidate_grasps(down_sampled_pcd, random_seed=5, align_grasp_axis=secondary_component, split_axis=2, minor_split_axis=0)
        
        grasps = self.grasp_node.get_best_grasps(candidate_num=1)

        print(grasps)
        
        # get end effector pose from grasp pose
        X_GE = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 0)), [0, 0, -0.09])

        ee_grasps = [X_WG.multiply(X_GE) for X_WG in grasps]

        X_G["pick"] = ee_grasps[0]

        X_G["display_traj"] = default_display_traj
        X_G, times = MakeGripperFrames(X_G, t0=context.get_time())
        print(
            f"Planned {times['postplace'] - times['initial']} second trajectory in mode {mode} at time {context.get_time()}."
        )
        state.get_mutable_abstract_state(int(self._times_index)).set_value(
            times
        )

        if False:  # Useful for debugging
            AddMeshcatTriad(self.meshcat, "X_Oinitial", X_PT=X_O["initial"])
            AddMeshcatTriad(self.meshcat, "X_Gprepick", X_PT=X_G["prepick"])
            AddMeshcatTriad(self.meshcat, "X_Gpick", X_PT=X_G["pick"])
            AddMeshcatTriad(self.meshcat, "X_Gplace", X_PT=X_G["place"])

        traj_X_G = MakeGripperPoseTrajectory(X_G, times)
        traj_wsg_command = MakeGripperCommandTrajectory(times)

        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(
            traj_X_G
        )
        state.get_mutable_abstract_state(int(self._traj_wsg_index)).set_value(
            traj_wsg_command
        )

    def start_time(self, context):
        return (
            context.get_abstract_state(int(self._traj_X_G_index))
            .get_value()
            .start_time()
        )

    def end_time(self, context):
        return (
            context.get_abstract_state(int(self._traj_X_G_index))
            .get_value()
            .end_time()
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
            default_home_pose
        )
    
    def GetCurrentJointPositionTrajectory(self, context, output):
        current_joint_traj = context.get_abstract_state(
            self._current_joint_traj_idx
        ).get_value()
        output.set_value(current_joint_traj)

    def CalcWsgPosition(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
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

        # Command the open position
        output.SetFromVector([opened])

    def CalcControlMode(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        if mode == PlannerState.GO_HOME or mode == PlannerState.GO_TO_START1 or mode == PlannerState.GO_TO_START2:
            output.set_value(InputPortIndex(2))  # Toppra joint traj
        else:
            output.set_value(InputPortIndex(1))  # Diff IK when executing display trajectories

    def CalcDiffIKReset(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        if mode == PlannerState.GO_HOME:
            output.set_value(True)
        else:
            output.set_value(False)

    def Initialize(self, context, discrete_state):
        discrete_state.set_value(
            int(self._q0_index),
            self.get_input_port(int(self._iiwa_position_index)).Eval(context),
        )