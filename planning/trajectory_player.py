from pydrake.systems.framework import LeafSystem
from iiwa_setup_dataclasses.trajectories import TrajectoryWithTimingInformation
from pydrake.common.value import (
    AbstractValue
)

class TrajectoryPlayer(LeafSystem):
    def __init__(
            self, 
            trajectories
        ):
        LeafSystem.__init__(self)

        self.trajectories = trajectories
        self._current_traj_index = self.DeclareDiscreteState(1)
        self.num_traj = len(trajectories)
        self._current_joint_traj_idx = self.DeclareAbstractState( # joint positions traj
            AbstractValue.Make(TrajectoryWithTimingInformation())
        )
        # output joint position trajectory
        self.DeclareAbstractOutputPort(
            "joint_position_trajectory",
            lambda: AbstractValue.Make(TrajectoryWithTimingInformation()),
            self.GetCurrentJointPositionTrajectory,
        )
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)
        self.done = False

    def Update(self, context, state):
        traj_q = context.get_abstract_state(
            int(self._current_joint_traj_idx)
        ).get_value().trajectory
        current_traj = context.get_discrete_state(self._current_joint_traj_idx).get_value()

        if self.done:
            return

        if context.get_time() > traj_q.end_time():
            state.set_value(
                int(self._current_traj_index),
                current_traj + 1,
            )
            if current_traj + 1 == self.num_traj:
                self.done == True
            else:
                current_time = context.get_time()
                state.get_mutable_abstract_state(self._current_joint_traj_idx).set_value(
                    TrajectoryWithTimingInformation(
                        trajectory=self.trajectories[current_traj],
                        start_time_s=current_time,
                    )
                )


    def GetCurrentJointPositionTrajectory(self, context, output):
        current_joint_traj = context.get_abstract_state(
            self._current_joint_traj_idx
        ).get_value()
        output.set_value(current_joint_traj)

    def Initialize(self, context, discrete_state):
        discrete_state.set_value(
            int(self._current_traj_index),
            0,
        )