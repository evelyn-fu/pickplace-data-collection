# Taken and modified from https://github.com/RussTedrake/manipulation/blob/master/manipulation/systems.py
import numpy as np
from pydrake.all import (
    AbstractValue,
    Body,
    DiagramBuilder,
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters,
    Frame,
    LeafSystem,
    MultibodyPlant,
    RigidTransform,
)

def AddIiwaDifferentialIK(
    builder: DiagramBuilder, 
    plant: MultibodyPlant, 
    frame: Frame | None = None, 
    velocity_lims=None, 
    acceleration_lims=None,
    joint_centering_gain=10.0
) -> DifferentialInverseKinematicsIntegrator:
    """Adds a DifferentialInverseKinematicsIntegrator system to the builder with default parameters suitable for use with the standard 7-link iiwa models or the 3-link planar iiwa models.

    Args:
        builder: The DiagramBuilder to which the system should be added.

        plant: The MultibodyPlant passed to the DifferentialInverseKinematicsIntegrator.

        frame: The frame to use for the end effector command. Defaults to the body
            frame of "iiwa_link_7".

    Returns:
        The DifferentialInverseKinematicsIntegrator system.
    """
    params = DifferentialInverseKinematicsParameters(
        plant.num_positions(), plant.num_velocities()
    )
    time_step = plant.time_step()
    q0 = plant.GetPositions(plant.CreateDefaultContext())
    params.set_nominal_joint_position(q0)
    if velocity_lims is not None:
        params.set_joint_velocity_limits((-velocity_lims, velocity_lims))
    else:
        params.set_end_effector_angular_speed_limit(2)
        params.set_end_effector_translational_velocity_limits([-2, -2, -2], [2, 2, 2])
    # Don't know why adding acceleration limits prevents the robot from moving lol
    # if acceleration_lims is not None:
    #     params.set_joint_acceleration_limits((-acceleration_lims, acceleration_lims))

    if frame is None:
        frame = plant.GetFrameByName("iiwa_link_7")
    if plant.num_positions() == 3:  # planar iiwa
        iiwa14_velocity_limits = np.array([1.4, 1.3, 2.3])
        params.set_joint_velocity_limits(
            (-iiwa14_velocity_limits, iiwa14_velocity_limits)
        )
        # These constants are in body frame
        assert (
            frame.name() == "iiwa_link_7"
        ), "Still need to generalize the remaining planar diff IK params for different frames"  # noqa
        params.set_end_effector_velocity_flag([True, False, False, True, False, True])
    else:
        if velocity_lims is not None:
            iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
            params.set_joint_velocity_limits(
                (-iiwa14_velocity_limits, iiwa14_velocity_limits)
            )
        params.set_joint_centering_gain(joint_centering_gain * np.eye(7))
    differential_ik = builder.AddSystem(
        DifferentialInverseKinematicsIntegrator(
            plant,
            frame,
            time_step,
            params,
            log_only_when_result_state_changes=True,
        )
    )
    return differential_ik