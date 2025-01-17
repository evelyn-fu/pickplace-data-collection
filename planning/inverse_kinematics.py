import logging

from typing import Optional, Union

import numpy as np

from pydrake.all import (
    InverseKinematics,
    MultibodyPlant,
    RigidTransform,
    RotationMatrix,
    Solve,
    HPolyhedron
)


def solve_global_inverse_kinematics(
    plant: MultibodyPlant,
    X_G: RigidTransform,
    initial_guess: np.ndarray,
    position_tolerance: float,
    orientation_tolerance: float,
    gripper_frame_name: str = "body",
    joint_limits = None
) -> Optional[np.ndarray]:
    """Computes global IK.

    Args:
        plant (MultibodyPlant): The robot control plant.
        X_G (RigidTransform): Gripper pose to compute the joint angles for.
        initial_guess (np.ndarray): The initial guess to use of shape (N,) where N are
        the number of joint positions.
        position_tolerance (float): The position tolerance to use for the global IK
        optimization problem.
        orientation_tolerance (float): The orientation tolerance to use for the global
        IK optimization problem.
        gripper_frame_name (str): The name of the gripper frame.

    Returns:
        Optional[np.ndarray]: Joint positions corresponding to X_G. Returns None if no
        IK solution could be found.
    """
    ik = InverseKinematics(plant)
    q_variables = ik.q()

    gripper_frame = plant.GetFrameByName(gripper_frame_name)

    # Position constraint
    p_G_ref = X_G.translation()
    ik.AddPositionConstraint(
        frameB=gripper_frame,
        p_BQ=np.zeros(3),
        frameA=plant.world_frame(),
        p_AQ_lower=p_G_ref - position_tolerance,
        p_AQ_upper=p_G_ref + position_tolerance,
    )

    # Orientation constraint
    R_G_ref = X_G.rotation()
    ik.AddOrientationConstraint(
        frameAbar=plant.world_frame(),
        R_AbarA=R_G_ref,
        frameBbar=gripper_frame,
        R_BbarB=RotationMatrix(),
        theta_bound=orientation_tolerance,
    )

    prog = ik.prog()
    prog.SetInitialGuess(q_variables, initial_guess)
    if joint_limits is not None: 
        # Set the joint limits to be a little less than the actual joint limits
        # to prevent q from being at the edge of cspace
        prog.AddBoundingBoxConstraint(
            joint_limits[:,0] + 0.01*np.ones(7), 
            joint_limits[:,1] - 0.01*np.ones(7), 
            q_variables
        )
    prog.AddQuadraticErrorCost(np.identity(len(q_variables)), initial_guess, q_variables)

    result = Solve(prog)
    if not result.is_success():
        logging.error(f"Failed to solve global IK for gripper pose {X_G}.")
        return None
    q_sol = result.GetSolution(q_variables)
    return q_sol

def solve_ik_problem_pete(pose, 
                     plant,
                     plant_context, 
                     q0,
                     domain:Union[HPolyhedron, None] = None,
                     track_orientation = True,
                     collision_free = True,
                     tol = 0.005):

    ik = InverseKinematics(plant, plant_context)
    prog = ik.get_mutable_prog()
    q = ik.q()  
    if domain is not None:
        domain_restrict = HPolyhedron(domain.A(), domain.b()-0.01)
        domain_restrict.AddPointInSetConstraints(prog, q)

    ik.AddPositionConstraint(plant.GetFrameByName('iiwa_link_7'), 
                         np.zeros(3),
                         plant.world_frame(),
                         pose.translation()-tol,
                         pose.translation()+tol,
                         )
    
    if track_orientation:
            ik.AddOrientationConstraint(
                plant.GetFrameByName('iiwa_link_7'),
                RotationMatrix(),
                plant.world_frame(),
                pose.rotation(),
                tol,
            )
    if collision_free:
        ik.AddMinimumDistanceLowerBoundConstraint(0.015, 0.1)
        #ik.AddMinimumDistanceConstraint(0.01, 0.1)
    prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
    prog.SetInitialGuess(q, q0)
    result = Solve(ik.prog())
    if result.is_success():
            return result.GetSolution(q)
    else:
        return None