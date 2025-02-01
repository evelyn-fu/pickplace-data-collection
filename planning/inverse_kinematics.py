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

from mmt_gcs.planning.corridor_planning_utils import CollisionCheckerBase
from planning.analytic_ik_iiwa_7 import Analytic_IK_7DoF, iiwa_limits_lower, iiwa_limits_upper, iiwa_alpha, iiwa_d

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

def sample_ik_params():
    #returns three discrete {-1,1} values and one continuous random value [0, 2*pi]
    return int(2*(np.random.randint(2)-0.5)), int(2*(np.random.randint(2)-0.5)), int(2*(np.random.randint(2)-0.5)), np.random.rand()*2*np.pi 

def solve_via_analytic_IK(pose: RigidTransform,
                          current_config : np.ndarray,
                          ik_domain: HPolyhedron,
                          checker: CollisionCheckerBase):
    
    analytic_ik = Analytic_IK_7DoF(iiwa_alpha, iiwa_d, iiwa_limits_lower, iiwa_limits_upper)
    N_configs = 150
    configs = []
    for _ in range(N_configs):
        GC2, GC4, GC6, psi = sample_ik_params()
        configs.append(analytic_ik.IK(pose.GetAsMatrix4(), [GC2, GC4, GC6], psi))
    configs =np.array(configs).T
    res = checker.CheckConfigsCollisionFree(configs)
    idx_col_free =np.where(res)[0]
    if len(idx_col_free)==0:
        return None
    print(f"[ANALYTIC IK] NUMBER OF CONFIGS {len(idx_col_free)}")
    candidates = configs[:, idx_col_free].reshape(7, len(idx_col_free))
    candidates = np.array([c for c in candidates.T if ik_domain.PointInSet(c)]).T
    print(f"[ANALYTIC IK] NUMBER OF CONFIGS in IK domain {len(candidates.T)}")
    if len(candidates)==0:
        return None        
    dists = np.linalg.norm(candidates - current_config.reshape(7, 1), axis=0)\
    + 0.5*np.linalg.norm(candidates, axis=0)
    print(f" dists : {len(dists)} {dists.shape}")
    return candidates[:, np.argmin(dists)]