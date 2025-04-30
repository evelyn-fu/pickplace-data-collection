import logging

from typing import Optional, Union

import numpy as np

from pydrake.all import (
    InverseKinematics,
    MultibodyPlant,
    RigidTransform,
    RotationMatrix,
    Solve,
    HPolyhedron,
    RollPitchYaw
)

import sys
from planning.utils.csdecomp_path import CSDECOMP_PATH
sys.path.append(f'{CSDECOMP_PATH}/bazel-bin/csdecomp/src/pybind/pycsdecomp')
import pycsdecomp as csd
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
                          csd_plant: csd.Plant):
    
    analytic_ik = Analytic_IK_7DoF(iiwa_alpha, iiwa_d, iiwa_limits_lower, iiwa_limits_upper)
    N_configs = 2000
    configs = []
    for _ in range(N_configs):
        GC2, GC4, GC6, psi = sample_ik_params()
        configs.append(analytic_ik.IK(pose.GetAsMatrix4(), [GC2, GC4, GC6], psi))
    configs =np.array(configs).T
    res = csd.CheckCollisionFreeCuda(configs, csd_plant.getMinimalPlant())
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

def solve_via_analytic_IK_with_retries(pose: RigidTransform,
                          current_config : np.ndarray,
                          ik_domain: HPolyhedron,
                          csd_plant: csd.Plant,
                          retries: int = 1):
    result = solve_via_analytic_IK(pose, current_config, ik_domain, csd_plant)
    if result is not None:
        return result
    
    def add_noise_to_transform(transform, max_rotation_deg=1.0, max_translation=0.005):
        # Convert max rotation degrees to radians
        max_rotation_rad = np.radians(max_rotation_deg)

        # Add small random noise to the rotation (in radians)
        noise_rotation = np.random.uniform(-max_rotation_rad, max_rotation_rad, 3)  # Roll, Pitch, Yaw noise
        noise_rpy = RotationMatrix(RollPitchYaw(*noise_rotation))

        # Add small random noise to the translation (in meters)
        noise_translation = np.random.uniform(-max_translation, max_translation, 3)  # x, y, z noise

        # Create noisy transform
        noisy_transform = RigidTransform(transform.rotation() @ noise_rpy,
                                        transform.translation() + noise_translation)

        return noisy_transform
    
    print("retrying with nudged end pose")
    for i in range(retries):
        nudged_pose = add_noise_to_transform(pose)
        result = solve_via_analytic_IK(nudged_pose, current_config, ik_domain, csd_plant)
        if result is not None:
            return result
        
    return None