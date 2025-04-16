import numpy as np
from pydrake.all import (
    Trajectory
)
import pycsdecomp as csd
from planning.toppra import make_pl_traj

def plan_drm(drm_planner: csd.DrmPlanner, start: np.ndarray, goal: np.ndarray, vox: np.ndarray, online_voxel_radius: float) -> Trajectory:
    """
    Plan a path from start to goalusing drm

    Args:
        drm_planner (csd.DrmPlanner): The DRM planner instance.
        start (np.ndarray): The starting point of the path.
        goal (np.ndarray): The goal point of the path.
        vox (np.ndarray): The voxel observation of the environment.
        online_voxel_radius (float): The radius of the voxel observation.

    Returns:
        Trajectory: pydrake Trajectory object representing the planned path.
    """
    if vox is None:
        online_voxel_observation = csd.Voxels()
    else:
        online_voxel_observation = csd.Voxels(vox)

    success, pwl_plan = drm_planner.Plan(start,
                     goal,
                     online_voxel_observation,
                     online_voxel_radius)
    
    return make_pl_traj(pwl_plan)

def plan_path_custom(start: np.ndarray, goal: np.ndarray, **kwargs) -> Trajectory:
    """
    Plan a path from start to goal.

    Args:
        start (np.ndarray): The starting point of the path.
        goal (np.ndarray): The goal point of the path.

    Returns:
        Trajectory: pydrake Trajectory object representing the planned path.
    """
    raise NotImplementedError("Path planning not implemented yet.")