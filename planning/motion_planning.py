import numpy as np
from pydrake.all import (
    Trajectory
)
import sys
from planning.utils.csdecomp_path import CSDECOMP_PATH
sys.path.append(f'{CSDECOMP_PATH}/bazel-bin/csdecomp/src/pybind/pycsdecomp')
import pycsdecomp as csd

from pydrake.all import Trajectory, PiecewisePolynomial

def make_pl_traj(path, speed=0.1, startup_time=0.5, start_time=0.0):
    cleaned_path = [path[0,:]]
    t_breaks = [startup_time + start_time]
    movement_between_segment = np.linalg.norm(path[1:,:] - path[:-1,:], axis=1, ord=np.inf)
    for s, next_config in zip(movement_between_segment/speed, path[1:]):
        if s + t_breaks[-1] > t_breaks[-1] + 2.220446049250313e-16:
            t_breaks += [s + t_breaks[-1]]
            cleaned_path.append(next_config)
    if startup_time > 0:
        t_breaks = [start_time] + t_breaks + [t_breaks[-1] + startup_time]
        cleaned_path = [cleaned_path[0]] + cleaned_path + [cleaned_path[-1]]
    
    return PiecewisePolynomial.FirstOrderHold(t_breaks, np.array(cleaned_path).T)

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
    
    return make_pl_traj(np.array(pwl_plan))

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