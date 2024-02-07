import numpy as np

from pydrake.all import MultibodyPlant, PathParameterizedTrajectory, Toppra, Trajectory, PiecewisePolynomial

def make_pl_traj(path, speed=0.1, startup_time=0.5):
    cleaned_path = [path[0,:]]
    t_breaks = [startup_time]
    movement_between_segment = np.linalg.norm(path[1:,:] - path[:-1,:], axis=1, ord=np.inf)
    for s, next_config in zip(movement_between_segment/speed, path[1:]):
        if s + t_breaks[-1] > t_breaks[-1]:
            t_breaks += [s + t_breaks[-1]]
            cleaned_path.append(next_config)
    if startup_time > 0:
        t_breaks = [0] + t_breaks + [t_breaks[-1] + startup_time]
        cleaned_path = [cleaned_path[0]] + cleaned_path + [cleaned_path[-1]]
    
    return PiecewisePolynomial.FirstOrderHold(t_breaks, np.array(cleaned_path).T)

def reparameterize_with_toppra(
    trajectory: Trajectory,
    plant: MultibodyPlant,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    num_grid_points: int = 1000,
) -> PathParameterizedTrajectory:
    """Reparameterize a trajectory/ path with Toppra.

    Args:
        trajectory (Trajectory): The trajectory on which the TOPPRA problem will be
        solved.
        plant (MultibodyPlant): The robot that will follow the solved trajectory. Used
        for enforcing torque and frame specific constraints.
        velocity_limits (np.ndarray): The velocity limits of shape (N,) where N is the
        number of robot joint joints.
        acceleration_limits (np.ndarray): The acceleration limits of shape (N,) where N
        is the number of robot joint joints.
        num_grid_points (int, optional): The number of uniform points along the path to
        discretize the problem and enforce constraints at.

    Returns:
        PathParameterizedTrajectory: The reparameterized trajectory.
    """
    pl_trajectory = make_pl_traj(trajectory)
    toppra = Toppra(
        path=pl_trajectory,
        plant=plant,
        gridpoints=np.linspace(
            pl_trajectory.start_time(), pl_trajectory.end_time(), num_grid_points
        ),
    )
    toppra.AddJointVelocityLimit(-velocity_limits, velocity_limits)
    toppra.AddJointAccelerationLimit(-acceleration_limits, acceleration_limits)
    time_trajectory = toppra.SolvePathParameterization()
    return PathParameterizedTrajectory(pl_trajectory, time_trajectory)
