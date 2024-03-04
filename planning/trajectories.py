import numpy as np
from pydrake.all import (
    PiecewisePolynomial,
    RigidTransform,
)
from planning.inverse_kinematics import solve_global_inverse_kinematics

def MakePickAndDisplayGripperFrames(X_G):
    """
    Takes a partial specification with X_G["pick"], X_G["prepick"], and
    X_G["display_traj"] (a tuple of two list of poses of any length that begin with the same pose,
    this is meant to display in one direction as far as possible then another as far as possible),
    and returns a X_G and times with all of the pick and display
    frames populated.
    """
    # put down where it was picked up, return gripper to initial position
    X_G["place"] = X_G["pick"]

    X_GprepickGpredisplay = X_G["prepick"].inverse() @ X_G["display_traj"][0][0]

    # Amount of time it takes to GET TO each frame
    times = {"prepick": 0}
    
    # Allow some time for the gripper to close.
    X_G["pick_start"] = X_G["pick"]
    X_G["pick_end"] = X_G["pick"]
    times["pick_start"] = 5.0
    times["pick_end"] = 2.0

    # raise object off surface
    X_G["postpick"] = RigidTransform(X_G["pick"].rotation(), X_G["pick"].translation() + [0, 0, 0.09])
    times["postpick"] = 2.0

    # Give time to get to start of display trajectory
    time_to_predisplay = 10.0 * np.linalg.norm(
        X_GprepickGpredisplay.translation()
    )
    # special case where first value is time to first frame in traj, and second is time to consecutive frames
    times["display_traj"] = [time_to_predisplay, 0.5] 

    # Prepare to place back down
    X_G["preplace"] = X_G["postpick"]
    times["preplace"] = time_to_predisplay

    # Place back down and allow some time for gripper to open
    X_G["place_start"] = X_G["place"]
    X_G["place_end"] = X_G["place"]
    times["place_start"] = 2.0
    times["place_end"] = 2.0

    # Go back to prepick pose
    X_G["postplace"] = X_G["prepick"]
    times["postplace"] = 2.0

    return X_G, times

def MakePickAndDisplayJointPositionsTrajectory(X_G, times, plant, q):
    """
    Constructs a gripper position trajectory from the plan "sketch".
    Returns three piecewise polynomial trajectories. One for before grasp, one for during, one for after.
    This is in order to close the gripper between these two trajectories.
    """
    sample_times1 = []
    positions1 = []
    sample_times2 = []
    positions2 = []
    sample_times3 = []
    positions3 = []
    q_prev = q
    for name in [
        "prepick",
        "pick_start",
        "pick_end",
        "postpick",
        "display_traj",
        "preplace",
        "place_start",
        "place_end",
        "postplace",
    ]:
        if name == "display_traj":
            # display direction 1 till failure
            for i in range(len(X_G["display_traj"][0])):
                q_next = solve_global_inverse_kinematics(
                    plant=plant,
                    X_G=X_G["display_traj"][0][i],
                    initial_guess=q_prev,
                    position_tolerance=0.0,
                    orientation_tolerance=0.0,
                    gripper_frame_name="iiwa_link_7",
                )
                if q_next is not None:
                    q_prev = q_next
                    sample_times2.append(sample_times2[-1] + (times["display_traj"][0] if i == 0 else times["display_traj"][1]))
                    positions2.append(q_next)
                else:
                    if i != 0:
                        last_t = sample_times2[-1]
                        sample_times2 += [last_t + j * times["display_traj"][1] for j in range(1, i-1)]
                        reversed_positions = list(positions2[-i:-1].__reversed__())
                        positions2 += reversed_positions[:-1] # add going backwards, dont add final one since that's assumed to be the first position of the next direction
                        q_prev = reversed_positions[-1] # start initial guess at first position, since it should be the first position of the next direction
                    break

            # display direction 2 till failure
            for i in range(len(X_G["display_traj"][1])):
                q_next = solve_global_inverse_kinematics(
                    plant=plant,
                    X_G=X_G["display_traj"][1][i],
                    initial_guess=q_prev,
                    position_tolerance=0.0,
                    orientation_tolerance=0.0,
                    gripper_frame_name="iiwa_link_7",
                )
                if q_next is not None:
                    q_prev = q_next
                    sample_times2.append(sample_times2[-1] + times["display_traj"][1])
                    positions2.append(q_next)
                else:
                    if i != 0:
                        last_t = sample_times2[-1]
                        sample_times2 += [last_t + j * times["display_traj"][1] for j in range(1, i)]
                        positions2 += list(positions2[-i:-1].__reversed__()) # add going backwards
                    break
            
            q_prev = positions2[-1]
        else:
            if name == "prepick" or name == "pick_start":
                sample_times1.append((sample_times1[-1] if len(sample_times1) != 0 else 0) + times[name])
            elif name == "place_end" or name == "postplace":
                sample_times3.append((sample_times3[-1] if len(sample_times3) != 0 else 0) + times[name])
            else:
                sample_times2.append((sample_times2[-1] if len(sample_times2) != 0 else 0) + times[name])

            q_next = solve_global_inverse_kinematics(
                plant=plant,
                X_G=X_G[name],
                initial_guess=q_prev,
                position_tolerance=0.0,
                orientation_tolerance=0.0,
                gripper_frame_name="iiwa_link_7",
            )
            q_prev = q_next
            if name == "prepick" or name == "pick_start":
                positions1.append(q_next)
            elif name == "place_end" or name == "postplace":
                positions3.append(q_next)
            else:
                positions2.append(q_next)

    sample_times2 = [t - sample_times2[0] for t in sample_times2]
    sample_times3 = [t - sample_times3[0] for t in sample_times3]
    
    t1 = PiecewisePolynomial.FirstOrderHold(sample_times1, np.array(positions1).T)
    t2 = PiecewisePolynomial.FirstOrderHold(sample_times2, np.array(positions2).T)
    t3 = PiecewisePolynomial.FirstOrderHold(sample_times3, np.array(positions3).T)
    return t1, t2, t3
