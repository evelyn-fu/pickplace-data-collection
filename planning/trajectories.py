# Based off of https://github.com/RussTedrake/manipulation/blob/master/manipulation/pick.py 
# with modifications to start and end at pregrasp pose, display object at location,
# and place back at same location

import numpy as np
from pydrake.all import (
    AngleAxis,
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
)

def MakePickAndDisplayGripperFrames(X_G, t0=0):
    """
    Takes a partial specification with X_G["pick"], X_G["prepick"], and
    X_G["display_traj"] (a list of poses of any length),
    and returns a X_G and times with all of the pick and display
    frames populated.
    """
    # put down where it was picked up, return gripper to initial position
    X_G["place"] = X_G["pick"]

    X_GprepickGpredisplay = X_G["prepick"].inverse() @ X_G["display_traj"][0]

    # Now let's set the timing
    times = {"prepick": t0}
    
    # Allow some time for the gripper to close.
    X_G["pick_start"] = X_G["pick"]
    X_G["pick_end"] = X_G["pick"]
    times["pick_start"] = times["prepick"] + 2.0
    times["pick_end"] = times["pick_start"] + 2.0

    # raise object off surface
    X_G["postpick"] = RigidTransform(X_G["pick"].rotation(), X_G["pick"].translation() + [0, 0, 0.09])
    times["postpick"] = times["pick_end"] + 2.0

    # Give time to get to start of display trajectory
    time_to_predisplay = 10.0 * np.linalg.norm(
        X_GprepickGpredisplay.translation()
    )
    times["display_traj"] = []
    times["display_traj"].append(times["postpick"] + time_to_predisplay)
    for i in range(len(X_G["display_traj"])-1):
        times["display_traj"].append(times["display_traj"][-1] + 2.0)

    # Prepare to place back down
    X_G["preplace"] = X_G["postpick"]
    times["preplace"] = times["display_traj"][-1] + time_to_predisplay

    # Place back down and allow some time for gripper to open
    X_G["place_start"] = X_G["place"]
    X_G["place_end"] = X_G["place"]
    times["place_start"] = times["preplace"] + 2.0
    times["place_end"] = times["place_start"] + 2.0

    # Go back to prepick pose
    X_G["postplace"] = X_G["prepick"]
    times["postplace"] = times["place_end"] + 2.0

    return X_G, times


def MakePickAndDisplayGripperPoseTrajectory(X_G, times):
    """Constructs a gripper position trajectory from the plan "sketch"."""
    sample_times = []
    poses = []
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
            for i in range(len(times["display_traj"])):
                sample_times.append(times["display_traj"][i])
                poses.append(X_G["display_traj"][i])
        else:
            sample_times.append(times[name])
            poses.append(X_G[name])

    return PiecewisePose.MakeLinear(sample_times, poses)


def MakePickAndDisplayGripperCommandTrajectory(times):
    """Constructs a WSG command trajectory from the plan "sketch"."""
    opened = np.array([0.107])
    closed = np.array([0.0])

    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(
        [times["prepick"], times["pick_start"]],
        np.hstack([[opened], [opened]]),
    )
    traj_wsg_command.AppendFirstOrderSegment(times["pick_end"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["place_start"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["place_end"], opened)
    traj_wsg_command.AppendFirstOrderSegment(times["postplace"], opened)
    return traj_wsg_command