# Based off of https://github.com/RussTedrake/manipulation/blob/master/manipulation/pick.py 
# with modifications to display object at location and place back at same location

import numpy as np
from pydrake.all import (
    AngleAxis,
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
)

def MakeGripperFrames(X_G, t0=0):
    """
    Takes a partial specification with X_G["initial"], X_G["pick"],
    X_G["display_traj"] (a list of poses of any length), and X_G["end"],
    and returns a X_G and times with all of the pick and display
    frames populated.
    """
    # put down where it was picked up, return gripper to initial position
    X_G["place"] = X_G["pick"]

    # pregrasp is negative z in the gripper frame
    X_GgraspGpregrasp = RigidTransform([0, 0.0, -0.15])

    X_G["prepick"] = X_G["pick"] @ X_GgraspGpregrasp

    # I'll interpolate a halfway orientation by converting to axis angle and
    # halving the angle.
    X_GinitialGprepick = X_G["initial"].inverse() @ X_G["prepick"]
    angle_axis = X_GinitialGprepick.rotation().ToAngleAxis()
    X_GinitialGprepare = RigidTransform(
        AngleAxis(angle=angle_axis.angle() / 2.0, axis=angle_axis.axis()),
        X_GinitialGprepick.translation() / 2.0,
    )
    X_G["prepare"] = X_G["initial"] @ X_GinitialGprepare
    p_G = np.array(X_G["prepare"].translation())
    p_G[2] = 0.5
    # To avoid hitting the cameras, make sure the point satisfies x - y < .5
    if p_G[0] - p_G[1] < 0.5:
        scale = 0.5 / (p_G[0] - p_G[1])
        p_G[:1] /= scale
    X_G["prepare"].set_translation(p_G)

    X_GprepickGpredisplay = X_G["prepick"].inverse() @ X_G["display_traj"][0]

    # Now let's set the timing
    times = {"initial": t0}
    prepare_time = 10.0 * np.linalg.norm(X_GinitialGprepare.translation())
    times["prepare"] = times["initial"] + prepare_time
    times["prepick"] = times["prepare"] + prepare_time
    # Allow some time for the gripper to close.
    times["pick_start"] = times["prepick"] + 2.0
    times["pick_end"] = times["pick_start"] + 2.0
    X_G["pick_start"] = X_G["pick"]
    X_G["pick_end"] = X_G["pick"]
    times["postpick"] = times["pick_end"] + 2.0
    X_G["postpick"] = RigidTransform(X_G["pick"].rotation(), X_G["pick"].translation() + [0, 0, 0.09])
    time_to_predisplay = 10.0 * np.linalg.norm(
        X_GprepickGpredisplay.translation()
    )
    times["display_traj"] = []
    times["display_traj"].append(times["postpick"] + time_to_predisplay)
    for i in range(len(X_G["display_traj"])-1):
        times["display_traj"].append(times["display_traj"][-1] + 2.0)

    X_G["preplace"] = X_G["postpick"]
    times["preplace"] = times["display_traj"][-1] + time_to_predisplay
    times["place_start"] = times["preplace"] + 2.0
    times["place_end"] = times["place_start"] + 2.0
    X_G["place_start"] = X_G["place"]
    X_G["place_end"] = X_G["place"]
    times["postplace"] = times["place_end"] + 2.0
    X_G["postplace"] = X_G["preplace"]

    X_GpostplaceGend = X_G["postplace"].inverse() @ X_G["end"]
    time_to_end = 10.0 * np.linalg.norm(
        X_GpostplaceGend.translation()
    )
    times["end"] = times["postplace"] + time_to_end

    return X_G, times


def MakeGripperPoseTrajectory(X_G, times):
    """Constructs a gripper position trajectory from the plan "sketch"."""
    sample_times = []
    poses = []
    for name in [
        "initial",
        "prepare",
        "prepick",
        "pick_start",
        "pick_end",
        "postpick",
        "display_traj",
        "preplace",
        "place_start",
        "place_end",
        "postplace",
        "end",
    ]:
        if name == "display_traj":
            for i in range(len(times["display_traj"])):
                sample_times.append(times["display_traj"][i])
                poses.append(X_G["display_traj"][i])
        else:
            sample_times.append(times[name])
            poses.append(X_G[name])

    return PiecewisePose.MakeLinear(sample_times, poses)


def MakeGripperCommandTrajectory(times):
    """Constructs a WSG command trajectory from the plan "sketch"."""
    opened = np.array([0.107])
    closed = np.array([0.0])

    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(
        [times["initial"], times["pick_start"]],
        np.hstack([[opened], [opened]]),
    )
    traj_wsg_command.AppendFirstOrderSegment(times["pick_end"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["place_start"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["place_end"], opened)
    traj_wsg_command.AppendFirstOrderSegment(times["postplace"], opened)
    return traj_wsg_command