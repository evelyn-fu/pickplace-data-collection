import numpy as np
from pydrake.all import (
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw
)
from planning.inverse_kinematics import solve_global_inverse_kinematics

def MakePickAndDisplayGripperFrames(X_G, gripper_length, pregrasp_dist, place_flipped=False):
    """
    Takes a partial specification with X_G["pick"], X_G["prepick"], and
    X_G["display_traj"] (a tuple of two list of poses of any length that begin with the same pose,
    this is meant to display in one direction as far as possible then another as far as possible),
    and returns a X_G and times with all of the pick and display
    frames populated.
    """
    # put down where it was picked up, rotated 180 to show other side
    rot_180 = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, np.pi)))
    X_G["place"] = X_G["pick"]
    if place_flipped:
        R = X_G["pick"].GetAsMatrix4()[:3, :3]
        t = X_G["pick"].GetAsMatrix4()[:3, 3]
        X_G["place"] = rot_180 @ RigidTransform(RotationMatrix(R))

        # calculate translation difference of bottom of gripper given rotation (t is top of gripper, want to place object back in same place)
        t_gripper_angle = X_G["place"] @ [0, 0, gripper_length] # 12 cm is roughly the length of the gripper?
        t_gripper_angle[2] = 0
        t_gripper_angle *= 2
        print(t_gripper_angle)
        X_G["place"].set_translation(t - t_gripper_angle)
        print("X_pick", X_G["pick"])
        print("X_place", X_G["place"])

    X_GprepickGpredisplay = X_G["prepick"].inverse() @ X_G["display_traj"][0]

    # Amount of time it takes to GET TO each frame
    times = {"prepick": 0.0}
    
    # Allow some time for the gripper to close.
    X_G["pick_start"] = X_G["pick"]
    X_G["pick_end"] = X_G["pick"]
    times["pick_start"] = 2.0
    times["pick_end"] = 1.0

    # raise object off surface
    X_G["postpick"] = RigidTransform(X_G["pick"].rotation(), X_G["pick"].translation() + [0, 0, 0.2])
    times["postpick"] = 2.0

    # time to consecutive frames
    times["display_traj"] = 0.1

    # Prepare to place back down
    if place_flipped:
        X_G["preplace"] = RigidTransform(X_G["place"].rotation(), X_G["place"].translation() + [0, 0, 0.2])
    else:
        X_G["preplace"] = X_G["postpick"]
    times["preplace"] = 0.0

    # Place back down and allow some time for gripper to open
    X_G["place_start"] = X_G["place"]
    X_G["place_end"] = X_G["place"]
    times["place_start"] = 2.0
    times["place_end"] = 1.0

    # Go back to prepick pose
    if place_flipped:
        X_GgraspGpostgrasp = RigidTransform([0, 0.0, -pregrasp_dist])
        X_G["postplace"] = X_G["place"] @ X_GgraspGpostgrasp
        times["postplace"] = 2.0
        X_G["postpostplace"] = X_G["prepick"]
        times["postpostplace"] = 2.0
    else:
        X_G["postplace"] = X_G["prepick"]
        times["postplace"] = 2.0

    return X_G, times

def MakeGripperPoseTrajectory(X_G, times, predisplay=True, t0=0.0):
    """Constructs a gripper position trajectory from the plan "sketch"."""
    if predisplay:
        names = [
        "prepick",
        "pick_start",
        "pick_end",
        "postpick",
        ]
    else:
        names = [
        "preplace", 
        "place_start",
        "place_end",
        "postplace",
        ]

    sample_times = []
    poses = []
    for name in names:
        if len(sample_times) == 0:
            sample_times.append(times[name] + t0)
        else:
            sample_times.append(sample_times[-1] + times[name])
        poses.append(X_G[name])

    return PiecewisePose.MakeLinear(sample_times, poses)

def MakeGripperCommandTrajectory(times, predisplay=True, t0=0.0):
    """Constructs a WSG command trajectory from the plan "sketch"."""
    opened = np.array([0.107])
    closed = np.array([0.0])

    if predisplay:
        names = [
        "prepick",
        "pick_start",
        "pick_end",
        "postpick",
        ]
    else:
        names = [
        "preplace", 
        "place_start",
        "place_end",
        "postplace",
        ]

    sample_times = []
    positions = []
    for name in names:
        if len(sample_times) == 0:
            sample_times.append(times[name] + t0)
        else:
            sample_times.append(sample_times[-1] + times[name])
        
        if name == "prepick" or name == "pick_start" or name == "place_end" or name == "postplace":
            positions.append(opened)
        else:
            positions.append(closed)

    t = PiecewisePolynomial.FirstOrderHold(sample_times, np.array(positions).T)
    return t

def MakeDisplayJointPositionsTrajectory(
        X_G, 
        times, 
        plant, 
        q,
        joint_limits=None):
    """
    Returns a peicewise polynomial trajectory for the display along the last joint

    X_G: map of gripper poses for each frame, with X_G["display_traj"] being a list of possible poses
        to center the display trajectory at
    times: map of time to get to each frame
    plant: plant with iiwa to solve for global IK
    q: iiwa starting position
    max_dislay_frames: maximum number of frames from X_G["display_traj"] to add to trajectory, prevents wasting time
        displaying already seen angles
    """
    sample_times = []
    positions = []
    q_display_center = None

    center_found = False
    for i in range(len(X_G["display_traj"])):
        # find one display pose to act as center
        q_display_center = solve_global_inverse_kinematics(
            plant=plant,
            X_G=X_G["display_traj"][i],
            initial_guess=q,
            position_tolerance=0.0,
            orientation_tolerance=0.005,
            gripper_frame_name="iiwa_link_7",
            joint_limits=joint_limits
        )
        if q_display_center is None:
            print("IK failed at q_display_center")
            attempts = 0
            while q_display_center is None and attempts < 10:
                print("trying global inverse kinematics with new initial guess randomized around q")
                q_display_center = solve_global_inverse_kinematics(
                    plant=plant,
                    X_G=X_G["display_traj"][i],
                    initial_guess=q + np.random.normal(0, np.pi/4, 7),
                    position_tolerance=0.0,
                    orientation_tolerance=0.005,
                    gripper_frame_name="iiwa_link_7",
                    joint_limits=joint_limits
                )
                attempts += 1

            if q_display_center is None:
                print("IK failed again at q_display_center")
            else:
                print("phew.")
        if q_display_center is not None:
            # construct trajectory of rotating 7th joint
            q7s = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi * 165.0 / 180.0, 
                        3*np.pi/4, np.pi/2, np.pi/4, 
                        0.0, -np.pi/4, -np.pi/2, -3*np.pi/4, -np.pi * 165.0 / 180.0,
                        -np.pi/2, 0.0]
            for j in range(len(q7s)):
                q_temp = q_display_center.copy()
                q_temp[6] = q7s[j]
                positions.append(q_temp)
                
                sample_times.append((sample_times[-1] if len(sample_times) != 0 else 0) + times["display_traj"])

            center_found = True
            display_center = X_G["display_traj"][i]
            break

    if not center_found:
        raise Exception("Failed to solve global IK for display trajectory center")
    
    q_display_center[6] = 0.0
    t = PiecewisePolynomial.FirstOrderHold(sample_times, np.array(positions).T)
    return t, q_display_center
