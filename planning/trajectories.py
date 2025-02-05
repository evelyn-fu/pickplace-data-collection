import numpy as np
from pydrake.all import (
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw
)
from planning.inverse_kinematics import solve_global_inverse_kinematics, solve_via_analytic_IK

def MakePickAndDisplayGripperFrames(X_G, gripper_length, pregrasp_dist, place_flipped=False, X_GE=None):
    """
    Takes a partial specification with X_G["pick"], X_G["prepick"], and
    X_G["display_traj"] (a tuple of two list of poses of any length that begin with the same pose,
    this is meant to display in one direction as far as possible then another as far as possible),
    and returns a X_G and times with all of the pick and display
    frames populated.
    """
    # put down where it was picked up, if place_flipped is true, rotate 180 to show other side
    X_G["place"] = X_G["pick"]
    if place_flipped:
        t_gripper_center = X_G["pick_gripper_frame"] @ [0, 0, gripper_length/2]
        
        # Calculate rotation matrix for 180 degrees around z-axis
        R_z180 = np.array([
            [-1, 0, 0],
            [0, -1, 0], 
            [0, 0, 1]
        ])
        
        # Get current rotation and translation
        R_current = X_G["pick_gripper_frame"].rotation().matrix()
        t_current = X_G["pick_gripper_frame"].translation()
        
        # Apply rotation around z-axis centered at t_gripper_center
        t_centered = t_current - t_gripper_center
        t_rotated = R_z180 @ t_centered
        t_final = t_rotated + t_gripper_center
        
        # Create new rotation matrix combining current rotation with z-axis rotation
        R_final = R_z180 @ R_current
        
        # Set X_G["place"] to the rotated pose
        X_G["place_gripper_frame"] = RigidTransform(RotationMatrix(R_final), t_final)
        X_G["place"] = X_G["place_gripper_frame"].multiply(X_GE)

        print("X_pick", X_G["pick"])
        print("X_place", X_G["place"])

    X_GprepickGpredisplay = X_G["prepick"].inverse() @ X_G["display_traj"][0]

    # Amount of time it takes to GET TO each frame
    times = {"prepick": 0.0}
    
    # Allow some time for the gripper to close.
    X_G["pick_start"] = X_G["pick"]
    X_G["pick_end"] = X_G["pick"]
    times["pick_start"] = 4.0
    times["pick_end"] = 2.0

    # raise object off surface
    X_G["postpick"] = RigidTransform(X_G["pick"].rotation(), X_G["pick"].translation() + [0, 0, 0.2])
    times["postpick"] = 6.0

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
    if place_flipped:
        times["place_start"] = 10.0
    else:
        times["place_start"] = 6.0
    times["place_end"] = 2.0

    # Go back to prepick pose
    if place_flipped:
        X_GgraspGpostgrasp = RigidTransform([0, 0.0, -pregrasp_dist])
        X_G["postplace"] = X_G["place"] @ X_GgraspGpostgrasp
        times["postplace"] = 4.0
    else:
        X_G["postplace"] = X_G["prepick"]
        times["postplace"] = 4.0

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
        q,
        ik_domain,
        cci_checker):
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
        q_display_center = solve_via_analytic_IK(
            pose=X_G["display_traj"][i],
            current_config=q,
            ik_domain=ik_domain,
            checker=cci_checker
        )
        # q_display_center = solve_global_inverse_kinematics(
        #     plant=plant,
        #     X_G=X_G["display_traj"][i],
        #     initial_guess=q,
        #     position_tolerance=0.0,
        #     orientation_tolerance=0.005,
        #     gripper_frame_name="iiwa_link_7",
        #     joint_limits=joint_limits
        # )
        # if q_display_center is None:
        #     print("IK failed at q_display_center")
        #     attempts = 0
        #     while q_display_center is None and attempts < 10:
        #         print("trying global inverse kinematics with new initial guess randomized around q")
        #         q_display_center = solve_global_inverse_kinematics(
        #             plant=plant,
        #             X_G=X_G["display_traj"][i],
        #             initial_guess=q + np.random.normal(0, np.pi/4, 7),
        #             position_tolerance=0.0,
        #             orientation_tolerance=0.005,
        #             gripper_frame_name="iiwa_link_7",
        #             joint_limits=joint_limits
        #         )
        #         attempts += 1

        #     if q_display_center is None:
        #         print("IK failed again at q_display_center")
        #     else:
        #         print("phew.")
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
            break

    if not center_found:
        raise Exception("Failed to solve global IK for display trajectory center")
    
    q_display_center[6] = 0.0
    t = PiecewisePolynomial.FirstOrderHold(sample_times, np.array(positions).T)
    return t, q_display_center

def MakePushingJointPositionsTrajectory(
        X_G_push, 
        X_G_reset, 
        q, 
        ik_domain,
        cci_checker
    ):
    sample_times = [0.0]
    positions = [q]
    q_prev = q
    for i in range(len(X_G_push)):
        X_G = X_G_push[i]
        q_goal = solve_via_analytic_IK(
            pose=X_G,
            current_config=q,
            ik_domain=ik_domain,
            checker=cci_checker
        )
        # q_goal = solve_global_inverse_kinematics(
        #     plant=plant,
        #     X_G=X_G,
        #     initial_guess=q_prev,
        #     position_tolerance=0.005,
        #     orientation_tolerance=0.01,
        #     gripper_frame_name="iiwa_link_7",
        #     joint_limits=joint_limits
        # )

        # attempts = 0
        # while q_goal is None and attempts < 5:
        #     print("trying global inverse kinematics with new initial guess randomized around q")
        #     q_goal = solve_global_inverse_kinematics(
        #         plant=plant,
        #         X_G=X_G,
        #         initial_guess=q + np.random.normal(0, np.pi/4, 7),
        #         position_tolerance=0.005,
        #         orientation_tolerance=0.01,
        #         gripper_frame_name="iiwa_link_7",
        #         joint_limits=joint_limits
        #     )
        #     attempts += 1
        # if q_goal is None:
        #     print("Cannot solve IK for", X_G, f"index {i} of push traj")
        q_prev = q_goal
        positions.append(q_goal)

        if i == 0:
            sample_times.append(0.01 + sample_times[-1])
        else:
            sample_times.append(0.04 + sample_times[-1])

    for i in range(len(X_G_reset)):
        X_G = X_G_reset[i]
        q_goal = solve_via_analytic_IK(
            pose=X_G,
            current_config=q_prev,
            ik_domain=ik_domain,
            checker=cci_checker
        )
        # q_goal = solve_global_inverse_kinematics(
        #     plant=plant,
        #     X_G=X_G,
        #     initial_guess=q_prev,
        #     position_tolerance=0.005,
        #     orientation_tolerance=0.01,
        #     gripper_frame_name="iiwa_link_7",
        #     joint_limits=joint_limits
        # )
        # while q_goal is None and attempts < 5:
        #     print("trying global inverse kinematics with new initial guess randomized around q")
        #     q_goal = solve_global_inverse_kinematics(
        #         plant=plant,
        #         X_G=X_G,
        #         initial_guess=q + np.random.normal(0, np.pi/4, 7),
        #         position_tolerance=0.005,
        #         orientation_tolerance=0.01,
        #         gripper_frame_name="iiwa_link_7",
        #         joint_limits=joint_limits
        #     )
        #     attempts += 1
        if q_goal is None:
            print("Cannot solve IK for", X_G, f"index {i} of reset traj")
        q_prev = q_goal
        positions.append(q_goal)

        if i == 0:
            sample_times.append(0.01 + sample_times[-1])
        else:
            sample_times.append(0.001 + sample_times[-1])
    
    positions += positions[1:] * 7
    cycle_time = sample_times[-1]
    for i in range(1, 8):
        timings = np.array([0.01] + [0.01 + 0.04*j for j in range(1, 8)] + [0.01 + 0.29] + [0.3 + 0.001*j for j in range(1,8)]) + cycle_time * i
        sample_times += timings.tolist()

    t = PiecewisePolynomial.FirstOrderHold(sample_times, np.array(positions).T)
    return t