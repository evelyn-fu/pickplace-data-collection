import numpy as np
from pydrake.all import (
    PiecewisePolynomial,
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
    times = {"prepick": 0.5}
    
    # Allow some time for the gripper to close.
    X_G["pick_start"] = X_G["pick"]
    X_G["pick_end"] = X_G["pick"]
    times["pick_start"] = 1.0
    times["pick_end"] = 0.0

    # raise object off surface
    X_G["postpick"] = RigidTransform(X_G["pick"].rotation(), X_G["pick"].translation() + [0, 0, 0.2])
    times["postpick"] = 1.0

    # Give time to get to start of display trajectory
    time_to_predisplay = 5.0 * np.linalg.norm(
        X_GprepickGpredisplay.translation()
    )
    # special case where first value is time to first frame in traj, and second is time to consecutive frames
    times["display_traj"] = [time_to_predisplay, 0.1] 

    # Prepare to place back down
    X_G["preplace"] = RigidTransform(X_G["place"].rotation(), X_G["place"].translation() + [0, 0, 0.2])
    times["preplace"] = time_to_predisplay

    # Place back down and allow some time for gripper to open
    X_G["place_start"] = X_G["place"]
    X_G["place_end"] = X_G["place"]
    times["place_start"] = 1.0
    times["place_end"] = 0.0

    # Go back to prepick pose
    if place_flipped:
        X_GgraspGpostgrasp = RigidTransform([0, 0.0, -pregrasp_dist])
        X_G["postplace"] = X_G["place"] @ X_GgraspGpostgrasp
        times["postplace"] = 1.0
        X_G["postpostplace"] = X_G["prepick"]
        times["postpostplace"] = 1.0
    else:
        X_G["postplace"] = X_G["prepick"]
        times["postplace"] = 1.0


    return X_G, times

def MakePickAndDisplayJointPositionsTrajectory(
        X_G, 
        times, 
        plant, 
        q, 
        q_prepick, 
        place_flipped=False, 
        max_display_frames=6,
        joint_limits=None):
    """
    Constructs a gripper position trajectory from the plan "sketch".
    Returns joint configuration for each gripper position in a dictionary joint_configs and a peicewise polynomial
    trajectory for the display along the last joint

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
    q_prev = q
    q_display_center = None
    joint_configs = {}
    for name in [
        #### Traj 1 Start ###
        "prepick",
        "pick_start",
        #### Traj 1 End ###
        #### Traj 2 Start ###
        "pick_end",
        "postpick",
        #### Traj 2 End ###
        # plan with gcs between postpick and display
        #### Traj 3 Start ###
        "display_traj",
        #### Traj 3 End ###
        # plan with gcs between display and preplace
        #### Traj 4 Start ###
        "preplace", 
        "place_start",
        #### Traj 4 End ###
        #### Traj 5 Start ###
        "place_end",
        "postplace",
        #### Traj 5 End ###
    ]:
        if name == "display_traj":
            center_found = False
            for i in range(len(X_G["display_traj"])):
                # find one display pose to act as center
                q_display_center = solve_global_inverse_kinematics(
                    plant=plant,
                    X_G=X_G["display_traj"][i],
                    initial_guess=q_prev,
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
                            initial_guess=q_prev + np.random.normal(0, np.pi/4, 7),
                            position_tolerance=0.0,
                            orientation_tolerance=0.005,
                            gripper_frame_name="iiwa_link_7",
                            joint_limits=joint_limits
                        )
                        attempts += 1

                    if q_next is None:
                        print("IK failed again at q_display_center")
                    else:
                        print("phew.")
                if q_display_center is not None:
                    # construct trajectory of rotating 7th joint
                    q7s = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi * 165.0 / 180.0, 
                               3*np.pi/4, np.pi/2, np.pi/4, 
                               0.0, -np.pi/4, -np.pi/2, -3*np.pi/4, -np.pi * 165.0 / 180.0]
                    for i in range(len(q7s)):
                        q_temp = q_display_center.copy()
                        q_temp[6] = q7s[i]
                        positions.append(q_temp)
                        
                        if i == 0:
                            sample_times.append((sample_times[-1] if len(sample_times) != 0 else 0) + times[name][0])
                        else:
                            sample_times.append((sample_times[-1] if len(sample_times) != 0 else 0) + times[name][1])

                    center_found = True
                    q_prev = q_temp
                    break
            if not center_found:
                raise Exception("Failed to solve global IK for display trajectory center")
            else:
                joint_configs["display_center"] = q_display_center
        else:
            q_next = solve_global_inverse_kinematics(
                plant=plant,
                X_G=X_G[name],
                initial_guess=q_prev,
                position_tolerance=0.0,
                orientation_tolerance=0.0,
                gripper_frame_name="iiwa_link_7",
                joint_limits=joint_limits
            )
            if q_next is None:
                print("IK failed at", name)
                attempts = 0
                while q_next is None and attempts < 10:
                    print("trying global inverse kinematics with new initial guess randomized around q")
                    q_next = solve_global_inverse_kinematics(
                        plant=plant,
                        X_G=X_G[name],
                        initial_guess=q_prev + np.random.normal(0, np.pi/4, 7),
                        position_tolerance=0.0,
                        orientation_tolerance=0.0,
                        gripper_frame_name="iiwa_link_7",
                        joint_limits=joint_limits
                    )
                    attempts += 1

                if q_next is None:
                    print("IK failed again at", name)
                else:
                    print("phew.")
            
            if q_next is not None:
                q_prev = q_next

            joint_configs[name] = q_next

            if name == "prepick":
                joint_configs[name] = q_prepick
            elif name == "pick_end":
                joint_configs[name] = joint_configs["pick_start"]
            elif name == "preplace" and not place_flipped:
                joint_configs[name] = joint_configs["postpick"]
            elif name == "place_start" and not place_flipped:
                joint_configs[name] = joint_configs["pick_start"]
            elif name == "place_end":
                joint_configs[name] = joint_configs["place_start"]
            elif name == "postplace" and not place_flipped:
                joint_configs[name] = joint_configs["prepick"]
    
    t = PiecewisePolynomial.FirstOrderHold(sample_times, np.array(positions).T)
    return t, joint_configs
