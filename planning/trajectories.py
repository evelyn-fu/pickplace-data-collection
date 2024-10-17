import numpy as np
from pydrake.all import (
    PiecewisePolynomial,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw
)
from planning.inverse_kinematics import solve_global_inverse_kinematics, forward_kinematics

def MakePickAndDisplayGripperFrames(X_G, place_flipped=False):
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
        t_gripper_angle = X_G["place"] @ [0, 0, 0.12] # 12 cm is roughly the length of the gripper?
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
    X_G["postpick"] = RigidTransform(X_G["pick"].rotation(), X_G["pick"].translation() + [0, 0, 0.15])
    times["postpick"] = 1.0

    # Give time to get to start of display trajectory
    time_to_predisplay = 5.0 * np.linalg.norm(
        X_GprepickGpredisplay.translation()
    )
    # special case where first value is time to first frame in traj, and second is time to consecutive frames
    times["display_traj"] = [time_to_predisplay, 0.1] 

    # Prepare to place back down
    X_G["preplace"] = RigidTransform(X_G["place"].rotation(), X_G["place"].translation() + [0, 0, 0.15])
    times["preplace"] = time_to_predisplay

    # Place back down and allow some time for gripper to open
    X_G["place_start"] = X_G["place"]
    X_G["place_end"] = X_G["place"]
    times["place_start"] = 1.0
    times["place_end"] = 0.0

    # Go back to prepick pose
    X_GgraspGpostgrasp = RigidTransform([0, 0.0, -0.15])
    X_G["postplace"] = X_G["place"] @ X_GgraspGpostgrasp
    times["postplace"] = 1.0
    X_G["postpostplace"] = X_G["prepick"]
    times["postpostplace"] = 1.0

    return X_G, times

def MakePickAndDisplayJointPositionsTrajectory(X_G, times, plant, q, max_display_frames=6, place_intermediate=False):
    """
    Constructs a gripper position trajectory from the plan "sketch".
    Returns three piecewise polynomial trajectories. One for before grasp, one for during, one for after.
    This is in order to close the gripper between these two trajectories.

    X_G: map of gripper poses for each frame, with X_G["display_traj"] being a list of possible poses
        to center the display trajectory at
    times: map of time to get to each frame
    plant: plant with iiwa to solve for global IK
    q: iiwa starting position
    max_dislay_frames: maximum number of frames from X_G["display_traj"] to add to trajectory, prevents wasting time
        displaying already seen angles
    """
    sample_times1 = [0.0]
    positions1 = [q]
    sample_times2 = []
    positions2 = []
    sample_times3 = []
    positions3 = []
    q_prev = q
    positions3_failed = False
    q_prepick = None

    place_height = X_G["pick_start"].translation()[2] + 0.02
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
        "postpostplace"
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
                    orientation_tolerance=0.0,
                    gripper_frame_name="iiwa_link_7",
                )
                if q_display_center is not None:
                    # construct trajectory of rotating 7th joint
                    q7s = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi * 165.0 / 180.0, 
                               3*np.pi/4, np.pi/2, np.pi/4, 
                               0.0, -np.pi/4, -np.pi/2, -3*np.pi/4, -np.pi * 165.0 / 180.0]
                    for i in range(len(q7s)):
                        current_traj_positions = positions2[-1]
                        current_traj_times = sample_times2[-1]

                        q_temp = q_display_center.copy()
                        q_temp[6] = q7s[i]
                        current_traj_positions.append(q_temp)
                        if place_intermediate or i == 0:
                            current_traj_times.append(current_traj_times[-1] + times[name][0])
                        else:
                            current_traj_times.append(current_traj_times[-1] + times[name][1])

                        if place_intermediate:
                            X_hover = forward_kinematics(
                                plant=plant,
                                q=q_temp,
                                gripper_frame_name="iiwa_link_7",
                            )

                            new_translation = X_hover.translation().copy()
                            new_translation[2] = place_height
                            X_place = RigidTransform(X_hover.rotation(), new_translation)

                            q_place = solve_global_inverse_kinematics(
                                plant=plant,
                                X_G=X_place,
                                initial_guess=q_temp,
                                position_tolerance=0.0,
                                orientation_tolerance=0.0,
                                gripper_frame_name="iiwa_link_7",
                            )
                            current_traj_positions.append(q_place)
                            current_traj_times.append(current_traj_times[-1] + times[name][0])

                            next_traj_positions = [q_place]
                            next_traj_times = [0.0]

                            positions2.append(next_traj_positions)
                            sample_times2.append(next_traj_times)
                    center_found = True
                    q_prev = q_temp
                    break
            if not center_found:
                raise Exception("Failed to solve global IK for display trajectory center")
        else:
            if name == "prepick" or name == "pick_start":
                sample_times1.append((sample_times1[-1] if len(sample_times1) != 0 else 0) + times[name])
            elif name == "place_end" or name == "postplace" or name == "postpostplace":
                sample_times3.append((sample_times3[-1] if len(sample_times3) != 0 else 0) + times[name])
            elif name == "pick_end":
                sample_times2.append([times[name]])
            else:
                sample_times2[-1].append((sample_times2[-1][-1] if len(sample_times2[-1]) != 0 else 0) + times[name])


            if name == "postpostplace":
                positions3.append(q_prepick)
                continue

            q_next = solve_global_inverse_kinematics(
                plant=plant,
                X_G=X_G[name],
                initial_guess=q_prev,
                position_tolerance=0.0,
                orientation_tolerance=0.0,
                gripper_frame_name="iiwa_link_7",
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
                    )
                    attempts += 1

                if q_next is None:
                    print("IK failed again at", name)
                else:
                    print("phew.")
            
            if q_next is not None:
                q_prev = q_next

            if name == "prepick":
                q_prepick = q_next
            if name == "prepick" or name == "pick_start":
                positions1.append(q_next)
            elif (name == "place_end" or name == "postplace") and not positions3_failed:
                if q_next is None:
                    # use the prepick and pick start reversed if this fails since it should be the same thing
                    print(f"IK failed at {name}, using prepick/pick_start positions for postpick/place_end")
                    positions3 = list(positions1[1:].__reversed__())
                    positions3_failed = True
                else:
                    positions3.append(q_next)
            elif name == "pick_end":
                positions2.append([q_next])
            else:
                positions2[-1].append(q_next)
    
    t1 = PiecewisePolynomial.FirstOrderHold(sample_times1, np.array(positions1).T)
    t2 = []
    for i in range(len(sample_times2)):
        t2.append(PiecewisePolynomial.FirstOrderHold(sample_times2[i], np.array(positions2[i]).T))
    t3 = PiecewisePolynomial.FirstOrderHold(sample_times3, np.array(positions3).T)
    return t1, t2, t3
