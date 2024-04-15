import numpy as np
from pydrake.all import (
    PiecewisePolynomial,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw
)
from planning.inverse_kinematics import solve_global_inverse_kinematics

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

        # TODO: calculate translation difference of bottom of gripper given rotation (t is top of gripper, want to place object back in same place)
        t_gripper_angle = X_G["place"] @ [0, 0, 0.12] # 12 cm is roughly the length of the gripper?
        t_gripper_angle[2] = 0
        t_gripper_angle *= 2
        print(t_gripper_angle)
        X_G["place"].set_translation(t - t_gripper_angle)
        print("X_pick", X_G["pick"])
        print("X_place", X_G["place"])

    X_GprepickGpredisplay = X_G["prepick"].inverse() @ X_G["display_traj"][0][0]

    # Amount of time it takes to GET TO each frame
    times = {"prepick": 0.5}
    
    # Allow some time for the gripper to close.
    X_G["pick_start"] = X_G["pick"]
    X_G["pick_end"] = X_G["pick"]
    times["pick_start"] = 5.0
    times["pick_end"] = 2.0

    # raise object off surface
    X_G["postpick"] = RigidTransform(X_G["pick"].rotation(), X_G["pick"].translation() + [0, 0, 0.15])
    times["postpick"] = 2.0

    # Give time to get to start of display trajectory
    time_to_predisplay = 10.0 * np.linalg.norm(
        X_GprepickGpredisplay.translation()
    )
    # special case where first value is time to first frame in traj, and second is time to consecutive frames
    times["display_traj"] = [time_to_predisplay, 0.5] 

    # Prepare to place back down
    X_G["preplace"] = RigidTransform(X_G["place"].rotation(), X_G["place"].translation() + [0, 0, 0.15])
    times["preplace"] = time_to_predisplay

    # Place back down and allow some time for gripper to open
    X_G["place_start"] = X_G["place"]
    X_G["place_end"] = X_G["place"]
    times["place_start"] = 2.0
    times["place_end"] = 2.0

    # Go back to prepick pose
    X_GgraspGpostgrasp = RigidTransform([0, 0.0, -0.15])
    X_G["postplace"] = X_G["place"] @ X_GgraspGpostgrasp
    times["postplace"] = 1.0
    X_G["postpostplace"] = X_G["prepick"]
    times["postpostplace"] = 1.0

    return X_G, times

def MakePickAndDisplayJointPositionsTrajectory(X_G, times, plant, q, max_display_frames=6):
    """
    Constructs a gripper position trajectory from the plan "sketch".
    Returns three piecewise polynomial trajectories. One for before grasp, one for during, one for after.
    This is in order to close the gripper between these two trajectories.

    X_G: map of gripper poses for each frame, with X_G["display_traj"] being a tuple of two lists, one with frames
        of the display trajectory in one direction, and one with frames of the display trajectory in the other direction
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
            display_frames = 0
            # display direction 1 till failure
            for i in range(len(X_G["display_traj"][0])):
                if display_frames == max_display_frames:
                    break
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
                    display_frames += 1
                else:
                    print("IK failed at display 1 index", i)
                    if i > 1:
                        last_t = sample_times2[-1]
                        sample_times2 += [last_t + j * times["display_traj"][1] for j in range(1, i-1)]
                        reversed_positions = list(positions2[-i:-1].__reversed__())
                        if len(reversed_positions) > 0:
                            positions2 += reversed_positions[:-1] # add going backwards, dont add final one since that's assumed to be the first position of the next direction
                        q_prev = reversed_positions[-1] # start initial guess at first position, since it should be the first position of the next direction
                    break

            # display direction 2 till failure
            for i in range(len(X_G["display_traj"][1])):
                if display_frames == max_display_frames:
                    break
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
                    if i != 0: # don't double count initial frame, already counted in direction 1
                        display_frames += 1
                else:
                    print("IK failed at display 2 index", i)
                    if i > 1:
                        last_t = sample_times2[-1]
                        sample_times2 += [last_t + j * times["display_traj"][1] for j in range(1, i)]
                        positions2 += list(positions2[-i:-1].__reversed__()) # add going backwards
                    break
            
            q_prev = positions2[-1]
        else:
            if name == "prepick" or name == "pick_start":
                sample_times1.append((sample_times1[-1] if len(sample_times1) != 0 else 0) + times[name])
            elif name == "place_end" or name == "postplace" or name == "postpostplace":
                sample_times3.append((sample_times3[-1] if len(sample_times3) != 0 else 0) + times[name])
            else:
                sample_times2.append((sample_times2[-1] if len(sample_times2) != 0 else 0) + times[name])

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
            if name == "prepick":
                q_prepick = q_next
            if name == "prepick" or name == "pick_start":
                positions1.append(q_next)
            elif (name == "place_end" or name == "postplace") and not positions3_failed:
                if q_next is None:
                    # use the prepick and pick start reversed if this fails since it should be the same thing
                    print(f"IK failed at {name}, using prepick/pick_start positions for postpick/place_end")
                    positions3 = list(reversed(positions1[1:]))
                    positions3_failed = True
                else:
                    positions3.append(q_next)
            else:
                positions2.append(q_next)

    sample_times2 = [t - sample_times2[0] for t in sample_times2]
    sample_times3 = [t - sample_times3[0] for t in sample_times3]
    
    t1 = PiecewisePolynomial.FirstOrderHold(sample_times1, np.array(positions1).T)
    t2 = PiecewisePolynomial.FirstOrderHold(sample_times2, np.array(positions2).T)
    t3 = PiecewisePolynomial.FirstOrderHold(sample_times3, np.array(positions3).T)
    return t1, t2, t3
