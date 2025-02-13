import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Plot joint positions and torques over time from .npy files."
    )
    parser.add_argument(
        "folder", type=str, help="Path to the folder containing the .npy files"
    )
    args = parser.parse_args()
    folder = args.folder

    # Construct full file paths
    positions_path = os.path.join(folder, "joint_positions.npy")
    torques_path = os.path.join(folder, "joint_torques.npy")
    sample_times_path = os.path.join(folder, "sample_times_s.npy")

    # Load the data
    joint_positions = np.load(positions_path)
    joint_torques = np.load(torques_path)
    sample_times = np.load(sample_times_path)

    # Verify data dimensions
    if joint_positions.shape[1] != 7 or joint_torques.shape[1] != 7:
        raise ValueError("Expected the second dimension of joint_positions and joint_torques to be 7.")

    # Create a figure for joint positions with 7 subplots
    fig_pos, axs_pos = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
    for i in range(7):
        axs_pos[i].plot(sample_times, joint_positions[:, i], label=f"Joint {i+1}")
        axs_pos[i].set_ylabel(f"Joint {i+1}")
        axs_pos[i].grid(True)
    axs_pos[-1].set_xlabel("Time (s)")
    fig_pos.suptitle("Joint Positions over Time")
    fig_pos.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Create a figure for joint torques with 7 subplots
    fig_torque, axs_torque = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
    for i in range(7):
        axs_torque[i].plot(sample_times, joint_torques[:, i], label=f"Joint {i+1}")
        axs_torque[i].set_ylabel(f"Joint {i+1}")
        axs_torque[i].grid(True)
    axs_torque[-1].set_xlabel("Time (s)")
    fig_torque.suptitle("Joint Torques over Time")
    fig_torque.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Show the plots
    plt.show()

if __name__ == "__main__":
    main()
