import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import argparse
import os

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale.

    Args:
        ax: a matplotlib axis, e.g., as output from plt.gca().
    """
    # Extract current limits
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of equal aspect ratio
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Plot joint positions, joint torques, wsg positions, and visualize a point cloud from .npy files."
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
    wsg_positions_path = os.path.join(folder, "wsg_positions.npy")
    point_cloud_path = os.path.join(folder, "manipuland_cloud_link7_frame.npy")

    # Load the data
    joint_positions = np.load(positions_path)
    joint_torques = np.load(torques_path)
    sample_times = np.load(sample_times_path)
    wsg_positions = np.load(wsg_positions_path)
    point_cloud = np.load(point_cloud_path)

    # Verify expected dimensions for joint positions and torques
    if joint_positions.shape[1] != 7 or joint_torques.shape[1] != 7:
        raise ValueError("Expected the second dimension of joint_positions and joint_torques to be 7.")

    # ------------------------------
    # Plot Joint Positions over Time
    # ------------------------------
    fig_pos, axs_pos = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
    for i in range(7):
        axs_pos[i].plot(sample_times, joint_positions[:, i], label=f"Joint {i+1}")
        axs_pos[i].set_ylabel(f"Joint {i+1}")
        axs_pos[i].grid(True)
    axs_pos[-1].set_xlabel("Time (s)")
    fig_pos.suptitle("Joint Positions over Time")
    fig_pos.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ------------------------------
    # Plot Joint Torques over Time
    # ------------------------------
    fig_torque, axs_torque = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
    for i in range(7):
        axs_torque[i].plot(sample_times, joint_torques[:, i], label=f"Joint {i+1}")
        axs_torque[i].set_ylabel(f"Joint {i+1}")
        axs_torque[i].grid(True)
    axs_torque[-1].set_xlabel("Time (s)")
    fig_torque.suptitle("Joint Torques over Time")
    fig_torque.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ------------------------------
    # Plot WSG Positions over Time
    # ------------------------------
    fig_wsg, ax_wsg = plt.subplots(figsize=(10, 4))
    ax_wsg.plot(sample_times, wsg_positions, label="WSG Position", color='green')
    ax_wsg.set_xlabel("Time (s)")
    ax_wsg.set_ylabel("WSG Position")
    ax_wsg.set_title("WSG Positions over Time")
    ax_wsg.grid(True)
    ax_wsg.legend()

    # ------------------------------
    # Visualize the Point Cloud
    # ------------------------------
    fig_pc = plt.figure(figsize=(8, 8))
    ax_pc = fig_pc.add_subplot(111, projection='3d')
    ax_pc.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                  s=1, c='blue', depthshade=True)
    ax_pc.set_xlabel("X")
    ax_pc.set_ylabel("Y")
    ax_pc.set_zlabel("Z")
    ax_pc.set_title("Point Cloud: manipuland_cloud_link7_frame")
    
    # Adjust the axes to have the same scale
    set_axes_equal(ax_pc)

    # Show all plots
    plt.show()

if __name__ == "__main__":
    main()
