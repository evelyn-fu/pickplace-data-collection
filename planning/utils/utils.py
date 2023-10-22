# All taken from Nicholas' code

import open3d as o3d
import numpy as np
import os
import sys
import random

import scipy.io as sio
import IPython
import time
# import ros_numpy
from scipy.spatial.transform import Rotation
# from geometry_msgs.msg import Transform

import cv2

import matplotlib.pyplot as plt
import tabulate
import yaml

import copy
import math
from torch.optim import Adam
from collections import deque
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import shutil

import numpy as np
import sys
import os
import scipy
# from geometry_msgs.msg import Pose, PoseArray
# from std_msgs.msg import Header
# from sensor_msgs.msg import PointCloud2
# import rospy
# from transforms3d.quaternions import quat2mat, mat2quat
# from transforms3d.euler import euler2mat, mat2euler, euler2quat
# from colored import fg
import signal
import GPUtil
import psutil
import sympy
from pydrake.all import PathParameterizedTrajectory
from typing import Iterable, Tuple, Optional, Union

np.set_printoptions(precision=3)
# predefined lists for replay buffer, environment, and policy
key_list = [
    "state",
    "action",
    "reward",
    "next_state",
    "done",
    "obs",
    "timestep",
    "joint_pos",
    "hand_camera_pose",
    "base_camera_pose",
    "ee_pose",
    "cam_intr",
    "curr_keypoint",
    "goal_keypoint",
    "goal_pose",
    "keypoint_wrench",
    "external_torque",
    "endeffector_wrench",
    "history_endeffector_wrench",
    "point_cloud",
    "tool_rel_pose",
]

key_shape_list = [
    (27,),
    (6,),
    (1,),
    (27,),
    (1,),
    (480, 640, 10),
    (1,),
    (9,),
    (4, 4),
    (4, 4),
    (16,),
    (3, 3),
    (4, 3),
    (3, 3),
    (4, 4),
    (3, 6),
    (9,),
    (6,),
    (6 * 32,),
    (4, 1024),
    (4, 4),
]

anchor_seeds = np.array(
    [
        [0.0, -1.285, 0, -2.356, 0.0, 1.571, 0.785, 0, 0],
        [2.5, 0.23, -2.89, -1.69, 0.056, 1.46, -1.27, 0, 0],
        [2.8, 0.23, -2.89, -1.69, 0.056, 1.46, -1.27, 0, 0],
        [2, 0.23, -2.89, -1.69, 0.056, 1.46, -1.27, 0, 0],
        [2.5, 0.83, -2.89, -1.69, 0.056, 1.46, -1.27, 0, 0],
        [0.049, 1.22, -1.87, -0.67, 2.12, 0.99, -0.85, 0, 0],
        [-2.28, -0.43, 2.47, -1.35, 0.62, 2.28, -0.27, 0, 0],
        [-2.02, -1.29, 2.20, -0.83, 0.22, 1.18, 0.74, 0, 0],
        [-2.2, 0.03, -2.89, -1.69, 0.056, 1.46, -1.27, 0, 0],
        [-2.5, -0.71, -2.73, -0.82, -0.7, 0.62, -0.56, 0, 0],
        [-2, -0.71, -2.73, -0.82, -0.7, 0.62, -0.56, 0, 0],
        [-2.66, -0.55, 2.06, -1.77, 0.96, 1.77, -1.35, 0, 0],
        [1.51, -1.48, -1.12, -1.55, -1.57, 1.15, 0.24, 0, 0],
        [-2.61, -0.98, 2.26, -0.85, 0.61, 1.64, 0.23, 0, 0],
    ]
)


def rotZ(rotz):
    RotZ = np.array(
        [[np.cos(rotz), -np.sin(rotz), 0, 0], [np.sin(rotz), np.cos(rotz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    return RotZ


def rotY(rotz):
    RotZ = np.array(
        [[np.cos(rotz), 0, np.sin(rotz), 0], [0, 1, 0, 0], [-np.sin(rotz), 0, np.cos(rotz), 0], [0, 0, 0, 1]]
    )
    return RotZ


def rotX(rotz):
    RotZ = np.array(
        [[1, 0, 0, 0], [0, np.cos(rotz), -np.sin(rotz), 0], [0, np.sin(rotz), np.cos(rotz), 0], [0, 0, 0, 1]]
    )
    return RotZ


# Relative to base 0
OVERHEAD_CAM_EXTR = np.eye(4)
OVERHEAD_CAM_EXTR[:3, 3] = 0.611968, -0.153414, 0.847576
# 0.855454, -0.164402, 0.895949
# OVERHEAD_CAM_EXTR[:3, :3] = quat2mat((0.725208, 0.678776, -0.026333, -0.112447))
# quat2mat((-0.0821656, 0.913104, 0.378826, -0.126417))
# needed extra conversion
# OVERHEAD_CAM_EXTR = OVERHEAD_CAM_EXTR.dot(rotZ(-np.pi / 2))

# D-435 rs-enumerate-devices --calib_data
# Intrinsic of "Color" / 640x480 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y16}
#  Width:        640
#  Height:       480
#  PPX:          324.458282470703
#  PPY:          241.954818725586
#  Fx:           616.904418945312
#  Fy:           617.003967285156
#  Distortion:   Inverse Brown Conrady
#  Coeffs:       0   0   0   0   0
#  FOV (deg):    54.83 x 42.51

# D-415
# Intrinsic of "Color" / 640x480 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y16}
#  Width:        640
#  Height:       480
#  PPX:          321.994598388672
#  PPY:          236.73258972168
#  Fx:           616.616149902344
#  Fy:           615.9443359375
#  Distortion:   Inverse Brown Conrady
#  Coeffs:       0   0   0   0   0
#  FOV (deg):    54.85 x 42.58


# needed extra conversion
# 0.0360556 -0.036485 0.057091   0.00554682 -0.000890409 0.704314 0.709866

# Relative to hand
HAND_CAM_EXTR = np.eye(4)
HAND_CAM_EXTR[:3, 3] = 0.030404, -0.0323965, 0.0618477
# HAND_CAM_EXTR[:3, :3] = quat2mat((0.00834598, -0.00171003, 0.706, 0.70816))


CAM_INTR_D415 = np.array(
    [
        616.616149902344,
        0.0,
        321.994598388672,
        0.0,
        615.9443359375,
        236.73258972168,
        0.0,
        0.0,
        1.0,
    ]
).reshape(3, 3)


CAM_INTR_D435 = np.array(
    [
        616.904418945312,
        0.0,
        324.458282470703,
        0.0,
        617.003967285156,
        241.954818725586,
        0.0,
        0.0,
        1.0,
    ]
).reshape(3, 3)


def pad_zero_finger_joint(target_joint):
    # pad with full opening
    return np.concatenate((target_joint, np.ones(2) * 0.04))


def ros_quat(tf_quat):
    """Converts a quaternion from wxyz to xyzw format."""
    quat = np.zeros(4)
    quat[-1] = tf_quat[0]
    quat[:-1] = tf_quat[1:]
    return quat


# def unpack_pose(pose, rot_type="quaternion"):
#     """
#     :param pose: A pose of form [tx,ty,tz,qx,qy,qz,qw] or [tx,ty,tz,rx,ry,rz].
#     :param rot_type: "quaternion" or "euler"
#     :return: A homogenous transform of shape (4,4).
#     """
#     unpacked = np.eye(4)
#     if rot_type == "quaternion":
#         assert len(pose[3:]) == 4, f"Rotation {pose[3:]} is not a quaternion."
#         unpacked[:3, :3] = quat2mat(pose[3:])
#     elif rot_type == "euler":
#         assert len(pose[3:]) == 3, f"Rotation {pose[3:]} is not in euler angle form."
#         unpacked[:3, :3] = euler2mat(*pose[3:], axes="sxyz")
#     else:
#         print(f"unpack_pose: Invalid rot_type: {rot_type}")
#         exit()
#     unpacked[:3, 3] = pose[:3]
#     return unpacked


# def pack_pose(pose):
#     """
#     :param pose: A homogenous transform of shape (4,4).
#     :return: A pose of form [tx,ty,tz,qx,qy,qz,qw].
#     """
#     packed = np.zeros(7)
#     packed[:3] = pose[:3, 3]
#     packed[3:] = mat2quat(pose[:3, :3])
#     return packed


# def unpack_posemsg(pose, mat=False):
#     orn = pose.pose.orientation
#     pos = pose.pose.position

#     orientation = [orn.w, orn.x, orn.y, orn.z]
#     position = [pos.x, pos.y, pos.z]
#     pose_list = position + orientation
#     if mat:
#         return unpack_pose(pose_list)


# def unpack_posemsg_to_dict(pose):
#     orn = pose.pose.orientation
#     pos = pose.pose.position
#     pose_dict = {"position": [pos.x, pos.y, pos.z], "orientation": orn}
#     return pose_dict


# def make_pose(pose, frame_id="base_link"):
#     # take pose matrix
#     posemsg = Pose()
#     quat = mat2quat(pose[:3, :3])
#     posemsg.orientation.x = quat[1]
#     posemsg.orientation.y = quat[2]
#     posemsg.orientation.z = quat[3]
#     posemsg.orientation.w = quat[0]
#     posemsg.position.x = pose[0, 3]
#     posemsg.position.y = pose[1, 3]
#     posemsg.position.z = pose[2, 3]
#     return posemsg


# def make_posearray(poses, frame_id="base_link"):
#     msgs = PoseArray()
#     msgs.header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
#     for idx, pose in enumerate(poses):
#         msgs.poses.append(make_pose(pose))
#     return msgs


def se3_inverse(RT):
    R = RT[:3, :3]
    T = RT[:3, 3].reshape((3, 1))
    RT_new = np.eye(4, dtype=np.float32)
    RT_new[:3, :3] = R.transpose()
    RT_new[:3, 3] = -1 * np.dot(R.transpose(), T).reshape((3))
    return RT_new


def make_gripper_pts(points, color=(1, 0, 0)):
    # o3d.visualization.RenderOption.line_width = 8.0
    line_index = [[0, 1], [1, 2], [1, 3], [3, 5], [2, 4]]

    cur_gripper_pts = points.copy()
    cur_gripper_pts[1] = (cur_gripper_pts[2] + cur_gripper_pts[3]) / 2.0
    line_set = o3d.geometry.LineSet()

    line_set.points = o3d.utility.Vector3dVector(cur_gripper_pts)
    line_set.lines = o3d.utility.Vector2iVector(line_index)
    line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(line_index))])
    return line_set


# def make_pose(pose):
#     # take pose matrix
#     posemsg = Pose()
#     quat = mat2quat(pose[:3, :3])  # quaternion_from_matrix
#     posemsg.orientation.x = quat[1]
#     posemsg.orientation.y = quat[2]
#     posemsg.orientation.z = quat[3]
#     posemsg.orientation.w = quat[0]
#     posemsg.position.x = pose[0, 3]
#     posemsg.position.y = pose[1, 3]
#     posemsg.position.z = pose[2, 3]
#     return posemsg


def backproject_camera_target(im_depth, K, target_mask=None):
    """
    :param im_depth: Depth image of shape (H,W).
    :param pcd_image: Point cloud image of shape (3,H,W).
    """
    Kinv = np.linalg.inv(K)

    width = im_depth.shape[1]
    height = im_depth.shape[0]
    depth = im_depth.astype(np.float32, copy=True).flatten()

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)  # each pixel

    # backprojection
    R = Kinv.dot(x2d.transpose())
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)

    return X.reshape(-1, height, width)


def transform_point(pose, pc):
    """transform points with the given pose
    param pose: 4 x 4
    param pc: 3 x N
    """
    return pose[:3, :3] @ pc + pose[:3, [3]]


# def downsample_point(pc, num_pt):
#     """param pc: B x N x 3
#     return 4 x num_pt
#     """
#     if type(pc) is not torch.Tensor:
#         pc = torch.from_numpy(pc).cuda().float()
#     if len(pc.shape) == 2:
#         pc = pc[None].cuda().float()
#     if pc.shape[1] == 0:
#         return torch.zeros((pc.shape[-1], 0)).float()
#     xyz = gather_operation(
#         pc.transpose(1, 2).contiguous(),
#         furthest_point_sample(pc[..., :3].contiguous(), num_pt),
#     ).contiguous()
#     return xyz[0].detach()


def convert_dict_to_joint(joints, joint_names):
    return np.array([joints[name] for name in joint_names])


def convert_dict_to_wrench(wrench):
    return np.concatenate((wrench["force"], wrench["torque"]))


def convert_joint_to_dict(joints, joint_names):
    return {name: joints[idx] for idx, name in enumerate(joint_names)}


def color_print(str, color="white"):
    print(str)  #


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def mkdir_if_missing(dst_dir, cleanup=False):
    if os.path.exists(dst_dir) and cleanup:
        shutil.rmtree(dst_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def print_and_write(file_handle, text):
    print(text)
    if file_handle is not None:
        file_handle.write(text + "\n")
    return text


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.sum_2 = 0
        self.count_2 = 0
        self.means = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sum_2 += val * n
        self.count_2 += n

    def set_mean(self):
        self.means.append(self.sum_2 / self.count_2)
        self.sum_2 = 0
        self.count_2 = 0

    def std(self):
        return np.std(np.array(self.means) + 1e-4)

    def __repr__(self):
        return "{:.3f} ({:.3f})".format(self.val, self.avg)


def module_max_param(module):
    def maybe_max(x):
        return float(torch.abs(x).max()) if x is not None else 0

    max_data = np.amax([(maybe_max(param.data)) for name, param in module.named_parameters()])
    return max_data


def module_mean_param(module):
    def maybe_mean(x):
        return float(torch.abs(x).mean()) if x is not None else 0

    max_data = np.mean([(maybe_mean(param.data)) for name, param in module.named_parameters()])
    return max_data


def module_max_gradient(module):
    def maybe_max(x):
        return torch.abs(x).max().item() if x is not None else 0

    max_grad = np.amax([(maybe_max(param.grad)) for name, param in module.named_parameters()])
    return max_grad


def normalize(v, axis=None, eps=1e-10):
    """L2 Normalize along specified axes."""
    return v / max(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def get_usage():
    GPUs = GPUtil.getGPUs()
    memory_usage = psutil.virtual_memory().percent
    gpu_usage = max([GPU.memoryUsed for GPU in GPUs])
    return gpu_usage, memory_usage


def visualize_image_observation(state):
    if type(state) is not np.ndarray:
        state = state.detach().cpu().numpy()

    for idx in range(0, len(state), 2):
        fig = plt.figure(figsize=(25.6, 9.6))
        ax = fig.add_subplot(1, 3, 1)
        plt.imshow((state[idx][:3].transpose(1, 2, 0) * 255).astype(np.uint8))

        if state[idx].shape[0] == 4:
            ax = fig.add_subplot(1, 3, 2)
            # IPython.embed()
            plt.imshow(state[idx][-1])

        if state[idx].shape[0] >= 5:
            ax = fig.add_subplot(1, 3, 2)
            plt.imshow(state[idx][-2])

            ax = fig.add_subplot(1, 3, 3)
            mask = (state[idx][-1]).astype(np.uint8)
            plt.imshow(mask)
        plt.show()
        # break


def projection_to_intrinsics(mat, width=224, height=224):
    intrinsic_matrix = np.eye(3)
    mat = np.array(mat).reshape([4, 4]).T
    fv = width / 2 * mat[0, 0]
    fu = height / 2 * mat[1, 1]
    u0 = width / 2
    v0 = height / 2

    intrinsic_matrix[0, 0] = fu
    intrinsic_matrix[1, 1] = fv
    intrinsic_matrix[0, 2] = u0
    intrinsic_matrix[1, 2] = v0
    return intrinsic_matrix


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


# from Isaac example
def control_ik(dpose, j_eef):
    # solve damped least squares
    damping = 0.05
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=j_eef.device) * (damping**2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose[..., None]).view(len(dpose), 7)
    return u


def control_osc(dpose, mm, default_dof_pos_tensor, j_eef, dof_pos, dof_vel, hand_vel):
    kp = 150.0
    kd = 2.0 * np.sqrt(kp)
    kp_null = 10
    kd_null = 2 * np.sqrt(kp_null)  # 2.0
    dof_pos = dof_pos.unsqueeze(-1)
    dpose = dpose.unsqueeze(-1)
    dof_vel = dof_vel.unsqueeze(-1)

    mm_inv = torch.inverse(mm.clone())
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (kp * dpose - kd * hand_vel.unsqueeze(-1))

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -dof_vel + kp_null * (
        (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi
    )
    u_null = u_null[:, :7]
    u_null = mm @ u_null
    u += (torch.eye(7, device=j_eef.device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
    return u.squeeze(-1)


def compute_frames(T_ee_to_world, point_cloud):
    index_split = point_cloud.shape[1] // 2
    tool_pts_in_ee = point_cloud[:, :index_split].copy()
    obj_pts_in_ee = point_cloud[:, index_split:].copy()

    # centered at the mean of the object point cloud and with rotation the same as base
    # object transform is the transform from end effector frame to object frame T_ee_to_obj
    # from to notation
    T_world_to_ee = se3_inverse(T_ee_to_world)

    p_obj_to_ee = obj_pts_in_ee[:3].mean(axis=1)
    p_obj_to_world = T_ee_to_world[:3, :3] @ p_obj_to_ee + T_ee_to_world[:3, 3]
    T_obj_to_world = np.eye(4)
    T_obj_to_world[:3, 3] = p_obj_to_world

    T_obj_to_ee = T_world_to_ee @ T_obj_to_world
    T_ee_to_obj = se3_inverse(T_obj_to_ee)

    # obj is static
    T_ee_to_obj_rot_only = T_ee_to_obj.copy()
    T_ee_to_obj_rot_only[:3, 3] = 0  #

    # the mask
    tool_pts_in_obj = tool_pts_in_ee.copy()
    obj_pts_in_obj = obj_pts_in_ee.copy()

    tool_pts_in_obj[:3] = T_ee_to_obj[:3, :3] @ tool_pts_in_ee[:3] + T_ee_to_obj[:3, [3]]
    obj_pts_in_obj[:3] = T_ee_to_obj[:3, :3] @ obj_pts_in_ee[:3] + T_ee_to_obj[:3, [3]]

    return (
        T_ee_to_obj_rot_only,
        T_ee_to_obj,
        T_obj_to_ee,
        T_obj_to_world,
        p_obj_to_ee,
        T_world_to_ee,
        T_ee_to_world,
        obj_pts_in_ee,
        tool_pts_in_ee,
        tool_pts_in_obj,
        obj_pts_in_obj,
    )


def module_max_param(module):
    def maybe_max(x):
        return float(torch.abs(x).max()) if x is not None else 0

    max_data = np.amax([(maybe_max(param.data)) for name, param in module.named_parameters()])
    return max_data


def module_max_gradient(module):
    def maybe_max(x):
        return torch.abs(x).max().item() if x is not None else 0

    max_grad = np.amax([(maybe_max(param.grad)) for name, param in module.named_parameters()])
    return max_grad


def inv_lookat(eye, target=[0, 0, 0], up=[0, 1, 0]):
    """Generate LookAt matrix."""
    up = np.random.uniform(size=(3,))
    eye = np.float32(eye)  #  trans - target
    forward = normalize(target - eye)  #  trans - target
    side = normalize(np.cross(forward, up))
    up = np.cross(side, forward)  # redo

    R = np.stack([side, up, -forward], axis=-1)
    return R


def safe_div(a, b):
    return a / (b + 1e-10)


def save_args_json(path, args):
    mkdir_if_missing(path)
    arg_json = os.path.join(path, "config.json")
    with open(arg_json, "w") as f:
        # args = vars(args)
        json.dump(args, f, indent=4, sort_keys=True)


def str_presenter(dumper, data):
    """configures yaml for dumping multiline strings
    Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data"""
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def save_args_yaml(path, args):
    mkdir_if_missing(path)
    arg_json = os.path.join(path, "config.yaml")
    # print("config path:", arg_json)
    yaml.add_representer(str, str_presenter)
    yaml.representer.SafeRepresenter.add_representer(str, str_presenter)  # to use with safe_dum
    from easydict import EasyDict as edict

    # IPython.embed()
    # s  = edict(args)
    #
    with open(arg_json, "w") as f:
        yaml.dump(args, f, default_flow_style=False)


def make_video_writer(name, window_width, window_height):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(name, fourcc, 10.0, (window_width, window_height))


def write_video(
    traj,
    scene_file,
    expert_traj=None,
    IMG_SIZE=(112, 112),
    output_dir="output_misc/",
    target_name="",
    extra_text=None,
):
    ratio = 1 if expert_traj is None else 2
    video_writer = make_video_writer(
        os.path.join(
            output_dir + "_{}_rollout.avi".format(scene_file),
        ),
        int(ratio * IMG_SIZE[1]),
        int(IMG_SIZE[0]),
    )

    for i in range(len(traj)):
        img = traj[i][..., :3] * 255
        # cv2.resize(, (IMG_SIZE[:2]), interpolation = cv2.INTER_AREA)[..., [2, 1, 0]]
        if expert_traj is not None:
            idx = min(len(expert_traj) - 1, i)
            img = np.concatenate((img, expert_traj[idx][..., [2, 1, 0]]), axis=1)

        img = img.astype(np.uint8)
        if extra_text is not None:
            img = add_extra_text(img, extra_text)
        video_writer.write(img)


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def PandaTaskSpace6D(trans_scale=0.04, rotation_scale=np.pi / 8):
    high = np.array(
        [
            trans_scale,
            trans_scale,
            trans_scale,
            rotation_scale,
            rotation_scale,
            rotation_scale,
        ]
    )  # np.pi/10
    low = np.array(
        [
            -trans_scale,
            -trans_scale,
            -trans_scale,
            -rotation_scale,
            -rotation_scale,
            -rotation_scale,
        ]
    )  # -np.pi/3
    shape = [6]
    # self.bounds = np.vstack([self.low, self.high])
    return gym.spaces.Box(low=low, high=high, shape=shape)


def chromatic_transform(im, label=None, d_h=None, d_s=None, d_l=None):
    """
    Given an image array, add the hue, saturation and luminosity to the image
    """
    # Set random hue, luminosity and saturation which ranges from -0.1 to 0.1
    if d_h is None:
        d_h = (np.random.rand(1) - 0.5) * 0.1 * 180
    if d_l is None:
        d_l = (np.random.rand(1) - 0.5) * 0.2 * 256
    if d_s is None:
        d_s = (np.random.rand(1) - 0.5) * 0.2 * 256
    # Convert the BGR to HLS
    hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
    h, l, s = cv2.split(hls)
    # Add the values to the image H, L, S
    new_h = (h + d_h) % 180
    new_l = np.clip(l + d_l, 0, 255)
    new_s = np.clip(s + d_s, 0, 255)
    # Convert the HLS to BGR
    new_hls = cv2.merge((new_h, new_l, new_s)).astype("uint8")

    new_im = cv2.cvtColor(new_hls, cv2.COLOR_HLS2RGB)
    # print(new_im.max(), new_im.min())
    if label is not None:
        I = np.where(label > 0)
        new_im[I[0], I[1], :] = im[I[0], I[1], :]
    return new_im


def add_noise(image, level=0.05, motion_blur_p=0.2):
    # random number
    r = np.random.rand(1)

    # gaussian noise
    if r < 1 - motion_blur_p:
        row, col, ch = image.shape
        mean = 0
        var = np.random.rand(1) * level * 256
        sigma = var**0.5
        gauss = sigma * np.random.randn(row, col) + mean
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
    else:
        # motion blur
        sizes = [3, 5, 7, 9, 11, 15]
        size = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = np.zeros((size, size))
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        else:
            kernel_motion_blur[:, int((size - 1) / 2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        noisy = cv2.filter2D(image, -1, kernel_motion_blur)

    return noisy.astype("uint8")


def get_interp_time(curr_time, finish_time, ratio):
    """get interpolated time between curr and finish"""
    return (finish_time - curr_time) * ratio + curr_time


def make_gripper_pts(points, color=(1, 0, 0)):
    # o3d.visualization.RenderOption.line_width = 8.0
    line_index = [[0, 1], [1, 2], [1, 3], [3, 5], [2, 4]]

    cur_gripper_pts = points.copy()
    cur_gripper_pts[1] = (cur_gripper_pts[2] + cur_gripper_pts[3]) / 2.0
    line_set = o3d.geometry.LineSet()

    line_set.points = o3d.utility.Vector3dVector(cur_gripper_pts)
    line_set.lines = o3d.utility.Vector2iVector(line_index)
    line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(line_index))])
    return line_set


def save_args_hydra(path, cfg):
    from omegaconf import OmegaConf

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    save_args_json(path, cfg_dict)


def tabulate_print_state(state_dict):
    """
    Print state dict
    """
    state_dict = sorted(state_dict.items())
    headers = [kv[0] for kv in state_dict]
    data = [[kv[1] for kv in state_dict]]
    print(tabulate.tabulate(data, headers, tablefmt="psql", floatfmt=".2f"))


def vis_batch(aux_state, vis_num=1):
    """
    Visualize gripper and point clouds in Open3d
    """
    batch = len(aux_state)

    for idx in range(0, batch, batch // vis_num):
        points = aux_state[idx].detach().cpu().numpy().copy()
        pcd = o3d.geometry.PointCloud()
        # print('curr:', points.T[:512, :3].mean(0))
        # print('object:', points.T[512:1024, :3].mean(0))
        # print('goal:', points.T[1024:, :3].mean(0))
        # IPython.embed()
        pcd.points = o3d.utility.Vector3dVector(points.T[:, :3])
        pc_colors = np.zeros_like(pcd.points)
        pc_colors[points[-1] == 0] = 0, 255, 255
        pc_colors[points[-1] == 1] = 255, 0, 0
        pc_colors[points[-1] == 2] = 0, 255, 0
        pc_colors[points[-1] == 3] = 0, 0, 255  # goal
        pc_colors[points[-1] == 4] = 255, 0, 255
        pcd.colors = o3d.utility.Vector3dVector(pc_colors)
        geometries = [pcd]

        # if (points[-1] == 2).sum() > 0:  # visualize gt
        #     gt_lines = make_gripper_pts(points.T[(points[-1] == 2), :3], (0, 1, 0))
        #     geometries.append(gt_lines)
        o3d.visualization.draw_geometries(geometries)


def proj_point_img(
    img,
    K,
    offset_pose,
    points,
    color=(255, 0, 0),
    vis=False,
    neg_y=False,
    real_world=False,
):
    xyz_points = offset_pose[:3, :3].dot(points) + offset_pose[:3, [3]]
    if neg_y:
        xyz_points[:2] *= -1
    p_xyz = K.dot(xyz_points)
    p_xyz = p_xyz[:, p_xyz[2] > 0.03]
    x, y = (p_xyz[0] / p_xyz[2]).astype(int), (p_xyz[1] / p_xyz[2]).astype(int)
    valid_idx_mask = (x > 0) * (x < img.shape[1] - 1) * (y > 0) * (y < img.shape[0] - 1)
    # IPython.embed()
    img[y[valid_idx_mask], x[valid_idx_mask], :3] = [0, 1, 1]
    return img


# def xyz_array_to_pointcloud2(points: np.ndarray) -> PointCloud2:
#     """
#     :param points: Points of shape (N,3).
#     NOTE: The returned msg does not yet contain a header.
#     """
#     data = np.zeros(
#         len(points),
#         dtype=[
#             ("x", np.float32),
#             ("y", np.float32),
#             ("z", np.float32),
#         ],
#     )
#     data["x"] = points[:, 0]
#     data["y"] = points[:, 1]
#     data["z"] = points[:, 2]
#     return ros_numpy.point_cloud2.array_to_pointcloud2(data)


def get_calibration_info(panda, joint_cmd):
    """get the information required for calibration"""

    extra_info = {}
    extra_info["joint_position_commanded"] = joint_cmd
    joint_name = panda.panda_arm._joint_names
    extra_info["joint_position_measured"] = convert_dict_to_joint(panda.panda_arm.joint_angles(), joint_name)
    extra_info["joint_velocity_estimated"] = convert_dict_to_joint(panda.panda_arm.joint_velocities(), joint_name)
    extra_info["joint_torque_measured"] = convert_dict_to_joint(panda.panda_arm.joint_efforts(), joint_name)
    extra_info["cartesian_measured"] = convert_dict_to_wrench(panda.panda_arm._cartesian_effort)
    extra_info["external_torque"] = panda.panda_arm.get_external_torque()
    # external_tau from ros # env.external_torque
    # extra_info["joint_torque_commanded"] = panda.panda_torque_commanded_port.Eval(env.context)  # not sure how to get this one
    return extra_info


def generate_sinuisodal_traj(
    env, total_time, prime_start_idx=1, initial_q=None, amp=0.1, vis=False, freq_divident=12, freq_scaling=40.0
):
    # For persistent excitation, sinuosidial plans for each of the 7 Franka joints
    # were sent. The frequencies were selected from prime numbers so that they never
    # sync with each other. The amplitudes were 0.8 rads.
    dt = 0.2  # env.env_dt should not matter. will be retimed.
    joint_min_limit = env.joint_min_limit
    joint_max_limit = env.joint_max_limit
    # print(joint_min_limit, joint_max_limit)
    joint_init = env.joint_positions if initial_q is None else initial_q
    joint_plans = []

    mode = "single"
    if mode == "concat":
        joint_plans = []
        time_traj = np.arange(0.0, total_time * 7, dt)
        for joint_idx in range(7):
            joint_plan = []
            for i in range(7):
                p = sympy.prime(prime_start_idx + joint_idx + i) % freq_divident
                freq = p / freq_scaling

                times = np.arange(total_time * i, total_time * (i + 1), dt)
                init_joint = joint_init[joint_idx]
                sin_wave = amp * np.sin(2 * np.pi * freq * times) + init_joint
                joint_plan = np.concatenate((joint_plan, sin_wave))

            print(joint_plan)
            joint_plans.append(joint_plan)

        joint_plans = np.array(joint_plans).T
        joint_plans = np.concatenate((joint_plans, np.zeros((len(joint_plans), 2))), axis=-1)

    elif mode == "smooth":
        joint_plans = []
        time_traj = np.arange(0.0, total_time * 7, dt)
        for joint_idx in range(7):
            p = sympy.prime(prime_start_idx + joint_idx) % freq_divident
            freq = p / freq_scaling

            init_joint = joint_init[joint_idx]
            # I'm sorry I couldn't use list comp :(
            sin_wave = []
            for time in time_traj:
                sin_wave.append(amp * np.sin(2 * np.pi * np.sin(0.01 * time) * freq * time) + init_joint)

            joint_plans.append(sin_wave)

        joint_plans = np.array(joint_plans).T
        joint_plans = np.concatenate((joint_plans, np.zeros((len(joint_plans), 2))), axis=-1)

    elif mode == "single":
        # reduce frequencies
        freqs = [
            (sympy.prime(prime_start_idx + joint_idx) % freq_divident) / freq_scaling for joint_idx in range(7)
        ]  # scale by 1/100 before
        time_traj = np.arange(0.0, total_time, dt)

        for joint_idx in range(7):
            # for each joint
            freq = freqs[joint_idx]  # sympy.prime(prime_start_idx + joint_idx) / 13  # downscale
            init_joint = joint_init[joint_idx]  #  - amp * np.sin(2 * np.pi * freq * 0.25)
            sin_wave = amp * np.sin(2 * np.pi * freq * time_traj) + init_joint
            joint_plans.append(sin_wave)

        joint_plans = np.array(joint_plans).T
        joint_plans = np.concatenate((joint_plans, np.zeros((len(joint_plans), 2))), axis=-1)

    if True:
        plot_traj(joint_plans, time_traj)

    # n x 9
    return time_traj, joint_plans


def plot_traj(joint_traj, time_traj):
    plt.figure(figsize=(15, 20))
    for joint_idx in range(7):
        plt.plot(time_traj, joint_traj[:, joint_idx], label=f"joint_{joint_idx}")
    plt.legend()
    plt.show()


def extract_waypoint(
    traj: PathParameterizedTrajectory,
    vel_traj: PathParameterizedTrajectory,
    time_path: Iterable[float],
    acc_traj: Optional[PathParameterizedTrajectory] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    position_path = np.array([traj.value(t) for t in time_path])
    velocity_path = np.array([vel_traj.value(t) for t in time_path])
    if acc_traj:
        acceleration_path = np.array([acc_traj.value(t) for t in time_path])
        return position_path, velocity_path, acceleration_path
    return position_path, velocity_path


# def ros_transform_to_numpy_pose(transform: Transform) -> np.ndarray:
#     """Converts a ROS Transform message into a homogenous numpy pose of shape (4,4)."""
#     translation = [transform.translation.x, transform.translation.y, transform.translation.z]
#     quaternion = [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
#     pose_np = np.eye(4)
#     pose_np[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
#     pose_np[:3, 3] = translation
#     return pose_np