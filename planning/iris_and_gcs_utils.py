from pydrake.all import (
    RigidTransform,
    LeafSystem,
    AbstractValue,
    QuaternionFloatingJoint,
    AbstractValue,
    InverseKinematics,
    Solve,
    BasicVector,
    PointCloud,
    Rgba,
    RandomGenerator,
    Point,
    logical_and,
    Trajectory,
    CompositeTrajectory,
    RotationMatrix,
)
from manipulation.meshcat_utils import AddMeshcatTriad

import time
import numpy as np
import logging
from scipy.spatial.transform import Rotation as R
import typing as T
import pickle

def load_data_for_trajectory(traj_num: int):
    """
    pass the trajectry index. 
    0th trajectory moves arm with object from grasp of object 0 to deposit of 0
    1st trajectory moves arm without object from deposit of object 0 to grasp of 1
    2nd trajectory moves arm with object from grasp of object 1 to deposit of 1
    3rd trajectory moves arm without object from deposit of object 1 to grasp of 2
    etc
    """
    file_name = "../shortest_walks_trajectories/traj_" + str(traj_num) + ".pkl"
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def get_trajectory_length(data):
    """
    tells you how long trajectory takes
    """
    return data["time_trajectory"][-1]

def get_pos_vel_acc_jerk(data, t:float):
    """
    pass the data (output of load_data_for_trajectory function)
    and the time since the beginning of the current trajectory.

    NOTE: not the time since start of execution, but time since beginning of this specific trajectory!
    
    returns pos, vel, acc, jerk
    """
    if t < 0:
        p = data["position_trajectory"][0]
        v = data["velocity_trajectory"][0]
        a = data["acceleration_trajectory"][0]
        j = np.zeros(6)
    elif t >= data["time_trajectory"][-1]:
        p = data["position_trajectory"][-1]
        v = data["velocity_trajectory"][-1]
        a = data["acceleration_trajectory"][-1]
        j = np.zeros(6)
    else:
        index = None
        for i, tstep in enumerate(data["time_trajectory"]):
            if tstep > t:
                index = i-1
                break
        dt = t - data["time_trajectory"][index]
        j = data["jerk_trajectory"][index]
        p = data["position_trajectory"][index] + data["velocity_trajectory"][index] * dt +  data["acceleration_trajectory"][index] * dt ** 2 / 2 + j*dt**3/6
        v = data["velocity_trajectory"][index] + data["acceleration_trajectory"][index] * dt +  j * dt ** 2 / 2
        a = data["acceleration_trajectory"][index] + j * dt
    return p,v,a,j

def test_iris_region(plant, plant_context, meshcat, regions, seed=42, num_sample=10000, colors=None):
    """
    Plot small spheres in the volume of each region. (we are using forward
    kinematics to return from configuration space to task space.)

    regions is a list of ConvexSets.
    """
    world_frame = plant.world_frame()
    ee_frame = plant.GetFrameByName("iiwa_link_7")

    rng = RandomGenerator(seed)

    # Allow caller to input custom colors
    if colors is None:
        colors = [
            Rgba(0.5,0.0,0.0,0.5),
            Rgba(0.0,0.5,0.0,0.5),
            Rgba(0.0,0.0,0.5,0.5),
            Rgba(0.5,0.5,0.0,0.5),
            Rgba(0.5,0.0,0.5,0.5),
            Rgba(0.0,0.5,0.5,0.5),
            Rgba(0.2,0.2,0.2,0.5),
            Rgba(0.5,0.2,0.0,0.5),
            Rgba(0.2,0.5,0.0,0.5),
            Rgba(0.5,0.0,0.2,0.5),
            Rgba(0.2,0.0,0.5,0.5),
            Rgba(0.0,0.5,0.2,0.5),
            Rgba(0.0,0.2,0.5,0.5),
        ]

    i = 0
    for name in regions:
        region = regions[name]

        xyzs = []  # List to hold XYZ positions of configurations in the IRIS region
        
        q_sample = region.UniformSample(rng)
        prev_sample = q_sample

        plant.SetPositions(plant_context, q_sample)
        T = plant.CalcRelativeTransform(plant_context, frame_A=world_frame, frame_B=ee_frame)
        xyzs.append(T.translation())
        AddMeshcatTriad(meshcat, f'triad_{i}', X_PT=T)
        i += 1

        for _ in range(num_sample-1):
            q_sample = region.UniformSample(rng, prev_sample)
            prev_sample = q_sample

            plant.SetPositions(plant_context, q_sample)
            T = plant.CalcRelativeTransform(plant_context, frame_A=world_frame, frame_B=ee_frame)
            xyzs.append(T.translation())
            AddMeshcatTriad(meshcat, f'triad_{i}', X_PT=T)
            i += 1

        # Create pointcloud from sampled point in IRIS region in order to plot in Meshcat
        # xyzs = np.array(xyzs)
        # pc = PointCloud(len(xyzs))
        # pc.mutable_xyzs()[:] = xyzs.T
        # meshcat.SetObject(f"visuals/regions/region {name}", pc, point_size=0.025, rgba=colors[i % len(colors)])
        # i += 1

def visualize_connectivity(iris_regions, output_file='../iris_connectivity.svg', skip_svg=False):
    """
    Create and save SVG graph of IRIS Region connectivity.

    iris_regions can be a list of ConvexSets or a dictionary with keys as
    labels and values as ConvexSets.
    """
    import pydot

    numEdges = 0
    numNodes = 0

    graph = pydot.Dot("IRIS region connectivity")

    if isinstance(iris_regions, dict):
        items = list(iris_regions.items())
    else:
        items = list(enumerate(iris_regions))

    for i, (label1, v1) in enumerate(items):
        numNodes += 1
        graph.add_node(pydot.Node(label1))
        for j in range(i + 1, len(items)):
            label2, v2 = items[j]
            if v1.IntersectsWith(v2):
                numEdges += 1
                graph.add_edge(pydot.Edge(label1, label2, dir="both"))

    # Add text annotations for numNodes and numEdges
    annotation = f"Nodes: {numNodes}, Edges: {numEdges}"
    graph.add_node(pydot.Node("annotation", label=annotation, shape="none", fontsize="12", pos="0,-1!", margin="0"))

    if not skip_svg:
        svg = graph.create_svg()

        with open(output_file, 'wb') as svg_file:
            svg_file.write(svg)

    return numNodes, numEdges


def VisualizePath(meshcat, plant, plant_context, traj, name):
    """
    Helper function that takes in trajopt basis and control points of Bspline
    and draws spline in meshcat.

    traj can be either a Drake Trajectory object or a dictionary (in the format
    Savva made).
    """
    if isinstance(traj, Trajectory):
        traj_start_time = traj.start_time()
        traj_end_time = traj.end_time()

        def get_traj_pos(vis_t):
            return traj.value(vis_t)
   
    else:
        traj_start_time = 0
        traj_end_time = get_trajectory_length(traj)

        def get_traj_pos(vis_t):
            return get_pos_vel_acc_jerk(traj, vis_t)[0]

    # Build matrix of 3d positions by doing forward kinematics at time steps in the bspline
    NUM_STEPS = 80
    pos_3d_matrix = np.zeros((3,NUM_STEPS))
    ctr = 0
    for vis_t in np.linspace(traj_start_time, traj_end_time, NUM_STEPS):
        pos = get_traj_pos(vis_t)
        plant.SetPositions(plant_context, pos)
        pos_3d = plant.CalcRelativeTransform(plant_context, plant.world_frame(), plant.GetFrameByName("iiwa_link_7")).translation()
        pos_3d_matrix[:,ctr] = pos_3d
        ctr += 1

    meshcat.SetLine(name, pos_3d_matrix)