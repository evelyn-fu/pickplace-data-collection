import numpy as np
import argparse
import json
import time
from tqdm import tqdm
from planning.misc.sdf_tools import SignedDensityField
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Parser,
)
from manipulation.utils import ConfigureParser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_url",
        default="file://./home/real2sim/src/Real2SimObjectManipulation/models/schunk_wsg_50_welded_fingers_w_buffer.sdf",
        help="url to object model file",
        nargs='?',
    )
    parser.add_argument(
        "--pkl_file",
        default="gripper_sdf.pkl",
        help="pkl file path to save sdf to",
        nargs='?',
    )
    parser.add_argument(
        "--x_min",
        default="-0.1",
        help="minimum x point to check for sdf",
        nargs='?',
    )
    parser.add_argument(
        "--x_max",
        default="0.1",
        help="maximum x point to check for sdf",
        nargs='?',
    )
    parser.add_argument(
        "--y_min",
        default="-0.1",
        help="minimum y point to check for sdf",
        nargs='?',
    )
    parser.add_argument(
        "--y_max",
        default="0.30",
        help="maximum y point to check for sdf",
        nargs='?',
    )
    parser.add_argument(
        "--z_min",
        default="-0.05",
        help="minimum z point to check for sdf",
        nargs='?',
    )
    parser.add_argument(
        "--z_max",
        default="0.05",
        help="maximum z point to check for sdf",
        nargs='?',
    )
    parser.add_argument(
        "--delta",
        default="0.001",
        help="maximum z point to check for sdf",
        nargs='?',
    )
    args = parser.parse_args()
    x_min = float(args.x_min)
    x_max = float(args.x_max)
    y_min = float(args.y_min)
    y_max = float(args.y_max)
    z_min = float(args.z_min)
    z_max = float(args.z_max)
    delta = float(args.delta)

    start = time.time()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0005)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl(args.model_url)
    plant.Finalize()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    scene_graph_context = scene_graph.GetMyContextFromRoot(context)
    
    query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)

    grid_size_x = int((x_max - x_min) / delta)
    grid_size_y = int((y_max - y_min) / delta)
    grid_size_z = int((z_max - z_min) / delta)

    # Create a 3D grid within the specified bounds
    x = np.linspace(x_min, x_max, grid_size_x)
    y = np.linspace(y_min, y_max, grid_size_y)
    z = np.linspace(z_min, z_max, grid_size_z)
    sdf_grid = np.ones((grid_size_x, grid_size_y, grid_size_z)) * np.inf
    
    for i in tqdm(range(grid_size_x)):
        for j in range(grid_size_y):
            for k in range(grid_size_z):
                pt = np.array([x[i], y[j], z[k]])
                distances = query_object.ComputeSignedDistanceToPoint(pt)
                for body_index in range(len(distances)):
                    distance = distances[body_index].distance
                    if distance < sdf_grid[i, j, k]:
                        sdf_grid[i, j, k] = distance

    print(np.min(sdf_grid))
    origin = np.array([x_min, y_min, z_min])
    sdf = SignedDensityField(sdf_grid, origin, delta)
    print("SDF computation time", time.time()-start)

    sdf.dump(args.pkl_file)
    # Query the SDF at an arbitrary point
    point = np.array([0.0, -0.049133, 0.025])
    start = time.time()
    distance_at_point = sdf.get_distance(point)
    print("SDF lookup time", time.time()-start)
    print(f"Signed distance at point {point}: {distance_at_point}")

    # sdf.visualize()
    # sdf.visualize_matplotlib()