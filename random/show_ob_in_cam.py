import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
    StartMeshcat,
)
import json
import sys
import os
import time

from manipulation.utils import ConfigureParser
from manipulation.meshcat_utils import AddMeshcatTriad

def main(argv):
    # Start the visualizer.
    meshcat = StartMeshcat()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0005)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl("file://./home/evelyn/sources/Real2SimObjectManipulation/models/006_mustard_bottle.sdf")
    parser.AddModelsFromUrl("package://manipulation/camera_box.sdf")
    plant.Finalize()

    params = MeshcatVisualizerParams()
    params.prefix = "planning"
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, params
    )
    diagram = builder.Build()

    ob_in_cam_path = argv[0]
    i = 0
    for transform_path in sorted(os.listdir(ob_in_cam_path)):
        T = np.loadtxt(os.path.join(ob_in_cam_path, transform_path))
        T = np.linalg.inv(T)
        AddMeshcatTriad(meshcat, f'camera_frame{i}', X_PT=RigidTransform(RotationMatrix(T[:3, :3]),
                                T[:3, -1]))
        i += 1
    print(i)

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    X_cam = RigidTransform(RotationMatrix(), [0, 0, 0])
    plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base"), X_cam)

    diagram.ForcedPublish(context)

    transform_paths = sorted(os.listdir(ob_in_cam_path))
    i = 0
    while (1):
        T = np.loadtxt(os.path.join(ob_in_cam_path, transform_paths[i % len(transform_paths)]))
        X_mustard = RigidTransform(RotationMatrix(T[:3, :3]), T[:3, -1])
        plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base_link_mustard"), X_mustard)
        diagram.ForcedPublish(context)
        time.sleep(0.05)
        i += 1

if __name__ == "__main__":
   main(sys.argv[1:]) # arguments: [data directory]