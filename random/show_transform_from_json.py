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

    transform_path = argv[0]

    # Load JSON data
    with open(transform_path, 'r') as f:
        data = json.load(f)

    i = 0
    # Iterate over each frame and print the transform_matrix as a numpy array
    for frame in data['frames']:
        T = np.array(frame['transform_matrix'])
        AddMeshcatTriad(meshcat, f'camera_frame{i}', X_PT=RigidTransform(RotationMatrix(T[:3, :3]),
                                T[:3, -1]))
        i += 1

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    X_cam = RigidTransform(RotationMatrix(), [0, 0, 0])
    plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base"), X_cam)

    diagram.ForcedPublish(context)

    i = 0
    while (1):
        T = np.array(data['frames'][i % len(data['frames'])]['transform_matrix'])
        X_mustard = RigidTransform(RotationMatrix(T[:3, :3]), T[:3, -1])
        plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base_link_mustard"), X_mustard)
        diagram.ForcedPublish(context)
        time.sleep(0.05)
        i += 1

if __name__ == "__main__":
   main(sys.argv[1:]) # arguments: [data directory]