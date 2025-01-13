import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    RigidTransform,
    RotationMatrix,
    StartMeshcat,
)
import json
import sys

from manipulation.utils import ConfigureParser
from manipulation.meshcat_utils import AddMeshcatTriad

def main(argv):
    # Start the visualizer.
    meshcat = StartMeshcat()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0005)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://drake_models/ycb/006_mustard_bottle.sdf")
    plant.Finalize()

    params = MeshcatVisualizerParams()
    params.prefix = "planning"
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, params
    )
    diagram = builder.Build()

    transforms_path = argv[0]
    with open(transforms_path) as f:
        data = json.load(f)
        for i in range(len(data["frames"])):
            frame = data["frames"][i]
            T = np.array(frame["transform_matrix"])
            # align grasp axis is blue, align minor axis is red
            AddMeshcatTriad(meshcat, f'camera_frame{i}', X_PT=RigidTransform(RotationMatrix(T[:3, :3]),
                                    T[:3, -1]))

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    X_mustard = RigidTransform(RotationMatrix(), [0, 0, 0])
    plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base_link_mustard"), X_mustard)

    diagram.ForcedPublish(context)

    while (1):
        pass

if __name__ == "__main__":
   main(sys.argv[1:]) # arguments: [data directory]