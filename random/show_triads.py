import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    StartMeshcat,
)
import json
import sys
import os

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
    plant.Finalize()

    params = MeshcatVisualizerParams()
    params.prefix = "planning"
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, params
    )
    diagram = builder.Build()

    transforms = [
        RigidTransform(RollPitchYaw(-np.pi/2, 0, 0), [0.4, 0, 0.7]),
        RigidTransform(RollPitchYaw(-5*np.pi/6, 0, np.pi/4), [0.1, 0.3, 0.4]),
        RigidTransform(RollPitchYaw(np.pi/6, np.pi, -np.pi/4), [0.1, -0.3, 0.4]),
        RigidTransform(RollPitchYaw(-5*np.pi/6, 0, 0), [0.4, 0.4, 0.4]),
        RigidTransform(RollPitchYaw(np.pi/6, np.pi, 0), [0.4, -0.4, 0.4]),
        RigidTransform(RollPitchYaw(-5*np.pi/6, 0, -np.pi/8), [0.5, 0.35, 0.4]),
        RigidTransform(RollPitchYaw(np.pi/6, np.pi, np.pi/10), [0.5, -0.35, 0.4]),
        RigidTransform(RollPitchYaw(-2*np.pi/3, 0, 0), [0.4, 0.25, 0.55]),
        RigidTransform(RollPitchYaw(-1*np.pi/3, 0, 0), [0.4, -0.25, 0.55]),
    ]
    i = 0
    for T in transforms:
        AddMeshcatTriad(meshcat, f'triad_{i}', X_PT=T)
        i += 1

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    X_mustard = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 110/180*np.pi)), [0.4, 0.0, 0.07])
    plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base_link_mustard"), X_mustard)

    diagram.ForcedPublish(context)

    while (1):
        pass

if __name__ == "__main__":
   main(sys.argv[1:]) # arguments: [data directory]