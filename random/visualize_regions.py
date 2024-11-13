from planning.iris_and_gcs_utils import *

import numpy as np
import os
from IPython.display import clear_output
from scipy.spatial.transform import Rotation as R
from pydrake.all import (
    AbstractValue,
    AddMultibodyPlantSceneGraph,
    Concatenate,
    DiagramBuilder,
    JointSliders,
    LeafSystem,
    MeshcatPoseSliders,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    PointCloud,
    RandomGenerator,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    UniformlyRandomRotationMatrix,
    RollPitchYaw
)
from pydrake.geometry.optimization import (
    HPolyhedron,
    IrisInConfigurationSpace,
    IrisOptions,
    LoadIrisRegionsYamlFile,
    SaveIrisRegionsYamlFile,
)

from manipulation import running_as_notebook
from manipulation.scenarios import AddFloatingRpyJoint, AddRgbdSensors, ycb
from manipulation.utils import ConfigureParser
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.station import MakeHardwareStation, LoadScenario

# Start the visualizer.
meshcat = StartMeshcat()

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0005)
parser = Parser(plant)
ConfigureParser(parser)
parser.AddModelsFromUrl("file://./home/evelyn/sources/Real2SimObjectManipulation/scenario_datas/scenario_data_grasping.dmd.yaml")
plant.Finalize()

params = MeshcatVisualizerParams()
params.prefix = "planning"
visualizer = MeshcatVisualizer.AddToBuilder(
    builder, scene_graph, meshcat, params
)
diagram = builder.Build()

context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)

diagram.ForcedPublish(context)

iris_regions = LoadIrisRegionsYamlFile("../regions/gaze_constrained_scanning_regions_3.yaml")
visualize_connectivity(iris_regions)

test_iris_region(plant, plant_context, meshcat, iris_regions, num_sample=100)

while (1):
    pass