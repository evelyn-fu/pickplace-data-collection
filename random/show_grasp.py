import numpy as np
from IPython.display import clear_output
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

from manipulation import running_as_notebook
from manipulation.scenarios import AddFloatingRpyJoint, AddRgbdSensors, ycb
from manipulation.utils import ConfigureParser
from manipulation.meshcat_utils import AddMeshcatTriad

# Start the visualizer.
meshcat = StartMeshcat()

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0005)
parser = Parser(plant)
ConfigureParser(parser)
parser.AddModelsFromUrl("package://manipulation/schunk_wsg_50_welded_fingers.sdf")
parser.AddModelsFromUrl("package://drake/manipulation/models/ycb/sdf/006_mustard_bottle.sdf")
parser.AddModelsFromUrl("package://drake/manipulation/models/iiwa_description/iiwa7/iiwa7_no_collision.sdf")
plant.Finalize()

params = MeshcatVisualizerParams()
params.prefix = "planning"
visualizer = MeshcatVisualizer.AddToBuilder(
    builder, scene_graph, meshcat, params
)
diagram = builder.Build()

X_G = RigidTransform(
  R=RotationMatrix([
    [-0.024862600566131942, 0.6118970944790467, 0.7905465178350922],
    [-0.2599776962763316, 0.7596246248145511, -0.5961392679686555],
    [-0.9652944879838184, -0.22034603500733402, 0.1401933533538813],
  ]),
  p=[0.568112593281828, 0.0371661061349634, 0.17595490100665073],
)

X_mustard = RigidTransform(RotationMatrix(RollPitchYaw(-np.pi/2, 0, np.pi/2)), [0.6, 0.0, 0.16])
X_WGfix = RigidTransform(RotationMatrix(RollPitchYaw(np.pi/2, 0, np.pi/2)))
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body"), X_G.multiply(X_WGfix))
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base_link_mustard"), X_mustard)

X_iiwa_base = RigidTransform(RotationMatrix())
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("iiwa_link_0"), X_iiwa_base)
new_positions = plant.GetPositions(plant_context)
print(new_positions)
new_positions[-7:] = [ 0.723 , 1.531 ,-0.257, -0.88 , -1.148 , 1.894,  2.533]
plant.SetPositions(plant_context, new_positions)
print(new_positions)

diagram.ForcedPublish(context)

while (1):
    pass