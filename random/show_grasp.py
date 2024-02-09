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
plant.Finalize()

params = MeshcatVisualizerParams()
params.prefix = "planning"
visualizer = MeshcatVisualizer.AddToBuilder(
    builder, scene_graph, meshcat, params
)
diagram = builder.Build()

X_G = RigidTransform(
        R=RotationMatrix([
            [0.584195116402473, 0.6687940071350957, -0.4598158783596781],
            [-0.5257346834908867, -0.11978661260019913, -0.8421723161066901],
            [-0.6183195844757288, 0.7337341095123974, 0.2816294515703403],
        ]),
        p=[0.5598079847188085, 0.07715878160574284, 0.15917199927752262],
      )

X_mustard = RigidTransform(RotationMatrix(RollPitchYaw(-np.pi/2, 0, np.pi/2)), [0.5, 0.0, 0.08])
# X_WGfix = RigidTransform(RotationMatrix(RollPitchYaw(np.pi/2, 0, 0))).multiply(RigidTransform([0, -0.049133, 0]))
X_WGfix = RigidTransform(RotationMatrix(RollPitchYaw(np.pi/2, 0, np.pi/2)))
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body"), X_G.multiply(X_WGfix))
# plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body"), X_G)
# plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body"), X_Fp)
# plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body"), X_mustard.multiply(X_WGfix))
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base_link_mustard"), X_mustard)
diagram.ForcedPublish(context)

while (1):
    pass