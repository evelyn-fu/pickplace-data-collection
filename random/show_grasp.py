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
                [0.3107252766435573, 0.9478457803191421, -0.07098013233280487],
                [0.9501638206650431, -0.3117334204461474, -0.003314887050161988],
                [-0.025268881138556104, -0.0664127345329445, -0.9974722213364458],
            ]),
            p=[0.6302492295015113, -0.0074317699693182675, 0.2834767828082438],
            )

X_mustard = RigidTransform(RotationMatrix(RollPitchYaw(-np.pi/2, 0, np.pi/2)), [0.5, 0.0, 0.08])
X_WGfix = RigidTransform(RotationMatrix(RollPitchYaw(np.pi/2, 0, np.pi/2)))
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body"), X_G.multiply(X_WGfix))
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base_link_mustard"), X_mustard)

X_iiwa_base = RigidTransform(RotationMatrix())
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("iiwa_link_0"), X_iiwa_base)
new_positions = plant.GetPositions(plant_context)
print(new_positions)
new_positions[-7:] = [ 0.034 , 0.782 ,-0.077, -1.232 , 0.053 , 1.201 , 1.845]
plant.SetPositions(plant_context, new_positions)
print(new_positions)

diagram.ForcedPublish(context)

while (1):
    pass