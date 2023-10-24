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

# X_Fp = RigidTransform(
#   R=RotationMatrix([
#     [0.32799985812900345, 0.30235130083281553, 0.8949859127115121],
#     [-0.30379968816918834, -0.8633108750183713, 0.40298893600608854],
#     [0.8944953004547637, -0.40407675503512863, -0.191311613616694],
#   ]),
#   p=[-0.043831818161671816, -0.623690601915065, 0.19470715975149566],
# )

# AddMeshcatTriad(meshcat, "darboux frame", length=0.25, radius=0.005, opacity=1.0, X_PT=X_Fp)

# X_G = RigidTransform(
#   R=RotationMatrix([
#     [0.01813626879178622, 0.6562873721637131, -0.7542930205780511],
#     [0.3956341543707907, 0.6881310724227954, 0.6082345296524723],
#     [0.9182291062962565, -0.30945518626240964, -0.24716956941677873],
#   ]),
#   p=[-0.07966969977818564, -0.649464046786021, 0.21422068661496743],
# )

X_G = RigidTransform(
  R=RotationMatrix([
    [-0.9869950008883301, 0.09192791461120958, -0.13187162976424666],
    [0.09596994332526952, 0.9950797897951107, -0.024616700011639597],
    [0.1289598317290659, -0.03695227268454138, -0.9909610947680326],
  ]),
  p=[0.008002690309663874, -0.6074651876896352, 0.23267431123246],
)

X_mustard = RigidTransform(RotationMatrix(RollPitchYaw(-np.pi/2, 0, 0)), [0, -0.6, 0.1])
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