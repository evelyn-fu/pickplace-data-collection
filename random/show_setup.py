import numpy as np
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
parser.AddModelsFromUrl("file://./home/real2sim/src/Real2SimObjectManipulation/scenario_datas/scenario_data_grasping_hardware.dmd.yaml")
plant.Finalize()

params = MeshcatVisualizerParams()
params.prefix = "planning"
visualizer = MeshcatVisualizer.AddToBuilder(
    builder, scene_graph, meshcat, params
)
diagram = builder.Build()

context = diagram.CreateDefaultContext()

diagram.ForcedPublish(context)

X_WC1 = RigidTransform(
  R=RotationMatrix([
    [-0.02329211172783142, -0.03397455108592995, -0.9991512435110952],
    [-0.012515200476671077, -0.9993341602581275, 0.03427252395450102],
    [-0.9996503625209601, 0.013302857576199574, 0.022851404596225578],
  ]),
  p=[0.864887, -0.00272318, 0.311732],
)
# AddMeshcatTriad(meshcat, "X_WC1", 
#                 X_PT=X_WC1)


r = R.from_quat([-0.698783, 0.0166324, 0.715101, 0.00750221])
x_world_rgb_camera0 = RigidTransform(
    R=RotationMatrix(r.as_matrix()),
    p = [0.864887, -0.00272318, 0.311732]
)
x_rgb_depth_camera0 = RigidTransform([  [0.999986,  0.000116105 , -0.00531402 ,  -0.0151041],
                                        [-0.000127587 ,    0.999998 , -0.00216038 ,-6.34105e-05],
                                        [0.00531376  , 0.00216102  ,   0.999984 ,  0.00034625],
                                                [0  ,          0  ,          0 ,           1]])
x_depth_rgb_camera0 = RigidTransform([[0.999986, -0.000127587,   0.00531376,     0.015102],
                                      [0.000116105,     0.999998 ,  0.00216102,  6.44158e-05],
                                      [-0.00531402,  -0.00216038,     0.999984, -0.000426644],
                                      [          0,            0,            0,            1]])
x_world_camera0 = x_world_rgb_camera0 @ x_depth_rgb_camera0

AddMeshcatTriad(meshcat, "x_world_rgb_camera0", 
                X_PT=x_world_rgb_camera0)
AddMeshcatTriad(meshcat, "x_world_depth_camera0", 
                X_PT=x_world_camera0)

# X_WC2 = RigidTransform(
#   R=RotationMatrix([
#     [0.014309915141510702, 0.8998166930801654, -0.4360334220939022],
#     [0.9902520814745662, 0.047671313024411986, 0.1308749825207932],
#     [0.13854977973252838, -0.4336558137149843, -0.8903631808241308],
#   ]),
#   p=[0.48208603917586756, -0.07049197623383739, 0.34842124090244536],
# )
# AddMeshcatTriad(meshcat, "X_WC2", 
#                 X_PT=X_WC2)

# X_WC3 = RigidTransform(
#   R=RotationMatrix([
#     [-0.716745137750116, -0.33859635017944, 0.609613745872524],
#     [-0.6867564727469487, 0.1910464833816359, -0.7013321526407308],
#     [0.1210039447775892, -0.9213325962065345, -0.3694648731822687],
#   ]),
#   p=[0.15246059550687813, 0.3287099366902628, 0.29252629368871186],
# )
# AddMeshcatTriad(meshcat, "X_WC3", 
#                 X_PT=X_WC3)

# X_WC4 = RigidTransform(
#   R=RotationMatrix([
#     [0.7462561424589804, -0.26036064576435, 0.6126288468391063],
#     [-0.66524114928318, -0.2591011075438441, 0.7002291263364783],
#     [-0.023579294786651654, -0.9300962048125586, -0.36655840823890545],
#   ]),
#   p=[0.1051619426674925, -0.29081861410529236, 0.2963461446071302],
# )
# AddMeshcatTriad(meshcat, "X_WC4", 
#                 X_PT=X_WC4)

while (1):
    pass