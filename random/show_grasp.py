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
parser.AddModelsFromUrl("file://./home/evelyn/sources/Real2SimObjectManipulation/models/schunk_wsg_50_welded_fingers_w_buffer.sdf")
parser.AddModelsFromUrl("package://drake/manipulation/models/ycb/sdf/006_mustard_bottle.sdf")
plant.Finalize()

params = MeshcatVisualizerParams()
params.prefix = "planning"
visualizer = MeshcatVisualizer.AddToBuilder(
    builder, scene_graph, meshcat, params
)
diagram = builder.Build()

X_GgraspGpregrasp = RigidTransform([0, 0.0, -0.15])

# X_G = RigidTransform(
#   R=RotationMatrix([
#     [-0.32800086091826547, -0.9327063454550955, 0.14991433682165606],
#     [0.036292819708348835, 0.14613500088627066, 0.9885987015738928],
#     [-0.9439800138109357, 0.32970203921688135, -0.014081862864553518],
#   ]),
#   p=[0.5287979356327239, -0.09737849855328161, 0.23556644990216116],
# )

X_G = RigidTransform(
  R=RotationMatrix([
    [0.5295776198751465, 0.846339745369453, -0.05706645192532622],
    [0.8477111514205422, -0.5304580047265671, -0.0003301193244864655],
    [-0.030550749330160602, -0.04820104386296692, -0.9983703276269217],
  ]),
  p=[0.5941796468903248, 0.01189624213382719, 0.3234708434220175],
)

rot = X_G.GetAsMatrix4()[:3, :3]
t = X_G.GetAsMatrix4()[:3, 3]
eff_vertical_vec = rot.dot(np.array([0, 0, 1]))
eff_horizontal_vec = rot.dot(np.array([0, 1, 0]))
z_axis, x_axis = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
rot_to_axes, _ = R.align_vectors(
    np.array([z_axis, x_axis]), np.stack([eff_vertical_vec, eff_horizontal_vec])
)

# align eff_vertical_vec is blue, eff_horizontal_vec is red
AddMeshcatTriad(meshcat, "eff_vec", X_PT=RigidTransform(RotationMatrix(rot_to_axes.as_matrix().T),
                        t))

align_grasp_axis = [4.396e-02, 9.990e-01, 4.446e-04]
align_minor_axis = [-0.999,  0.044 , 0.002]
z_axis, x_axis = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
rot_to_axes, _ = R.align_vectors(
    np.array([z_axis, x_axis]), np.stack([align_grasp_axis, align_minor_axis])
)
# align grasp axis is blue, align minor axis is red
AddMeshcatTriad(meshcat, "align_grasp_axis", X_PT=RigidTransform(RotationMatrix(rot_to_axes.as_matrix().T),
                        [0.6, 0.0, 0.12]))

gripper_axis_alignment_cost = -np.abs(eff_vertical_vec @ align_grasp_axis)  # want vertical axis of gripper to face towards desired axis, larger worse
gripper_minor_alignment_cost = -np.abs(eff_horizontal_vec @ align_minor_axis)

print(gripper_axis_alignment_cost, gripper_minor_alignment_cost)

X_mustard = RigidTransform(RotationMatrix(RollPitchYaw(-np.pi/2, 0, np.pi/2)), [0.57, 0.0, 0.16])
X_WGfix = RigidTransform(RotationMatrix(RollPitchYaw(np.pi/2, 0, np.pi/2)))
rot_180 = RigidTransform(RotationMatrix(RollPitchYaw(0, np.pi, 0)))
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body"), X_G.multiply(X_WGfix) @ rot_180)
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base_link_mustard"), X_mustard)

diagram.ForcedPublish(context)

while (1):
    pass