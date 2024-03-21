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
parser.AddModelsFromUrl("package://manipulation/schunk_wsg_50_welded_fingers.sdf")
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
    [-0.07695407022646775, -0.8083822133486049, 0.583606261290256],
    [-0.9078308609565966, 0.29881542961254354, 0.2941980063159417],
    [-0.41221499120044597, -0.50717604060827, -0.7568693842814194],
  ]),
  p=[0.46679165812005635, -0.0360518455274682, 0.3533314571063431],
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

X_mustard = RigidTransform(RotationMatrix(RollPitchYaw(-np.pi/2, 0, np.pi/2)), [0.57, 0.0, 0.12])
X_WGfix = RigidTransform(RotationMatrix(RollPitchYaw(np.pi/2, 0, np.pi/2)))
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body"), X_G.multiply(X_WGfix))
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base_link_mustard"), X_mustard)

diagram.ForcedPublish(context)

while (1):
    pass