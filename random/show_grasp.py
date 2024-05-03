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
parser.AddModelsFromUrl("file://./home/evelyn/sources/Real2SimObjectManipulation/models/box_2.sdf")
parser.AddModelsFromUrl("file://./home/evelyn/sources/Real2SimObjectManipulation/models/box.sdf")
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

# X_G = RigidTransform(
#   R=RotationMatrix([
#     [0.20681969377898063, 0.9685467022240954, 0.1383578688618696],
#     [0.9774142324949082, -0.21081815591263353, 0.014735102781668053],
#     [0.043439985975579215, 0.13218544075814884, -0.9902726780387395],
#   ]),
#   p=[0.5798396344223478, -0.0008122595958626092, 0.3250179079887979],
# )

# X_G = RigidTransform(
#   R=RotationMatrix([
#     [0.02780691805464589, -0.3672344941274829, 0.9297126446549144],
#     [-0.9871632429272026, 0.13624557061353873, 0.08334192403370885],
#     [-0.1572752590897611, -0.9200956313400926, -0.3587318247202575],
#   ]),
#   p=[0.5783754957459597, -0.0475038303546954, 0.40298295222281433],
# ) # q = [0, 0.7, -0.2, -1.1, 0.2, 1.75, 2.9]

# X_G = RigidTransform(
#   R=RotationMatrix([
#     [0.7241199295848425, 0.5971874186123144, -0.3449891514659612],
#     [0.6834126756784046, -0.6885730425216102, 0.24251655579489312],
#     [-0.09272239373596361, -0.41138103038873125, -0.90673491470009],
#   ]),
#   p=[0.07129913273973337, 0.3783192220042352, 0.2677729463033564],
# ) # q = [2.5, 1.3, 1.8, 1.8, 0.2, -1.5, -1.4]

X_G = RigidTransform(
  R=RotationMatrix([
    [-0.7536161473974042, 0.59819944231936, -0.2724337159580315],
    [0.65730679904535, 0.6878723749385145, -0.30786257928693544],
    [0.003236403968931164, -0.4110827447184517, -0.9115922897239176],
  ]),
  p=[0.07046840429505286, -0.3783725206382699, 0.2672424814188106],
) # q = [0.64, -1.3, 1.34, 1.8, -0.2, -1.5, 1.3]

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

X_mustard = RigidTransform(RotationMatrix(RollPitchYaw(-np.pi/2, 0, np.pi/2)), [0.4, 0.0, 0.16])
X_WGfix = RigidTransform(RotationMatrix(RollPitchYaw(np.pi/2, 0, np.pi/2)))
rot_180 = RigidTransform(RotationMatrix(RollPitchYaw(0, np.pi, 0)))
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body"), X_G)
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base_link_mustard"), X_mustard)
X_box2 = RigidTransform(RotationMatrix(), [0.4, 0.0, 0.07])
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("box2_body"), X_box2)
X_box = RigidTransform(RotationMatrix(), [0.4, 0.0, 0.0])
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("box_body"), X_box)

diagram.ForcedPublish(context)

while (1):
    pass