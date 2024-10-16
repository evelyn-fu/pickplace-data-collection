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
from planning.inverse_kinematics import solve_global_inverse_kinematics

# Start the visualizer.
meshcat = StartMeshcat()

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0005)
parser = Parser(plant)
ConfigureParser(parser)
parser.AddModelsFromUrl("package://manipulation/iiwa_and_wsg.dmd.yaml")
parser.AddModelsFromUrl("package://drake_models/ycb/006_mustard_bottle.sdf")


object_area_urdf = """<?xml version="1.0"?>
<robot name="object_area">
<link name="object_area">
<visual name="object_area">
    <origin rpy="0 0 0" xyz="0.4 0.0 0.15"/>
    <geometry>
    <box size="0.4 0.4 0.16"/>
    </geometry>
</visual>
<collision name="object_area">
    <origin rpy="0 0 0" xyz="0.4 0.0 0.15"/>
    <geometry>
    <box size="0.4 0.4 0.16"/>
    </geometry>
</collision>
</link>
<joint name="fixed_link_weld" type="fixed">
<parent link="world"/>
<child link="object_area"/>
</joint>
</robot>
"""
parser.AddModelsFromString(object_area_urdf, "urdf")
plant.Finalize()

params = MeshcatVisualizerParams()
params.prefix = "planning"
visualizer = MeshcatVisualizer.AddToBuilder(
    builder, scene_graph, meshcat, params
)
diagram = builder.Build()

X_GgraspGpregrasp = RigidTransform([0, 0.0, -0.15])

align_grasp_axis = [-0.109,  0.994, -0.002]
z_axis, x_axis = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
rot_to_axes, _ = R.align_vectors(
    np.array([z_axis, x_axis]), np.stack([align_grasp_axis, x_axis])
)
# align grasp axis is blue
AddMeshcatTriad(meshcat, "align_grasp_axis", X_PT=RigidTransform(RotationMatrix(rot_to_axes.as_matrix().T),
                        [0.6, 0.0, 0.12]))

X_mustard = RigidTransform(RotationMatrix(RollPitchYaw(-np.pi/2, 0, np.pi/2)), [0.57, 0.0, 0.12])
X_box = RigidTransform(RotationMatrix(), [0.15, 0.0, 0.0])
X_WGfix = RigidTransform(RotationMatrix(RollPitchYaw(np.pi/2, 0, np.pi/2)))
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base_link_mustard"), X_mustard)

X_iiwa_base = RigidTransform(RotationMatrix())
new_positions = plant.GetPositions(plant_context)
print(new_positions)

plant.GetBodyByName("iiwa_link_7").index()

X_G = RigidTransform(
  R=RotationMatrix([
    [0.5295776198751465, 0.846339745369453, -0.05706645192532622],
    [0.8477111514205422, -0.5304580047265671, -0.0003301193244864655],
    [-0.030550749330160602, -0.04820104386296692, -0.9983703276269217],
  ]),
  p=[0.6078755953524031, 0.011975470771703942, 0.5630797220524787],
)

# q = [0] * 7 + list(new_positions[7:])
# q_goal = solve_global_inverse_kinematics(
#     plant=plant,
#     X_G=X_G,
#     initial_guess=q,
#     position_tolerance=0.01,
#     orientation_tolerance=0.01,
#     gripper_frame_name="iiwa_link_7",
# )

q_goal = [ 1.54171116,  1.2877689 ,  1.68350761,  1.21460623 , 0.24698678, -1.85501355,
 -1.39106498] + list(new_positions[7:])

plant.SetPositions(plant_context, q)
print(new_positions)

diagram.ForcedPublish(context)

while (1):
    pass