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

X_G =RigidTransform(
            R=RotationMatrix([
                [-0.03442034895124868, 0.9994073003679098, 0.0005362363300874173],
                [-0.04829586151222898, -0.002199273198199581, 0.9988306527926499],
                [0.9982398255624083, 0.0343542016167908, 0.04834293632380032],
            ]) @ RotationMatrix(RollPitchYaw(0, -0.6, 0)),
            p=[0.5857679659224048, -0.10382563627445854, 0.24689332681875962],
            )


align_grasp_axis = [-0.109,  0.994, -0.002]
z_axis, x_axis = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
rot_to_axes, _ = R.align_vectors(
    np.array([z_axis, x_axis]), np.stack([align_grasp_axis, x_axis])
)
# align grasp axis is blue
AddMeshcatTriad(meshcat, "align_grasp_axis", X_PT=RigidTransform(RotationMatrix(rot_to_axes.as_matrix().T),
                        [0.6, 0.0, 0.12]))

X_mustard = RigidTransform(RotationMatrix(RollPitchYaw(-np.pi/2, 0, np.pi/2)), [0.57, 0.0, 0.12])
X_WGfix = RigidTransform(RotationMatrix(RollPitchYaw(np.pi/2, 0, np.pi/2)))
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body"), X_G.multiply(X_WGfix))
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("base_link_mustard"), X_mustard)

diagram.ForcedPublish(context)

while (1):
    pass