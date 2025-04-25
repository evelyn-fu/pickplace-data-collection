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
from scipy.spatial.transform import Rotation as R
import os

# Start the visualizer.
meshcat = StartMeshcat()

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0005)
parser = Parser(plant)
directory_path = os.path.dirname(os.path.abspath(__file__))
models_package = os.path.abspath(os.path.join(directory_path, "..", "models", "package.xml"))
parser.package_map().AddPackageXml(models_package)
ConfigureParser(parser)
directory_path = os.path.dirname(os.path.abspath(__file__))
gripper_model_path = "package://pickplace_data_collection/schunk_wsg_50_large_grippers_w_buffer.sdf"
parser.AddModelsFromUrl(gripper_model_path)
plant.Finalize()

params = MeshcatVisualizerParams()
params.prefix = "planning"
visualizer = MeshcatVisualizer.AddToBuilder(
    builder, scene_graph, meshcat, params
)
diagram = builder.Build()

X_G = RigidTransform(
  R=RotationMatrix([
    [0.20681969377898063, 0.9685467022240954, 0.1383578688618696],
    [0.9774142324949082, -0.21081815591263353, 0.014735102781668053],
    [0.043439985975579215, 0.13218544075814884, -0.9902726780387395],
  ]),
  p=[0.5798396344223478, -0.0008122595958626092, 0.3250179079887979],
)

r = R.from_quat([0.010822, -0.0145512, -0.702256, 0.711694])
x_ee_camera = RigidTransform(
    R=RotationMatrix(r.as_matrix()),
    p = [-0.0730357, 0.032904, 0.151341]
)
X_WCamera = X_G @ x_ee_camera

rot = X_WCamera.GetAsMatrix4()[:3, :3]
t = X_WCamera.GetAsMatrix4()[:3, 3]
camera_z_vec = rot.dot(np.array([0, 0, 1]))
camera_x_vec = rot.dot(np.array([1, 0, 0]))
z_axis, x_axis = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
rot_to_axes, _ = R.align_vectors(
    np.array([z_axis, x_axis]), np.stack([camera_z_vec, camera_x_vec])
)

# align eff_vertical_vec is blue, eff_horizontal_vec is red
AddMeshcatTriad(meshcat, "camera", X_PT=RigidTransform(RotationMatrix(rot_to_axes.as_matrix().T),
                        t))

X_WGfix = RigidTransform(RotationMatrix(RollPitchYaw(np.pi/2, 0, np.pi/2)))
rot_180 = RigidTransform(RotationMatrix(RollPitchYaw(0, np.pi, 0)))
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)
plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body"), X_G.multiply(X_WGfix) @ rot_180)

diagram.ForcedPublish(context)

while (1):
    pass