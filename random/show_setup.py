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
parser.AddModelsFromUrl("file://./home/evelyn/sources/Real2SimObjectManipulation/scenario_datas/scenario_data_grasping.dmd.yaml")
urdf = """
<?xml version="1.0"?>
<robot name="bounding_box">
  <link name="bounding_box">
    <visual name="bounding_box">
        <origin rpy="0.002455 0.000157 -0.044413" xyz="0.626098 -0.005697 0.208112"/>
      <geometry>
        <box size="0.057120 0.096090 0.189933"/>
      </geometry>
    </visual>
    <collision name="bounding_box">
        <origin rpy="0.002455 0.000157 -0.044413" xyz="0.626098 -0.005697 0.208112"/>
      <geometry>
        <box size="0.057120 0.096090 0.189933"/>
      </geometry>
    </collision>
  </link>
    <joint name="fixed_link_weld" type="fixed">
    <parent link="world"/>
    <child link="bounding_box"/>
    </joint>
</robot>"""
parser.AddModelsFromString(urdf, "urdf")
plant.Finalize()

params = MeshcatVisualizerParams()
params.prefix = "planning"
visualizer = MeshcatVisualizer.AddToBuilder(
    builder, scene_graph, meshcat, params
)
diagram = builder.Build()

context = diagram.CreateDefaultContext()

diagram.ForcedPublish(context)

while (1):
    pass