import copy
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from pydrake.geometry import (
    ClippingRange,
    ColorRenderCamera,
    DepthRange,
    DepthRenderCamera,
    MakeRenderEngineVtk,
    RenderCameraCore,
    RenderEngineVtkParams,
    RenderLabel,
    Role,
    StartMeshcat,
)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import BodyIndex
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import (
    CameraInfo,
    RgbdSensor,
)
from pydrake.visualization import (
    AddDefaultVisualization,
    ColorizeDepthImage,
    ColorizeLabelImage,
)

def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)

def start_camera_sim():
    meshcat = StartMeshcat()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)

    iiwa_url = (
        "package://drake/manipulation/models/iiwa_description/sdf/"
        "iiwa14_no_collision.sdf"
    )

    mustard = parser.AddModelsFromUrl(
        "package://drake/manipulation/models/ycb/sdf/006_mustard_bottle.sdf"
    )[0]

    mustard_body = plant.GetBodyByName("006_mustard_bottle", mustard)
    plant.SetDefaultFreeBodyPose(mustard_body, xyz_rpy_deg([0, 0, 0.5], [0, 0, 0]))

    renderer_name = "renderer"
    scene_graph.AddRenderer(
        renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))

    # N.B. These properties are chosen arbitrarily.
    intrinsics = CameraInfo(
        width=640,
        height=480,
        fov_y=np.pi/4,
    )
    core = RenderCameraCore(
        renderer_name,
        intrinsics,
        ClippingRange(0.01, 10.0),
        RigidTransform(),
    )
    color_camera = ColorRenderCamera(core, show_window=False)
    depth_camera = DepthRenderCamera(core, DepthRange(0.01, 10.0))

    world_id = plant.GetBodyFrameIdOrThrow(plant.world_body().index())
    X_WB = xyz_rpy_deg([2, 0, 0.75], [-90, 0, 90])
    sensor = RgbdSensor(
        world_id,
        X_PB=X_WB,
        color_camera=color_camera,
        depth_camera=depth_camera,
    )

    instrinsic_matrix = sensor.depth_camera_info().instrinsic_matrix()

    builder.AddSystem(sensor)
    builder.Connect(
        scene_graph.get_query_output_port(),
        sensor.query_object_input_port(),
    )

    plant.Finalize()

    AddDefaultVisualization(builder=builder, meshcat=meshcat)

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()

    Simulator(diagram).Initialize()

if __name__ == "__main__":
    start_camera_sim()