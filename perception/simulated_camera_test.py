import copy
import os
from PIL import Image

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
    StartMeshcat,
)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import (
    CameraInfo,
    RgbdSensor,
)
from pydrake.visualization import (
    AddDefaultVisualization,
)

import time

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    StartMeshcat,
)

def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)

def Visualizer(dirstr):
    builder = DiagramBuilder()

    # Make plant, scene graph, add body
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    mustard_url = "package://drake/manipulation/models/ycb/sdf/006_mustard_bottle.sdf"
    (mustard,) = Parser(plant).AddModels(url=mustard_url)
    mustard_body = plant.GetBodyByName("base_link_mustard", mustard)
    
    # Add cameras
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
    X_WB = xyz_rpy_deg([0.5, 0, 0.5], [-90, 0, 90])
    sensor = RgbdSensor(
        world_id,
        X_PB=X_WB,
        color_camera=color_camera,
        depth_camera=depth_camera,
    )

    # Save camera intrinsics here
    K = sensor.color_camera_info().intrinsic_matrix()
    if not os.path.exists(dirstr):
        os.makedirs(dirstr)
    if not os.path.exists(dirstr+"/rgb/"):
        os.makedirs(dirstr+"/rgb/")
    if not os.path.exists(dirstr+"/depth/"):
        os.makedirs(dirstr+"/depth/")
    if not os.path.exists(dirstr+"/masks/"):
        os.makedirs(dirstr+"/masks/")
    np.savetxt(dirstr+"/cam_K.txt", K)

    builder.AddSystem(sensor)
    builder.Connect(
        scene_graph.get_query_output_port(),
        sensor.query_object_input_port(),
    )

    plant.Finalize()
    print(plant.GetStateNames())

    AddDefaultVisualization(builder=builder, meshcat=meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    meshcat.Delete()

    def visualize(q, timestr):
        plant.SetPositions(plant_context, q)
        diagram.ForcedPublish(context)

        color = sensor.color_image_output_port().Eval(
            sensor.GetMyContextFromRoot(context)).data
        depth = copy.deepcopy(
            sensor.GetOutputPort("depth_image_16u").Eval(
                sensor.GetMyContextFromRoot(context)).data.squeeze()
        )
        label_image = copy.deepcopy(
            sensor.GetOutputPort("label_image").Eval(
                sensor.GetMyContextFromRoot(context)).data.squeeze()
        )

        plt.imsave(dirstr+"/rgb/"+timestr+".png", color)
        
        object_labels = np.unique(label_image)
        masks = [
            np.uint8(np.where(label_image == label, 255, 0)) for label in object_labels
        ]

        mask_pil = Image.fromarray(masks[0])
        mask_pil.save(dirstr+"/masks/"+timestr+".png")

        depth[depth > 3000] = 3000
        depth_pil = Image.fromarray(depth)
        depth_pil.save(dirstr+"/depth/"+timestr+".png")

    return visualize

if __name__ == "__main__":
    meshcat = StartMeshcat()

    visualize = Visualizer("test1_manual")

    meshcat.AddSlider(name="x", value=0, min=-0.5, max=0.5, step=0.01)
    meshcat.AddSlider(name="y", value=0, min=-0.5, max=0.5, step=0.01)
    meshcat.AddSlider(name="z", value=0.5, min=-0.0, max=1.0, step=0.01)
    meshcat.AddSlider(name="x_rot", value=0, min=-np.pi, max=np.pi, step=0.01)
    meshcat.AddSlider(name="y_rot", value=0, min=-np.pi, max=np.pi, step=0.01)
    meshcat.AddSlider(name="z_rot", value=0,  min=-np.pi, max=np.pi, step=0.01)

    meshcat.AddButton("Stop Interaction Loop")
    time_step = 0
    x = -3.14
    y = -3.14
    z = -3.14
    while meshcat.GetButtonClicks("Stop Interaction Loop") < 1:
        # if time_step < 314:
        #     x += 0.02
        # elif time_step < 628:
        #     y += 0.02
        # elif time_step < 942:
        #     z += 0.02
        # else:
        #     break

        # q = [1, x, y, z, meshcat.GetSliderValue("x"), meshcat.GetSliderValue("y"), meshcat.GetSliderValue("z")]

        q = [1, meshcat.GetSliderValue("x_rot"), meshcat.GetSliderValue("y_rot"), meshcat.GetSliderValue("z_rot"),
            meshcat.GetSliderValue("x"), meshcat.GetSliderValue("y"), meshcat.GetSliderValue("z")]

        visualize(q, f"{time_step:03d}")
        time.sleep(0.01)
        time_step += 1
    meshcat.DeleteAddedControls()