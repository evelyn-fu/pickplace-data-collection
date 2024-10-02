import copy
import os
from PIL import Image
import argparse

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
    Quaternion,
    RotationMatrix
)

def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)

def Visualizer(dirstr):
    builder = DiagramBuilder()

    # Make plant, scene graph, add body
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    scenario_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'scenario_datas', 'scenario_data_grasping.dmd.yaml'))
    Parser(plant).AddModels(scenario_path)
    # mustard_body = plant.GetBodyByName("base_link_mustard")
    # mustard_joint = plant.AddJoint(
    #     QuaternionFloatingJoint(
    #         "mustard_joint",
    #         plant.GetFrameByName("world"),
    #         plant.GetFrameByName("mustard_origin"),
    #     )
    # )
    
    # Add cameras
    renderer_name = "renderer"
    scene_graph.AddRenderer(
        renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))
    
    # N.B. These properties are chosen arbitrarily.
    intrinsics = CameraInfo(
        width=1920,
        height=1440,
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
    X_WB = xyz_rpy_deg([1.2, 0, 0.54], [-110, 0, 90])
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
    if not os.path.exists(dirstr+"/ob_in_cam/"):
        os.makedirs(dirstr+"/ob_in_cam/")
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
        q_old = plant.GetPositions(plant_context)
        q_combined = q_old.copy()
        q_combined[7:] = q
        q_combined[8:11] = [0,0,0]
        plant.SetPositions(plant_context, q_combined)

        mustard_body = plant.GetBodyByName("base_link_mustard")
        old_pose = plant.GetFreeBodyPose(plant_context, mustard_body)
        new_pose = RigidTransform(old_pose.rotation() @ RotationMatrix(RollPitchYaw(q[1], q[2], q[3])), old_pose.translation())
        plant.SetFreeBodyPose(plant_context, mustard_body, 
            new_pose)
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

        color = color[:, :, :3]
        color_pil = Image.fromarray(color)
        color_pil.save(dirstr+"/rgb/"+timestr+".png")
        
        object_labels = np.unique(label_image)
        masks = [
            np.uint8(np.where(label_image == label, 255, 0)) for label in object_labels
        ]

        mask_pil = Image.fromarray(masks[4])
        mask_pil.save(dirstr+"/masks/"+timestr+".png")

        depth[depth > 3000] = 3000
        depth_pil = Image.fromarray(depth)
        depth_pil.save(dirstr+"/depth/"+timestr+".png")

        o2w = new_pose
        c2w = X_WB

        o2c = c2w.inverse() @ o2w
        T = o2c.GetAsMatrix4()
        np.savetxt(dirstr+"/ob_in_cam/"+timestr+".txt", T)

    return visualize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "save_dir",
        default="temp",
        help="directory to save images in",
        nargs='?',
    )
    args = parser.parse_args()
    save_dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'tests', args.save_dir))

    meshcat = StartMeshcat()

    visualize = Visualizer(save_dir_path)

    meshcat.AddSlider(name="x", value=0.6, min=-0.5, max=0.6, step=0.01)
    meshcat.AddSlider(name="y", value=0, min=-0.5, max=0.5, step=0.01)
    meshcat.AddSlider(name="z", value=0.4, min=-0.0, max=1.0, step=0.01)
    meshcat.AddSlider(name="x_rot", value=0, min=-2*np.pi, max=2*np.pi, step=0.01)
    meshcat.AddSlider(name="y_rot", value=0, min=-2*np.pi, max=2*np.pi, step=0.01)
    meshcat.AddSlider(name="z_rot", value=0,  min=-2*np.pi, max=2*np.pi, step=0.01)

    meshcat.AddButton("Stop Interaction Loop")
    time_step = 0
    x = -1.57
    y = 0
    z = -1.57
    while meshcat.GetButtonClicks("Stop Interaction Loop") < 1:
        # x = meshcat.GetSliderValue("x_rot")
        # y = meshcat.GetSliderValue("y_rot")
        # z = meshcat.GetSliderValue("z_rot")
        if time_step < 157:
            z += 0.02
            if time_step == 157 - 1:
                z = 1.57
        elif time_step < 471:
            z -= 0.02
            if time_step == 471 - 1:
                z = -4.71
        elif time_step < 628:
            z += 0.02
            if time_step == 628 - 1:
                z = -1.57
        elif time_step < 707:
            y += 0.02
            if time_step == 707 - 1:
                y = 1.57
        elif time_step < 864:
            x += 0.02
            if time_step == 864 - 1:
                x = 1.57
        elif time_step < 1178:
            x -= 0.02
            if time_step == 1178 - 1:
                x = -4.71
        elif time_step < 1335:
            x += 0.02
            if time_step == 1335 - 1:
                x = -1.57
        else:
            break

        q = [1, x, y, z, meshcat.GetSliderValue("x"), meshcat.GetSliderValue("y"), meshcat.GetSliderValue("z")]

        # q = [1, meshcat.GetSliderValue("x_rot"), meshcat.GetSliderValue("y_rot"), meshcat.GetSliderValue("z_rot"),
        #     meshcat.GetSliderValue("x"), meshcat.GetSliderValue("y"), meshcat.GetSliderValue("z")]

        visualize(q, f"{time_step:04d}")
        # time.sleep(0.001)
        time_step += 1
    meshcat.DeleteAddedControls()