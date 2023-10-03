import numpy as np
import os
import copy
from pydrake.geometry import (
    StartMeshcat,
    SceneGraph,
    ClippingRange,
    ColorRenderCamera,
    DepthRange,
    DepthRenderCamera,
    MakeRenderEngineVtk,
    RenderCameraCore,
    RenderEngineVtkParams,
    RenderLabel,
    Role,
)
from pydrake.multibody.inverse_kinematics import (
    DifferentialInverseKinematicsParameters,
    DifferentialInverseKinematicsStatus,
    DoDifferentialInverseKinematics,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, EventStatus, LeafSystem
from pydrake.visualization import (
    MeshcatPoseSliders,
    ColorizeDepthImage,
    ColorizeLabelImage,
    AddDefaultVisualization,
)
from pydrake.systems.sensors import (
    ImageRgba8U,
    CameraInfo,
    RgbdSensor,
    ImageWriter,
    PixelType,
)
from pydrake.common.value import Value
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph

from manipulation import running_as_notebook
from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK, ExtractBodyPose
from manipulation.station import MakeHardwareStation, load_scenario


def teleop_with_camera(dirstr = "test2"):
    meshcat.ResetRenderMode()

    builder = DiagramBuilder()

    scenario = load_scenario(filename="scenario_data.yml")
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))

    # TODO(russt): Replace with station.AddDiffIk(...)
    controller_plant = station.GetSubsystemByName(
        "iiwa.controller"
    ).get_multibody_plant_for_control()
    # Set up differential inverse kinematics.
    differential_ik = AddIiwaDifferentialIK(
        builder,
        controller_plant,
        frame=controller_plant.GetFrameByName("iiwa_link_7"),
    )
    builder.Connect(
        differential_ik.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.state_estimated"),
        differential_ik.GetInputPort("robot_state"),
    )

    # Set up teleop widgets.
    meshcat.DeleteAddedControls()
    teleop = builder.AddSystem(
        MeshcatPoseSliders(
            meshcat,
            lower_limit=[0, -0.5, -np.pi, -0.6, -0.8, 0.0],
            upper_limit=[2 * np.pi, np.pi, np.pi, 0.8, 0.3, 1.1],
        )
    )
    builder.Connect(
        teleop.get_output_port(), differential_ik.GetInputPort("X_WE_desired")
    )
    # Note: This is using "Cheat Ports". For it to work on hardware, we would
    # need to construct the initial pose from the HardwareStation outputs.
    plant = station.GetSubsystemByName("plant")
    ee_pose = builder.AddSystem(
        ExtractBodyPose(
            station.GetOutputPort("body_poses"),
            plant.GetBodyByName("iiwa_link_7").index(),
        )
    )
    builder.Connect(
        station.GetOutputPort("body_poses"), ee_pose.get_input_port()
    )
    builder.Connect(ee_pose.get_output_port(), teleop.get_input_port())
    wsg_teleop = builder.AddSystem(WsgButton(meshcat))
    builder.Connect(
        wsg_teleop.get_output_port(0), station.GetInputPort("wsg.position")
    )

    colorize_depth = builder.AddSystem(ColorizeDepthImage())
    colorize_label = builder.AddSystem(ColorizeLabelImage())
    colorize_label.background_color.set([0,0,0])
    builder.Connect(station.GetOutputPort("camera0.depth_image"),
                    colorize_depth.GetInputPort("depth_image_32f"))
    builder.Connect(station.GetOutputPort("camera0.label_image"),
                    colorize_label.GetInputPort("label_image"))

    # initialize image writer and save directories
    sensor = station.GetSubsystemByName("rgbd_sensor_camera0")
    K = sensor.color_camera_info().intrinsic_matrix()
    if not os.path.exists(dirstr):
        os.makedirs(dirstr)
    if not os.path.exists(dirstr+"/RGB/"):
        os.makedirs(dirstr+"/RGB/")
    if not os.path.exists(dirstr+"/depth/"):
        os.makedirs(dirstr+"/depth/")
    if not os.path.exists(dirstr+"/masks/"):
        os.makedirs(dirstr+"/masks/")
    np.savetxt(dirstr+"/cam_K.txt", K)

    img_writer = builder.AddSystem(ImageWriter())
    img_writer.DeclareImageInputPort(
        pixel_type=PixelType.kRgba8U, 
        port_name="RGB", 
        file_name_format=dirstr+"/{port_name}/{time_usec}",
        publish_period=0.03,
        start_time=0.0,
    )   
    img_writer.DeclareImageInputPort(
        pixel_type=PixelType.kRgba8U, 
        port_name="depth", 
        file_name_format=dirstr+"/{port_name}/{time_usec}",
        publish_period=0.03,
        start_time=0.0,
    )   
    img_writer.DeclareImageInputPort(
        pixel_type=PixelType.kRgba8U, 
        port_name="masks", 
        file_name_format=dirstr+"/{port_name}/{time_usec}",
        publish_period=0.03,
        start_time=0.0,
    )   
    
    # Connect to image writer
    builder.Connect(station.GetOutputPort("camera0.rgb_image"),
                    img_writer.GetInputPort("RGB"))
    builder.Connect(colorize_depth.get_output_port(),
                    img_writer.GetInputPort("depth"))
    builder.Connect(colorize_label.get_output_port(),
                    img_writer.GetInputPort("masks"))

    # Build diagram
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()

    # Simulate
    simulator = Simulator(diagram)
    simulator_context = simulator.get_mutable_context()

    # Remove labels of anything but mustard
    scene_graph = station.GetSubsystemByName("scene_graph")
    source_id = plant.get_source_id()
    scene_graph_context = scene_graph.GetMyMutableContextFromRoot(simulator_context)
    query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
    inspector = query_object.inspector()
    for geometry_id in inspector.GetAllGeometryIds():
        properties = copy.deepcopy(inspector.GetPerceptionProperties(geometry_id))
        if properties is None:
            continue
        frame_id = inspector.GetFrameId(geometry_id)
        body = plant.GetBodyFromFrameId(frame_id)
        if body.model_instance() == plant.GetModelInstanceByName("mustard_bottle"):
            properties.UpdateProperty("label", "id", RenderLabel(0)) # Make mustard label 0
        else:
            properties.UpdateProperty("label", "id", RenderLabel.kDontCare)
        scene_graph.RemoveRole(scene_graph_context, source_id, geometry_id, Role.kPerception)
        scene_graph.AssignRole(scene_graph_context, source_id, geometry_id, properties)


    simulator.set_target_realtime_rate(1.0)

    meshcat.AddButton("Stop Simulation", "Escape")
    print("Press Escape to stop the simulation")
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.03)
    meshcat.DeleteButton("Stop Simulation")


if __name__ == "__main__":
    # Start the visualizer.
    meshcat = StartMeshcat()
    teleop_with_camera()