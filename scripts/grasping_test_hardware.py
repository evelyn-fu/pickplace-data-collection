import argparse
import numpy as np
import os
import copy
from pydrake.geometry import (
    StartMeshcat,
    RenderLabel,
    Role,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import (
    PortSwitch,
    Multiplexer,
    Demultiplexer
)
from pydrake.perception import (
    DepthImageToPointCloud
)

from manipulation.scenarios import AddIiwaDifferentialIK, ExtractBodyPose
from manipulation.station import MakeHardwareStation, load_scenario

from planning.two_grasp_display_planner import TwoGraspPlanner
from perception.image_saver import ImageSaver


def start_scenario(dirstr = "test4", scenario_path="scenario_data_grasping.yml", use_hardware=False, save_imgs=False):
    meshcat.ResetRenderMode()

    builder = DiagramBuilder()

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    filename = os.path.join(dir_path, os.path.join("scenario_datas", scenario_path))
    scenario = load_scenario(filename=filename)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat, hardware=False))
    if use_hardware:
        external_station = builder.AddSystem(MakeHardwareStation(scenario, meshcat, hardware=True))
    plant = station.GetSubsystemByName("plant")

    # initialize image writer and save directories
    camera0 = station.GetSubsystemByName("rgbd_sensor_camera0")
    camera1 = station.GetSubsystemByName("rgbd_sensor_camera1")
    camera2 = station.GetSubsystemByName("rgbd_sensor_camera2")
    K = camera0.color_camera_info().intrinsic_matrix()

    if save_imgs:
        if not os.path.exists(dirstr):
            os.makedirs(dirstr)
        if not os.path.exists(dirstr+"/rgb/"):
            os.makedirs(dirstr+"/rgb/")
        if not os.path.exists(dirstr+"/depth/"):
            os.makedirs(dirstr+"/depth/")
        if not os.path.exists(dirstr+"/masks/"):
            os.makedirs(dirstr+"/masks/")
        np.savetxt(dirstr+"/cam_K.txt", K)

        # save drake simulated images
        img_saver = builder.AddSystem(ImageSaver(dirstr))
        builder.Connect(station.GetOutputPort("camera0.rgb_image"), img_saver.GetInputPort("rgb_in"))
        builder.Connect(station.GetOutputPort("camera0.depth_image_16u"), img_saver.GetInputPort("depth_in"))
        builder.Connect(station.GetOutputPort("camera0.label_image"), img_saver.GetInputPort("label_in"))

    # initialize point cloud output ports
    camera0_pcd = builder.AddSystem(DepthImageToPointCloud(camera0.depth_camera_info()))
    camera1_pcd = builder.AddSystem(DepthImageToPointCloud(camera1.depth_camera_info()))
    camera2_pcd = builder.AddSystem(DepthImageToPointCloud(camera2.depth_camera_info()))

    builder.Connect(station.GetOutputPort("camera0.depth_image"), camera0_pcd.GetInputPort("depth_image"))
    # builder.Connect(station.GetOutputPort("camera0.rgb_image"), camera0_pcd.color_image_input_port())
    camera_pose0 = builder.AddSystem(
        ExtractBodyPose(
            plant.get_body_poses_output_port(), plant.GetBodyIndices(plant.GetModelInstanceByName("camera_main"))[0]
        )
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        camera_pose0.get_input_port(),
    )
    builder.Connect(
        camera_pose0.get_output_port(),
        camera0_pcd.GetInputPort("camera_pose"),
    )

    builder.Connect(station.GetOutputPort("camera1.depth_image"), camera1_pcd.GetInputPort("depth_image"))
    # builder.Connect(station.GetOutputPort("camera1.rgb_image"), camera1_pcd.color_image_input_port())
    camera_pose1 = builder.AddSystem(
        ExtractBodyPose(
            plant.get_body_poses_output_port(), plant.GetBodyIndices(plant.GetModelInstanceByName("camera_1"))[0]
        )
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        camera_pose1.get_input_port(),
    )
    builder.Connect(
        camera_pose1.get_output_port(),
        camera1_pcd.GetInputPort("camera_pose"),
    )

    builder.Connect(station.GetOutputPort("camera2.depth_image"), camera2_pcd.GetInputPort("depth_image"))
    # builder.Connect(station.GetOutputPort("camera2.rgb_image"), camera2_pcd.color_image_input_port())
    camera_pose2 = builder.AddSystem(
        ExtractBodyPose(
            plant.get_body_poses_output_port(), plant.GetBodyIndices(plant.GetModelInstanceByName("camera_2"))[0]
        )
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        camera_pose2.get_input_port(),
    )
    builder.Connect(
        camera_pose2.get_output_port(),
        camera2_pcd.GetInputPort("camera_pose"),
    )

    controller_plant = station.GetSubsystemByName(
        "iiwa.controller"
    ).get_multibody_plant_for_control()

    # Set up planner
    planner = builder.AddSystem(TwoGraspPlanner(
            plant, 
            controller_plant,
            camera_body_indices=[
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera_main"))[
                    0
                ],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera_1"))[
                    0
                ],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera_2"))[
                    0
                ]
            ],
            meshcat=meshcat,))

    if use_hardware:
        # Connect the output of external station to the input of internal station
        builder.Connect(
            external_station.GetOutputPort("iiwa.position_commanded"),
            station.GetInputPort("iiwa.position"),
        )

        wsg_state_demux: Demultiplexer = builder.AddSystem(Demultiplexer(2, 1))
        builder.Connect(
            external_station.GetOutputPort("wsg.state_measured"),
            wsg_state_demux.get_input_port(),
        )
        builder.Connect(
            wsg_state_demux.get_output_port(0),
            station.GetInputPort("wsg.position"),
        )
    
    if not use_hardware:
        builder.Connect(
            station.GetOutputPort("iiwa.position_measured"),
            planner.GetInputPort("iiwa_position"),
        )
    else:
        builder.Connect(
            external_station.GetOutputPort("iiwa.position_measured"),
            planner.GetInputPort("iiwa_position"),
        )

    # Set up differential inverse kinematics.
    differential_ik = AddIiwaDifferentialIK(
        builder,
        controller_plant,
        frame=controller_plant.GetFrameByName("iiwa_link_7"),
    )

    builder.Connect(planner.GetOutputPort("X_WG"), differential_ik.get_input_port(0))

    if not use_hardware:
        builder.Connect(
            station.GetOutputPort("iiwa.state_estimated"),
            differential_ik.GetInputPort("robot_state"),
        )
    else:
        # Export external state output
        iiwa_state_mux: Multiplexer = builder.AddSystem(Multiplexer([7, 7]))
        builder.Connect(
            external_station.GetOutputPort("iiwa.position_measured"),
            iiwa_state_mux.get_input_port(0),
        )
        builder.Connect(
            external_station.GetOutputPort("iiwa.velocity_estimated"),
            iiwa_state_mux.get_input_port(1),
        )
        builder.Connect(
            iiwa_state_mux.get_output_port(),
            differential_ik.GetInputPort("robot_state"),
        )

    builder.Connect(
        planner.GetOutputPort("reset_diff_ik"),
        differential_ik.GetInputPort("use_robot_state"),
    )

    if use_hardware:
        builder.Connect(
            planner.GetOutputPort("wsg_position"),
            external_station.GetInputPort("wsg.position"),
        )
    else:
        builder.Connect(
            planner.GetOutputPort("wsg_position"),
            station.GetInputPort("wsg.position"),
        )

    # The DiffIK and the direct position-control modes go through a PortSwitch
    switch = builder.AddSystem(PortSwitch(7))
    builder.Connect(
        differential_ik.get_output_port(), switch.DeclareInputPort("diff_ik")
    )
    builder.Connect(
        planner.GetOutputPort("iiwa_position_command"),
        switch.DeclareInputPort("position"),
    )
    if use_hardware:
        builder.Connect(
            switch.get_output_port(), external_station.GetInputPort("iiwa.position")
        )
    else:
        builder.Connect(
            switch.get_output_port(), station.GetInputPort("iiwa.position")
        )
    builder.Connect(
        planner.GetOutputPort("control_mode"),
        switch.get_port_selector_input_port(),
    )

    builder.Connect(
        camera0_pcd.GetOutputPort("point_cloud"),
        planner.GetInputPort("cloud0_W"),
    )
    builder.Connect(
        camera1_pcd.GetOutputPort("point_cloud"),
        planner.GetInputPort("cloud1_W"),
    )
    builder.Connect(
        camera2_pcd.GetOutputPort("point_cloud"),
        planner.GetInputPort("cloud2_W"),
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        planner.GetInputPort("body_poses"),
    )

    # Build diagram
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "save_dir",
        default="temp",
        help="directory to save images in",
    )
    parser.add_argument(
        "scenario_path",
        default="scenario_data_grasping.yml",
        help="yaml file with scenario",
    )
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )
    parser.add_argument(
        "--save_imgs",
        action='store_true',
        help="yaml file with scenario",
    )
    args = parser.parse_args()

    # Start the visualizer.
    meshcat = StartMeshcat()

    save_dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'tests', args.save_dir))
    start_scenario(save_dir_path, scenario_path= args.scenario_path, use_hardware=args.use_hardware, save_imgs=args.save_imgs)