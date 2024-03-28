import argparse
import numpy as np
import os
import copy
import pickle
import datetime
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

from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.systems import ExtractPose
from manipulation.station import MakeHardwareStation, LoadScenario

import pydrake.planning as mut
from pydrake.common import RandomGenerator, Parallelism, use_native_cpp_logging
from pydrake.planning import (RobotDiagramBuilder,
                              SceneGraphCollisionChecker,
                              CollisionCheckerParams)
from pydrake.solvers import MosekSolver, GurobiSolver

from planning.two_grasp_display_planner import TwoGraspPlanner
from perception.image_saver import ImageSaver
from planning.trajectory_sources import TrajectoryWithTimingInformationSource, DummyTrajSource

def get_regions_static(scenario_path, dirstr):
    print("generating static regions")
    use_native_cpp_logging()
    params = dict(edge_step_size=0.125)
    builder = RobotDiagramBuilder()
    builder.parser().AddModels(scenario_path)
    iiwa_model_instance_index = builder.plant().GetModelInstanceByName("iiwa")
    wsg_model_instance_index = builder.plant().GetModelInstanceByName("wsg")
    params["robot_model_instances"] = [iiwa_model_instance_index, wsg_model_instance_index]
    params["model"] = builder.Build()
    checker = SceneGraphCollisionChecker(**params)

    options = mut.IrisFromCliqueCoverOptions()
    options.num_points_per_coverage_check = 5000
    options.num_points_per_visibility_round = 1000
    options.minimum_clique_size = 16
    options.coverage_termination_threshold = 0.7

    generator = RandomGenerator(0)

    if (MosekSolver().available() and MosekSolver().enabled()) or (
            GurobiSolver().available() and GurobiSolver().enabled()):
        # We need a MIP solver to be available to run this method.
        sets = mut.IrisInConfigurationSpaceFromCliqueCover(
            checker=checker, options=options, generator=generator,
            sets=[]
        )

        if len(sets) < 1:
            raise("No regions found")
        
        time_str = datetime.datetime.now().strftime('%d%m%y_%H%M%S')
        with open(dirstr+f'/{scenario_path.split("/")[-1]}_{time_str}_regions.pkl', 'wb') as f:
            pickle.dump(sets, f)

        return sets
    else:
        print("No solvers available")

def start_scenario(
        dirstr = "temp", 
        scenario_path="scenario_data_grasping.yml", 
        models_path="scenario_data_grasping_no_object.dmd.yaml", 
        gripper_model_path="",
        pkl1_path="", 
        pkl2_path="", 
        traj_dir="",
        use_hardware=False, 
        save_imgs=False,
        load_pkl_region1=False,
        load_pkl_region2=False,
        use_same_pkl_regions=False,
        static_regions=False,
        no_obstacles=False,
        load_trajectories=False
    ):
    if load_pkl_region1:
        if pkl1_path == "":
            print("Must provide path to at pkl file to load region 1")
            return
    if use_same_pkl_regions:
        pkl2_path = pkl1_path
        print("Using regions 1 for regions 2")
    elif load_pkl_region2:
        if pkl2_path == "":
            print("Must provide path to at pkl file to load region 2")
            return
        
    if load_trajectories and traj_dir == "":
        print("Must provide path to directory with trajectories to load trajectories")
        return
    if not load_trajectories:
        traj_dir = None

    meshcat.ResetRenderMode()

    builder = DiagramBuilder()

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    filename = os.path.join(dir_path, os.path.join("scenario_datas", scenario_path))
    scenario = LoadScenario(filename=filename)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat, hardware=False))
    if use_hardware:
        external_station = builder.AddSystem(MakeHardwareStation(scenario, meshcat, hardware=True))
    plant = station.GetSubsystemByName("plant")

    # initialize image writer and save directories
    camera0 = station.GetSubsystemByName("rgbd_sensor_camera0")
    camera1 = station.GetSubsystemByName("rgbd_sensor_camera1")
    camera2 = station.GetSubsystemByName("rgbd_sensor_camera2")
    K = camera0.color_camera_info().intrinsic_matrix()

    if not os.path.exists(dirstr):
        os.makedirs(dirstr)
    if save_imgs:
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
        builder.Connect(station.GetOutputPort("camera0.depth_image"), img_saver.GetInputPort("depth_in"))
        builder.Connect(station.GetOutputPort("camera0.label_image"), img_saver.GetInputPort("label_in"))

    # initialize point cloud output ports
    camera0_pcd = builder.AddSystem(DepthImageToPointCloud(camera0.depth_camera_info()))
    camera1_pcd = builder.AddSystem(DepthImageToPointCloud(camera1.depth_camera_info()))
    camera2_pcd = builder.AddSystem(DepthImageToPointCloud(camera2.depth_camera_info()))

    builder.Connect(station.GetOutputPort("camera0.depth_image"), camera0_pcd.GetInputPort("depth_image"))
    camera_pose0 = builder.AddSystem(
        ExtractPose(
            plant.GetBodyIndices(plant.GetModelInstanceByName("camera_main"))[0]
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
    camera_pose1 = builder.AddSystem(
        ExtractPose(
            plant.GetBodyIndices(plant.GetModelInstanceByName("camera_1"))[0]
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
    camera_pose2 = builder.AddSystem(
        ExtractPose(
            plant.GetBodyIndices(plant.GetModelInstanceByName("camera_2"))[0]
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

    iris_regions1 = None
    iris_regions2 = None
    if static_regions:
        iris_regions1 = get_regions_static(os.path.join(dir_path, os.path.join("scenario_datas", models_path)), dirstr)
        iris_regions2 = iris_regions1
    if load_pkl_region1:
        with open(pkl1_path, 'rb') as f:
            iris_regions1 = pickle.load(f)
        if use_same_pkl_regions:
            iris_regions2 = iris_regions1
    if load_pkl_region2 and iris_regions2 is None:
        with open(pkl2_path, 'rb') as f:
            iris_regions2 = pickle.load(f)
    
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
            meshcat=meshcat,
            regions1=iris_regions1,
            regions2=iris_regions2,
            traj_dir=traj_dir,
            models_path=os.path.join(dir_path, os.path.join("scenario_datas", models_path)),
            dirstr=dirstr,
            no_obstacles=no_obstacles,
            gripper_model_path=gripper_model_path))

    if use_hardware:
        # Connect the output of external station to the input of internal station
        builder.Connect(
            external_station.GetOutputPort("iiwa.position_measured"),
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

    joint_traj_source: TrajectoryWithTimingInformationSource = (
        builder.AddNamedSystem(
            "joint_traj_source",
            TrajectoryWithTimingInformationSource(
                trajectory_size=7
            ),
        )
    )

    builder.Connect(
        planner.GetOutputPort("joint_position_trajectory"),
        joint_traj_source.GetInputPort("trajectory"),
    )
    if use_hardware:
        builder.Connect(
            external_station.GetOutputPort("iiwa.position_commanded"),
            joint_traj_source.GetInputPort("current_cmd"),
        )
    else:
        builder.Connect(
            station.GetOutputPort("iiwa.position_measured"),
            joint_traj_source.GetInputPort("current_cmd"),
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

    if use_hardware:
        builder.Connect(
            joint_traj_source.get_output_port(), external_station.GetInputPort("iiwa.position")
        )
    else:
        builder.Connect(
            joint_traj_source.get_output_port(), station.GetInputPort("iiwa.position")
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
    while meshcat.GetButtonClicks("Stop Simulation") < 1 and not planner.done:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.03)

    meshcat.DeleteButton("Stop Simulation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scenario_path",
        default="scenario_data_grasping.yml",
        help="yaml file with scenario",
    )
    parser.add_argument(
        "models_path",
        default="scenario_data_grasping.dmd.yaml",
        help="yaml file with scenario",
        nargs='?',
    )
    parser.add_argument(
        "save_dir",
        default="temp",
        help="directory to save images in",
        nargs='?',
    )
    parser.add_argument(
        "pkl1_path",
        default="",
        help="path to first regions pkl file",
        nargs='?',
    )
    parser.add_argument(
        "pkl2_path",
        default="",
        help="path to first regions pkl file",
        nargs='?',
    )
    parser.add_argument(
        "traj_dir",
        default="",
        help="path to directory with saved gcs trajectories",
        nargs='?',
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
    parser.add_argument(
        "--load_pkl_region1",
        action='store_true',
        help="whether to load regions from pkl file",
    )
    parser.add_argument(
        "--load_pkl_region2",
        action='store_true',
        help="whether to load regions from pkl file",
    )
    parser.add_argument(
        "--use_same_pkl_regions",
        action='store_true',
        help="whether to load regions from pkl file",
    )
    parser.add_argument(
        "--static_regions",
        action='store_true',
        help="if false includes object from point cloud as obstacle",
    )
    parser.add_argument(
        "--no_obstacles",
        action='store_true',
        help="if true, plans gcs without any obstacles",
    )
    parser.add_argument(
        "--load_trajectories",
        action='store_true',
        help="whether to load gcs trajectories from traj_dir",
    )
    args = parser.parse_args()

    gripper_model_path = "file://./home/evelyn/sources/Real2SimObjectManipulation/models/schunk_wsg_50_welded_fingers_w_buffer.sdf"
    if args.use_hardware:
        gripper_model_path = "file://./home/real2sim/src/Real2SimObjectManipulation/models/schunk_wsg_50_welded_fingers_w_buffer.sdf"

    # Start the visualizer.
    meshcat = StartMeshcat()

    save_dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'tests', args.save_dir))
    start_scenario(
        save_dir_path, 
        scenario_path= args.scenario_path, 
        gripper_model_path=gripper_model_path,
        models_path=args.models_path, 
        pkl1_path=args.pkl1_path,
        pkl2_path=args.pkl2_path,
        traj_dir=args.traj_dir,
        use_hardware=args.use_hardware, 
        save_imgs=args.save_imgs,
        load_pkl_region1=args.load_pkl_region1,
        load_pkl_region2=args.load_pkl_region2,
        use_same_pkl_regions=args.use_same_pkl_regions,
        static_regions=args.static_regions,
        no_obstacles=(args.no_obstacles or args.load_trajectories),
        load_trajectories=args.load_trajectories
    )