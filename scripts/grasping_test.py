import argparse
import numpy as np
import os
import copy
import pickle
import datetime
from scipy.spatial.transform import Rotation as R
from pydrake.geometry import (
    StartMeshcat,
    RenderLabel,
    Role,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import CameraInfo
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
from pydrake.math import (
    RigidTransform,
    RotationMatrix
)
from pydrake.solvers import MosekSolver, GurobiSolver

from planning.two_grasp_display_planner import TwoGraspPlanner
from perception.image_saver import ImageSaver
from perception.camera_in_world import CameraPoseInWorldSource
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
        scanning_traj_dir="scanning_traj",
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
        load_trajectories=False,
        regrasp=False,
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
    if not os.path.exists(dirstr):
        os.makedirs(dirstr)
    if not use_hardware and save_imgs:
        if not os.path.exists(dirstr+"/rgb/"):
            os.makedirs(dirstr+"/rgb/")
        if not os.path.exists(dirstr+"/depth/"):
            os.makedirs(dirstr+"/depth/")
        if not os.path.exists(dirstr+"/masks/"):
            os.makedirs(dirstr+"/masks/")
        if not os.path.exists(dirstr+"/gripper_masks/"):
            os.makedirs(dirstr+"/gripper_masks/")
        if not use_hardware and not os.path.exists(dirstr+"/ob_in_cam/"):
            os.makedirs(dirstr+"/ob_in_cam/")

        # save images
        if use_hardware:
            # this doesn't work because saving images takes too long and bottlenecks the whole system
            # please use scripts/realsense.py instead in parallel
            img_saver = builder.AddSystem(ImageSaver(depth_format="32F", dirstr=dirstr, labels=False, camera_info=True))
            builder.Connect(external_station.GetOutputPort("camera0.rgb_image"), img_saver.GetInputPort("rgb_in"))
            builder.Connect(external_station.GetOutputPort("camera0.depth_image"), img_saver.GetInputPort("depth_in"))
            builder.Connect(external_station.GetOutputPort("camera0.rgb_camera_info"), img_saver.GetInputPort("rgb_info_in"))
            builder.Connect(external_station.GetOutputPort("handeye_camera.depth_camera_info"), img_saver.GetInputPort("depth_info_in"))
        else:
            img_saver = builder.AddSystem(
                ImageSaver(
                    depth_format="32F",
                    dirstr=dirstr, 
                    labels=True, 
                    camera_info=False, 
                    ob_in_cam=True,
                    object_index=plant.GetBodyByName("base_link_mustard").index(),
                    camera_index=plant.GetBodyByName("base").index()
                )
            )
            builder.Connect(station.GetOutputPort("camera0.label_image"), img_saver.GetInputPort("label_in"))
            builder.Connect(station.GetOutputPort("camera0.rgb_image"), img_saver.GetInputPort("rgb_in"))
            builder.Connect(station.GetOutputPort("camera0.depth_image"), img_saver.GetInputPort("depth_in"))
            builder.Connect(station.GetOutputPort("body_poses"), img_saver.GetInputPort("body_poses"))


    # initialize point cloud output ports and save camera instrinsics
    if not use_hardware:
        camera0 = station.GetSubsystemByName("rgbd_sensor_camera0")
        handeye_camera = station.GetSubsystemByName("rgbd_sensor_handeye_camera")
        K = camera0.color_camera_info().intrinsic_matrix()
        if save_imgs:
            np.savetxt(dirstr+"/cam_K.txt", K)

        handeye_camera_pcd = builder.AddSystem(DepthImageToPointCloud(handeye_camera.depth_camera_info()))
        builder.Connect(station.GetOutputPort("handeye_camera.depth_image"), handeye_camera_pcd.GetInputPort("depth_image"))

    else:
        handeye_camera_pcd = builder.AddSystem(DepthImageToPointCloud(CameraInfo(848, 480, 639.036, 639.036, 425.131, 244.165)))

        builder.Connect(external_station.GetOutputPort("handeye_camera.depth_image"), handeye_camera_pcd.GetInputPort("depth_image"))

    
    # from camera calibation
    r = R.from_quat([0.00969807, -0.0140297, -0.70331, 0.710679])
    x_ee_camera = RigidTransform(
        R=RotationMatrix(r.as_matrix()),
        p = [-0.074597, 0.0324164, 0.155892]
    )

    camera_pose_source = builder.AddSystem(CameraPoseInWorldSource(x_ee_camera))
    eef_pose = builder.AddSystem(
        ExtractPose(
            plant.GetBodyByName("iiwa_link_7").index()
        )
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        eef_pose.get_input_port(),
    )
    builder.Connect(
        eef_pose.get_output_port(),
        camera_pose_source.GetInputPort("X_EE")
    )

    builder.Connect(
        camera_pose_source.GetOutputPort("X_WC"),
        handeye_camera_pcd.GetInputPort("camera_pose"),
    )

    controller_plant = station.GetSubsystemByName(
        "iiwa_controller_plant_pointer_system"
    ).get()

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
            plant=plant, 
            controller_plant=controller_plant,
            eef_body_index=plant.GetBodyByName("iiwa_link_7").index(),
            X_EefC=x_ee_camera,
            scanning_traj_dir=scanning_traj_dir,
            meshcat=meshcat,
            dirstr=dirstr,
            regions1=iris_regions1,
            regions2=iris_regions2,
            traj_dir=traj_dir,
            models_path=os.path.join(dir_path, os.path.join("scenario_datas", models_path)),
            no_obstacles=no_obstacles,
            regrasp=regrasp,
            gripper_model_path=gripper_model_path))

    if save_imgs:
        builder.Connect(planner.GetOutputPort("planner_state"), img_saver.GetInputPort("planner_state"))

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
        handeye_camera_pcd.GetOutputPort("point_cloud"),
        planner.GetInputPort("cloud_W"),
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

    # Remove labels of anything but mustard and gripper
    if not use_hardware:
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
            elif body.model_instance() == plant.GetModelInstanceByName("wsg"):
                properties.UpdateProperty("label", "id", RenderLabel(1)) # Make gripper label 1
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
        "--scenario_path",
        type=str,
        default="scenario_data_grasping.yml",
        help="yaml file with scenario",
    )
    parser.add_argument(
        "--models_path",
        type=str,
        default="scenario_data_grasping.dmd.yaml",
        help="dmd.yaml file with scenario, used for generating iris regions",
        nargs='?',
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="temp",
        help="directory to save images in",
        nargs='?',
    )
    parser.add_argument(
        "--pkl1_path",
        type=str,
        default="",
        help="path to first regions pkl file",
        nargs='?',
    )
    parser.add_argument(
        "--pkl2_path",
        type=str,
        default="",
        help="path to first regions pkl file",
        nargs='?',
    )
    parser.add_argument(
        "--traj_dir",
        type=str,
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
    parser.add_argument(
        "--regrasp",
        action='store_true',
        help="whether to regrasp when displaying",
    )
    args = parser.parse_args()

    gripper_model_path = "file://./home/real2sim/src/Real2SimObjectManipulation/models/schunk_wsg_50_welded_fingers_w_buffer.sdf"
    # gripper_model_path = "file://./home/evelyn/sources/Real2SimObjectManipulation/models/schunk_wsg_50_welded_fingers_w_buffer.sdf"
    # if args.use_hardware:
    #     gripper_model_path = "file://./home/real2sim/src/Real2SimObjectManipulation/models/schunk_wsg_50_welded_fingers_w_buffer.sdf"

    # Start the visualizer.
    meshcat = StartMeshcat()

    save_dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'tests', args.save_dir))
    scanning_traj_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'scanning_traj'))
    start_scenario(
        save_dir_path, 
        scenario_path= args.scenario_path, 
        gripper_model_path=gripper_model_path,
        models_path=args.models_path, 
        scanning_traj_dir=scanning_traj_path,
        pkl1_path=args.pkl1_path,
        pkl2_path=args.pkl2_path,
        traj_dir=args.traj_dir,
        use_hardware=args.use_hardware, 
        save_imgs=args.save_imgs,
        load_pkl_region1=args.load_pkl_region1,
        load_pkl_region2=args.load_pkl_region2,
        use_same_pkl_regions=args.use_same_pkl_regions,
        static_regions=args.static_regions,
        no_obstacles=args.no_obstacles,
        load_trajectories=args.load_trajectories,
        regrasp=args.regrasp,
    )