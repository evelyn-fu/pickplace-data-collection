import argparse
import numpy as np
import os
import copy
import pickle
import datetime
import shutil
from scipy.spatial.transform import Rotation as R
from pydrake.geometry import (
    StartMeshcat,
    RenderLabel,
    Role,
)
from pydrake.all import (
    ConstantVectorSource, 
    VectorLogSink,
    ApplySimulatorConfig,
    DiagramBuilder,
    RobotDiagram,
    PiecewisePolynomial,
    Simulator,
    TrajectorySource,
    StartMeshcat,
    LeafSystem,CompositeTrajectory,PathParameterizedTrajectory
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
from robot_payload_id.control.trajectory import FourierSeriesTrajectory
from robot_payload_id.utils import FourierSeriesTrajectoryAttributes

# from manipulation.systems import AddIiwaDifferentialIK
from manipulation.systems import ExtractPose
from manipulation.station import MakeHardwareStation, LoadScenario

import pydrake.planning as mut
from pydrake.common import RandomGenerator, Parallelism, use_native_cpp_logging
from pydrake.planning import (RobotDiagramBuilder,
                              SceneGraphCollisionChecker,
                              CollisionCheckerParams)
from pydrake.math import (
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
)
from pathlib import Path
from planning.utils.planner_states import PlannerState, PickState
from pydrake.solvers import MosekSolver, GurobiSolver
from pydrake.all import LeafSystem, Value, Context, InputPort
from planning.two_grasp_display_planner import TwoGraspPlanner
from planning.turntable_planner import TurntablePlanner
from perception.image_saver import ImageSaver
from perception.camera_in_world import CameraPoseInWorldSource
from planning.trajectory_sources import TrajectoryWithTimingInformationSource, DummyTrajSource
from planning.diffik import AddIiwaDifferentialIK
from planning.motion_planning import plan_path_custom, plan_drm
import sys
from planning.utils.csdecomp_path import CSDECOMP_PATH
sys.path.append(f'{CSDECOMP_PATH}/bazel-bin/csdecomp/src/pybind/pycsdecomp')
import pycsdecomp as csd

# from iiwa import IiwaHardwareStationDiagram

PETE_ASSETS = os.path.dirname(__file__) + "/../pete_assets/"
ONLINE_VOXEL_RADIUS = 0.005
SYS_ID_TRAJ_PARAMETER_PATH = Path(os.path.abspath(os.path.join(__file__ ,"../../traj_feb8")))

def get_csd_plant():
    parser = csd.URDFParser()
    parser.register_package("adaptive_decomp", PETE_ASSETS+"assets")
    parser.register_package("iiwa_description", PETE_ASSETS+"assets/iiwa")
    parser.register_package("wsg_description", PETE_ASSETS+"assets/wsg_description")
    parser.register_package("tri_finray_gripper", PETE_ASSETS+"assets/tri_finray_gripper")
    parser.parse_directives(PETE_ASSETS+"assets/directives/iiwa7_on_table.yaml")
    plant = parser.build_plant()

    return plant

def drm_planner(csd_plant, vox=None):
    drm_pl_opts = csd.DrmPlannerOptions()
    drm_pl_opts.max_number_planning_attempts = 50
    drm_pl_opts.try_shortcutting = True
    drm_pl_opts.online_edge_step_size = 0.005

    drm_planner = csd.DrmPlanner(csd_plant, drm_pl_opts)
    root = os.path.abspath(os.path.dirname(__file__)+'/../')
    drm_planner.LoadRoadmap(root+"/planning/roadmaps/iiwa_roadmap_0_0_0.01_0.2_100000_10_4.5_0.45.rm")

    if vox is not None:
        online_voxel_observation = csd.Voxels(vox.T)
        drm_planner.BuildCollisionSet(online_voxel_observation)
    
    return drm_planner

class TrajSourceInitializer(LeafSystem):
    """Prevents the robot from falling down uppon simulator creation."""

    def __init__(self, excitation_traj, use_custom_path_planner=False):
        super().__init__()
        self._excitation_traj = excitation_traj

        self._traj_source: TrajectorySource = None
        self._initialized = False

        self._iiwa_position_measured_input_port = self.DeclareVectorInputPort(
            "iiwa.position_measured", 7
        )

        # Create drm planner.
        self.csd_plant = get_csd_plant()
        self.drm_planner = drm_planner(self.csd_plant)
        self.use_custom_path_planner = use_custom_path_planner

        self.DeclareInitializationDiscreteUpdateEvent(self._init)

    def set_traj_source(self, traj_source):
        self._traj_source = traj_source

    def _init(self, context, discrete_values):
        if self._initialized:
            return

        assert self._traj_source is not None

        q_current = self._iiwa_position_measured_input_port.Eval(context)

        q_start = self._excitation_traj.value(0.0)
        if self.use_custom_path_planner:
            # Use custom path planner. Warning: Must be implemented by user
            to_start_traj = plan_path_custom(
                q_current, 
                q_start,
            )
        else:
            # Use drm path planner
            to_start_traj = plan_drm(
                self.drm_planner,
                start=q_current,
                goal=q_start,
                vox=None,
                online_voxel_radius=ONLINE_VOXEL_RADIUS
            )

        wait_at_start_traj = PiecewisePolynomial.ZeroOrderHold(
            breaks=[
                to_start_traj.end_time(),
                to_start_traj.end_time() + 2.0,
            ],
            samples=np.stack([q_start, q_start], axis=1),
        )

        excitation_traj_time = PiecewisePolynomial().FirstOrderHold(
            [0.0, self._excitation_traj.end_time()],
            [[0.0, self._excitation_traj.end_time()]],
        )
        excitation_traj_time.shiftRight(wait_at_start_traj.end_time())
        shifted_excitation_traj = PathParameterizedTrajectory(
            path=self._excitation_traj, time_scaling=excitation_traj_time
        )

        self.excitation_traj_start_time = shifted_excitation_traj.start_time()
        self.excitation_traj_end_time = shifted_excitation_traj.end_time()

        composite_traj = CompositeTrajectory([to_start_traj, wait_at_start_traj, shifted_excitation_traj])
        self._traj_source.UpdateTrajectory(composite_traj)

        print("Initialized traj source.")

    def reset(self):
        self._initialized = False


def start_scenario(
        meshcat,
        dirstr = "temp", 
        scenario_path="scenario_data_grasping.yml", 
        models_path="scenario_data_grasping_no_object.dmd.yaml", 
        gripper_model_path="",
        use_hardware=False,
        time_horizon=10.0,
        traj_parameter_path="",
        use_custom_path_planner=False,
    ):

    meshcat.ResetRenderMode()

    builder = DiagramBuilder()

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    filename = os.path.join(dir_path, os.path.join("scenario_datas", scenario_path))
    scenario = LoadScenario(filename=filename)
    # station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
    #     "station",
    #     IiwaHardwareStationDiagram(
    #         scenario=scenario, has_wsg=True, use_hardware=use_hardware
    #     ),
    # )
    directory_path = os.path.dirname(os.path.abspath(__file__))
    models_package = os.path.abspath(os.path.join(directory_path, "..", "models", "package.xml"))
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat, hardware=False, package_xmls=[models_package]))
    if use_hardware:
        scenario.plant_config.time_step = 5e-3 # Controller frequency
        external_station = builder.AddSystem(MakeHardwareStation(scenario, meshcat, hardware=True, package_xmls=[models_package]))
    plant = station.GetSubsystemByName("plant")
    # plant = station.get_plant()

    # initialize save directories
    if os.path.exists(dirstr):
        input(f"{dirstr} already exists. Press enter to delete and re-create. Ctr+c to stop.")
        shutil.rmtree(dirstr)
    os.makedirs(dirstr)

    # initialize point cloud output ports and save camera instrinsics
    if not use_hardware:
        camera0 = station.GetSubsystemByName("rgbd_sensor_camera0")
        camera1 = station.GetSubsystemByName("rgbd_sensor_camera1")
        camera2 = station.GetSubsystemByName("rgbd_sensor_camera2")

        camera0_pcd = builder.AddSystem(DepthImageToPointCloud(camera0.default_depth_render_camera().core().intrinsics()))
        camera1_pcd = builder.AddSystem(DepthImageToPointCloud(camera1.default_depth_render_camera().core().intrinsics()))
        camera2_pcd = builder.AddSystem(DepthImageToPointCloud(camera2.default_depth_render_camera().core().intrinsics()))
        builder.Connect(station.GetOutputPort("camera0.depth_image"), camera0_pcd.GetInputPort("depth_image"))
        builder.Connect(station.GetOutputPort("camera1.depth_image"), camera1_pcd.GetInputPort("depth_image"))
        builder.Connect(station.GetOutputPort("camera2.depth_image"), camera2_pcd.GetInputPort("depth_image"))

    else:
        camera0_pcd = builder.AddSystem(DepthImageToPointCloud(CameraInfo(848, 480, 600.165, 600.165, 429.152, 232.822)))
        camera1_pcd = builder.AddSystem(DepthImageToPointCloud(CameraInfo(848, 480, 626.633, 626.633, 432.041, 245.465)))
        camera2_pcd = builder.AddSystem(DepthImageToPointCloud(CameraInfo(848, 480, 596.492, 596.492, 416.694, 240.225)))

        builder.Connect(external_station.GetOutputPort("camera0.depth_image"), camera0_pcd.GetInputPort("depth_image"))
        builder.Connect(external_station.GetOutputPort("camera1.depth_image"), camera1_pcd.GetInputPort("depth_image"))
        builder.Connect(external_station.GetOutputPort("camera2.depth_image"), camera2_pcd.GetInputPort("depth_image"))

    
    if use_hardware:
        # from camera calibation
        # Front camera
        x_front_rgb = RigidTransform(np.loadtxt("/home/real2sim/calibrations/2_10_calibrations_aligned/front.txt"))

        # Back Right camera
        x_back_right_rgb = RigidTransform(np.loadtxt("/home/real2sim/calibrations/2_10_calibrations_aligned/back_right.txt"))

        # Back Left camera
        x_back_left_rgb = RigidTransform(np.loadtxt("/home/real2sim/calibrations/2_10_calibrations_aligned/back_left.txt"))

        # rgb calibration to depth calibration (from realsense specs)
        # Front camera
        x_depth_rgb_front = RigidTransform([[0.999986,      -0.000127587,   0.00531376, 0.015102],
                                            [0.000116105,   0.999998,       0.00216102, 6.44158e-05],
                                            [-0.00531402,   -0.00216038,    0.999984,   -0.000426644],
                                            [0,             0,              0,          1]])
        x_front_camera = x_front_rgb @ x_depth_rgb_front

        # Back Right camera
        x_depth_rgb_back_right = RigidTransform([[0.999968,  -0.00700185,   0.00399879,     0.015085],
                                                [ 0.00701494,      0.99997,  -0.00326805,  -2.1265e-05],
                                                [-0.00397579,   0.00329599,     0.999987, -0.000455872],
                                                [          0,            0,            0,            1]])
        x_back_right_camera = x_back_right_rgb @ x_depth_rgb_back_right

        # Back Left camera
        x_depth_rgb_back_left = RigidTransform([[0.999998, -0.000191981,  -0.00215977,    0.0150991],
                                                [0.000214442,     0.999946,    0.0104041,  7.71731e-05],
                                                [0.00215765,   -0.0104046,     0.999944, -0.000317806],
                                                [          0,            0,            0,            1]])
        x_back_left_camera = x_back_left_rgb @ x_depth_rgb_back_left

    else:
        # Front camera
        x_front_camera = RigidTransform(np.loadtxt("/home/real2sim/calibrations/2_10_calibrations_aligned/front.txt"))

        # Back Right camera
        x_back_right_camera = RigidTransform(np.loadtxt("/home/real2sim/calibrations/2_10_calibrations_aligned/back_right.txt"))

        # Back Left camera
        x_back_left_camera = RigidTransform(np.loadtxt("/home/real2sim/calibrations/2_10_calibrations_aligned/back_left.txt"))

    # connect stationary camera pcd source
    camera0_pose_source = builder.AddSystem(CameraPoseInWorldSource(x_front_camera, handeye=False))
    camera1_pose_source = builder.AddSystem(CameraPoseInWorldSource(x_back_right_camera, handeye=False))
    camera2_pose_source = builder.AddSystem(CameraPoseInWorldSource(x_back_left_camera, handeye=False))

    builder.Connect(
        camera0_pose_source.GetOutputPort("X_WC"),
        camera0_pcd.GetInputPort("camera_pose"),
    )

    builder.Connect(
        camera1_pose_source.GetOutputPort("X_WC"),
        camera1_pcd.GetInputPort("camera_pose"),
    )

    builder.Connect(
        camera2_pose_source.GetOutputPort("X_WC"),
        camera2_pcd.GetInputPort("camera_pose"),
    )

    controller_plant = station.GetSubsystemByName(
        "iiwa_controller_plant_pointer_system"
    ).get()

    planner = builder.AddSystem(
        TwoGraspPlanner(
            plant=plant, 
            controller_plant=controller_plant,
            X_WC0=x_front_camera,
            X_WC1=x_back_left_camera,
            X_WC2=x_back_right_camera,
            X_WC_bin=None,
            meshcat=meshcat,
            dirstr=dirstr,
            time_horizon=time_horizon,
            models_path=os.path.join(dir_path, os.path.join("scenario_datas", models_path)),
            gripper_model_path=gripper_model_path,
            sys_id_grasp_only=True,
            num_objs=1,
            use_custom_path_planner=use_custom_path_planner
        )
    )

    wsg_state_demux: Demultiplexer = builder.AddSystem(Demultiplexer(2, 1))
    if use_hardware:
        # Connect the output of external station to the input of internal station
        builder.Connect(
            external_station.GetOutputPort("iiwa.position_measured"),
            station.GetInputPort("iiwa.position"),
        )

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

    # Increase max force.
    wsg_force_source = builder.AddNamedSystem(
        "wsg_force_source", ConstantVectorSource([80.0]) # 80N is max
    )
    builder.Connect(
        wsg_force_source.get_output_port(), station.GetInputPort("wsg.force_limit")
    )

    # Set up logging for system ID.
    num_positions = 7
    measured_position_logger: VectorLogSink = builder.AddNamedSystem(
        "measured_position_logger",
        VectorLogSink(num_positions, publish_period=scenario.plant_config.time_step),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        measured_position_logger.get_input_port(),
    )
    measured_torque_logger: VectorLogSink = builder.AddNamedSystem(
        "measured_torque_logger",
        VectorLogSink(num_positions, publish_period=scenario.plant_config.time_step),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.torque_measured"),
        measured_torque_logger.get_input_port(),
    )

    # Set up differential inverse kinematics.
    velocity_limits = 0.4 * np.ones(7)
    acceleration_limits = 1.0 * np.ones(7)
    diff_ik = AddIiwaDifferentialIK(
        builder, 
        controller_plant, 
        frame=None,
        velocity_lims=velocity_limits, 
        acceleration_lims=acceleration_limits, # doesn't actually do anything since using this stops the robot from moving???
        joint_centering_gain=5.0
    )
    builder.Connect(planner.GetOutputPort("X_WG"), diff_ik.get_input_port(0))
    if use_hardware:
        builder.Connect(
            external_station.GetOutputPort("iiwa.state_estimated"),
            diff_ik.GetInputPort("robot_state"),
        )
    else:
        builder.Connect(
            station.GetOutputPort("iiwa.state_estimated"),
            diff_ik.GetInputPort("robot_state"),
        )
    builder.Connect(
        planner.GetOutputPort("reset_diff_ik"),
        diff_ik.GetInputPort("use_robot_state"),
    )

    # The DiffIK and the direct position-control modes go through a PortSwitch
    switch = builder.AddSystem(PortSwitch(7))
    builder.Connect(diff_ik.get_output_port(), switch.DeclareInputPort("diff_ik"))
    builder.Connect(
        joint_traj_source.get_output_port(),
        switch.DeclareInputPort("position"),
    )
    if use_hardware:
        builder.Connect(switch.get_output_port(), external_station.GetInputPort("iiwa.position"))
    else:
        builder.Connect(switch.get_output_port(), station.GetInputPort("iiwa.position"))
    builder.Connect(
        planner.GetOutputPort("control_mode"),
        switch.get_port_selector_input_port(),
    )

    builder.Connect(
        camera0_pcd.GetOutputPort("point_cloud"),
        planner.GetInputPort("cloud_front"),
    )
    builder.Connect(
        camera1_pcd.GetOutputPort("point_cloud"),
        planner.GetInputPort("cloud_back_left"),
    )
    builder.Connect(
        camera2_pcd.GetOutputPort("point_cloud"),
        planner.GetInputPort("cloud_back_right"),
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

    simulator.set_target_realtime_rate(1.0)

    meshcat.AddButton("Stop Simulation", "Escape")
    print("Press Escape to stop the simulation")
    while meshcat.GetButtonClicks("Stop Simulation") < 1 and not planner.doing_sys_id:
        simulator.AdvanceTo(simulator.get_context().get_time() + 1.0)

    builder = DiagramBuilder()
    scenario = LoadScenario(filename=filename)
    # Ensure correct timestep for position control mode.
    scenario.plant_config.time_step == 5e-3

    station: RobotDiagram = builder.AddNamedSystem(
        "hardware_station",
        MakeHardwareStation(
            scenario=scenario,
            meshcat=meshcat,
            hardware=use_hardware,
            package_xmls=[models_package]
        ),
    )

    # Load trajectory parameters
    is_fourier_series = os.path.exists(traj_parameter_path / "a_value.npy")
    if is_fourier_series:
        traj_attrs = FourierSeriesTrajectoryAttributes.load(traj_parameter_path)
        excitation_traj = FourierSeriesTrajectory(
            traj_attrs=traj_attrs,
            time_horizon=time_horizon,
        )
    else:
        raise ValueError(f"Invalid trajectory type: {traj_parameter_path}")

    # Placeholder trajectory
    traj_source: TrajectorySource = builder.AddNamedSystem(
        "trajectory_source",
        TrajectorySource(
            trajectory=PiecewisePolynomial.ZeroOrderHold(
                [0.0, 1.0], np.zeros((len(excitation_traj.value(0.0)), 2))
            ),
            output_derivative_order=0,
        ),
    )
    builder.Connect(
        traj_source.get_output_port(), station.GetInputPort("iiwa.position")
    )
    initializer: TrajSourceInitializer = builder.AddSystem(
        TrajSourceInitializer(excitation_traj, use_custom_path_planner)
    )
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        initializer.get_input_port(),
    )
    initializer.set_traj_source(traj_source)

    # Add a placeholder traj.
    wsg_traj_source: TrajectorySource = builder.AddNamedSystem(
        "wsg_trajectory_source",
        TrajectorySource(
            trajectory=PiecewisePolynomial.ZeroOrderHold([0.0, 1.0], np.zeros((1, 2))),
        ),
    )
    builder.Connect(
        wsg_traj_source.get_output_port(), station.GetInputPort("wsg.position")
    )

    # Add data loggers
    num_positions = station.GetInputPort("iiwa.position").size()
    logging_period = 1e-3
    measured_position_logger: VectorLogSink = builder.AddNamedSystem(
        "measured_position_logger",
        VectorLogSink(num_positions, publish_period=logging_period),
    )
    measured_torque_logger: VectorLogSink = builder.AddNamedSystem(
        "measured_torque_logger",
        VectorLogSink(num_positions, publish_period=logging_period),
    )
    measured_wsg_logger: VectorLogSink = builder.AddNamedSystem(
        "measured_wsg_logger",
        VectorLogSink(2, publish_period=logging_period),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        measured_position_logger.get_input_port(),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.torque_measured"),
        measured_torque_logger.get_input_port(),
    )
    builder.Connect(
        station.GetOutputPort("wsg.state_measured"),
        measured_wsg_logger.get_input_port(),
    )

    # Build and setup simulation
    diagram = builder.Build()

    wsg_traj_source.UpdateTrajectory(
        PiecewisePolynomial.ZeroOrderHold([0.0, 1.0],
        np.array([[0.0, 0.0]]))
    )
    print(f"Set wsg to position {0.0}")

    # Execute trajs.
    sys_id_simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    sys_id_simulator.Initialize()
    sys_id_simulator.set_target_realtime_rate(1.0)
    sys_id_simulator.AdvanceTo(initializer.excitation_traj_end_time + 1.0)

    # Save data
    measured_position_data = (
        measured_position_logger.FindLog(sys_id_simulator.get_context()).data().T
    )
    measured_torque_data = (
        measured_torque_logger.FindLog(sys_id_simulator.get_context()).data().T
    )
    measured_wsg_data = (
        measured_wsg_logger.FindLog(sys_id_simulator.get_context()).data().T
    )
    sample_times_s = measured_position_logger.FindLog(
        sys_id_simulator.get_context()
    ).sample_times()

    # Only keep data during excitation trajectory execution
    data_start_time = initializer.excitation_traj_start_time
    excitation_traj_end_time = initializer.excitation_traj_end_time
    excitation_traj_start_idx = np.argmax(sample_times_s >= data_start_time)
    excitation_traj_end_idx = np.argmax(
        sample_times_s >= excitation_traj_end_time
    )
    measured_position_data = measured_position_data[
        excitation_traj_start_idx:excitation_traj_end_idx
    ]
    measured_torque_data = measured_torque_data[
        excitation_traj_start_idx:excitation_traj_end_idx
    ]
    measured_wsg_data = measured_wsg_data[
        excitation_traj_start_idx:excitation_traj_end_idx
    ]
    sample_times_s = sample_times_s[
        excitation_traj_start_idx:excitation_traj_end_idx
    ]
    # Shift sample times to start at 0
    sample_times_s -= sample_times_s[0]

    # Remove duplicated samples
    _, unique_indices = np.unique(sample_times_s, return_index=True)
    if len(unique_indices) < len(sample_times_s):
        print(
            f"{len(unique_indices)} out of {len(sample_times_s)} data points "
            "are unique!"
        )
        measured_position_data = measured_position_data[unique_indices]
        measured_torque_data = measured_torque_data[unique_indices]
        measured_wsg_data = measured_wsg_data[unique_indices]
        sample_times_s = sample_times_s[unique_indices]

    # Save data
    system_id_savedir = os.path.join(dirstr, "system_id_data")
    np.save(
        os.path.join(system_id_savedir, "joint_positions.npy"),
        measured_position_data,
    )
    np.save(
        os.path.join(system_id_savedir, "joint_torques.npy"),
        measured_torque_data,
    )
    np.save(
        os.path.join(system_id_savedir, "wsg_positions.npy"),
        measured_wsg_data[:, 0],
    )
    np.save(
        os.path.join(system_id_savedir, "sample_times_s.npy"),
        sample_times_s,
    )

    print(
        f"Collected {len(sample_times_s)} data samples "
    )

    initializer.reset()

    planner.doing_sys_id = False
    while meshcat.GetButtonClicks("Stop Simulation") < 1 and not planner.done:
        simulator.AdvanceTo(simulator.get_context().get_time() + 1.0)


    meshcat.DeleteButton("Stop Simulation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario_path",
        default="scenario_data_grasping_hardware.yml",
        help="yaml file with scenario",
    )
    parser.add_argument(
        "--save_dir",
        default="temp",
        help="directory to save images in",
        nargs='?',
    )
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )
    parser.add_argument(
        "--time_horizon",
        type=float,
        default=10.0,
        help="The time horizon/ duration of the trajectory. Only used for Fourier "
        + "series trajectories.",
    )
    parser.add_argument(
        "--traj_parameter_path",
        type=Path,
        default=SYS_ID_TRAJ_PARAMETER_PATH,
        help="Path to the trajectory parameter folder. The folder must contain "
        + "'a_value.npy', 'b_value.npy', and 'q0_value.npy' or 'control_points.npy', "
        + "'knots.npy', and 'spline_order.npy'.",
    )
    parser.add_argument(
        "--use_custom_path_planner",
        action="store_true",
        help="Whether to use user implemented path planner.",
    )
    args = parser.parse_args()

    directory_path = os.path.dirname(os.path.abspath(__file__))
    gripper_model_path = "package://pickplace_data_collection/schunk_wsg_50_large_grippers_w_buffer.sdf"
    
    # Start the visualizer.
    meshcat = StartMeshcat()

    save_dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'sys_id_tests', args.save_dir))
    start_scenario(
        meshcat,
        save_dir_path, 
        scenario_path= args.scenario_path, 
        gripper_model_path=gripper_model_path,
        use_hardware=args.use_hardware,
        time_horizon=args.time_horizon,
        traj_parameter_path=args.traj_parameter_path,
        use_custom_path_planner=args.use_custom_path_planner
    )