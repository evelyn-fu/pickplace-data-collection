"""
This script collects joint position and torque data at multiple gripper openings. The
data is collected multiple times at each gripper opening.

The output is saved to a directory with the gripper position as the subdirectory name
with the following format:

<save_data_path>/gripper_position_<gripper_position>/
    run_<run_idx>/
        joint_positions.npy
        joint_torques.npy
        sample_times_s.npy
"""

import argparse
import logging
import os

from pathlib import Path

import numpy as np

from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    DiagramBuilder,
    PiecewisePolynomial,
    Simulator,
    TrajectorySource,
    VectorLogSink,
    StartMeshcat,
    LeafSystem,CompositeTrajectory,PathParameterizedTrajectory
)
from tqdm import tqdm
import shutil
from robot_payload_id.control.trajectory import FourierSeriesTrajectory
from robot_payload_id.utils import FourierSeriesTrajectoryAttributes
from manipulation.station import MakeHardwareStation, RobotDiagram

from planning.motion_planning import plan_path_custom, plan_drm

import sys
from planning.utils.csdecomp_path import CSDECOMP_PATH
sys.path.append(f'{CSDECOMP_PATH}/bazel-bin/csdecomp/src/pybind/pycsdecomp')
import pycsdecomp as csd

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario_path",
        type=str,
        default="scenario_datas/scenario_data_grasping_hardware.yml",
        help="Path to the scenario file. This must contain an iiwa model named 'iiwa' "
        "and a gripper model named 'wsg'.",
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
        "--save_data_path",
        type=Path,
        required=True,
        help="Path to save the data to. Each data collection run will be saved to a "
        + "separate subdirectory of this path.",
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
        "--num_gripper_openings",
        type=int,
        default=5,
        help="The number of gripper openings to collect data at.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=10,
        help="The number of times to collect data at each gripper opening.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level.",
    )
    parser.add_argument(
        "--use_custom_path_planner",
        action="store_true",
        help="Whether to use user implemented path planner.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    scenario_path = args.scenario_path
    traj_parameter_path = args.traj_parameter_path
    save_data_path = args.save_data_path
    use_hardware = args.use_hardware
    time_horizon = args.time_horizon
    num_gripper_openings = args.num_gripper_openings
    num_runs = args.num_runs
    use_custom_path_planner = args.use_custom_path_planner

    if os.path.exists(save_data_path):
        input(f"{save_data_path} already exists. Press enter to delete and re-create. Ctr+c to stop.")
        shutil.rmtree(save_data_path)

    builder = DiagramBuilder()
    scenario = LoadScenario(filename=scenario_path)
    # Ensure correct timestep for position control mode.
    scenario.plant_config.time_step == 5e-3

    meshcat = StartMeshcat()

    directory_path = os.path.dirname(os.path.abspath(__file__))
    models_package = os.path.abspath(os.path.join(directory_path, "..", "models", "package.xml"))
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
    start_positions = excitation_traj.value(0.0)

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
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        measured_position_logger.get_input_port(),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.torque_measured"),
        measured_torque_logger.get_input_port(),
    )

    # Build and setup simulation
    diagram = builder.Build()

    gripper_closed = 0.0
    gripper_open = 0.1
    gripper_positions = np.linspace(gripper_closed, gripper_open, num_gripper_openings)

    for gripper_position in tqdm(gripper_positions):
        wsg_traj_source.UpdateTrajectory(
            PiecewisePolynomial.ZeroOrderHold([0.0, 1.0],
            np.array([[gripper_position, gripper_position]]))
        )
        print(f"Set wsg to position {gripper_position}")

        for run_idx in range(num_runs):
            # Execute trajs.
            simulator = Simulator(diagram)
            ApplySimulatorConfig(scenario.simulator_config, simulator)
            simulator.Initialize()
            simulator.set_target_realtime_rate(1.0)
            simulator.AdvanceTo(initializer.excitation_traj_end_time + 1.0)

            # Save data
            measured_position_data = (
                measured_position_logger.FindLog(simulator.get_context()).data().T
            )
            measured_torque_data = (
                measured_torque_logger.FindLog(simulator.get_context()).data().T
            )
            sample_times_s = measured_position_logger.FindLog(
                simulator.get_context()
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
                sample_times_s = sample_times_s[unique_indices]

            # Save data
            output_dir = (
                save_data_path
                / f"gripper_position_{gripper_position:.2f}"
                / f"run_{run_idx}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_dir / "joint_positions.npy", measured_position_data)
            np.save(output_dir / "joint_torques.npy", measured_torque_data)
            np.save(output_dir / "sample_times_s.npy", sample_times_s)

            print(
                f"Collected {len(sample_times_s)} data samples at gripper position "
                f"{gripper_position:.2f} and run {run_idx}."
            )

            initializer.reset()


if __name__ == "__main__":
    main()
