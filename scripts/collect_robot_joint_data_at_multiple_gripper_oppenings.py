import argparse
import logging
import os

from pathlib import Path

import numpy as np
import pycuci as cci

from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    DiagramBuilder,
    PiecewisePolynomial,
    Simulator,
    TrajectorySource,
    VectorLogSink,
    StartMeshcat,
)

from robot_payload_id.control.trajectory import FourierSeriesTrajectory
from robot_payload_id.utils import FourierSeriesTrajectoryAttributes
from manipulation.station import MakeHardwareStation, RobotDiagram
from mmt_gcs.planning.mintime_scs import MintimeSCSWithPathFixing
from mmt_gcs.planning.corridor_planning_utils import CCICollisionChecker
from mmt_gcs.planning.region_generation import CCI_inflate_edges_given_pwl_path

PETE_ASSETS = os.path.dirname(__file__) + "/../pete_assets/"
MMT_GCS_ROOT = os.path.abspath(os.path.join(__file__, "../../../mmt_gcs/"))
ONLINE_VOXEL_RADIUS = 0.005


def get_cci_edge_inflator(verbose=False):
    cci_parser = cci.URDFParser()
    cci_parser.register_package("adaptive_decomp", PETE_ASSETS + "assets")
    cci_parser.register_package("iiwa_description", PETE_ASSETS + "assets/iiwa")
    cci_parser.register_package(
        "wsg_description", PETE_ASSETS + "assets/wsg_description"
    )
    cci_parser.register_package(
        "tri_finray_gripper", PETE_ASSETS + "assets/tri_finray_gripper"
    )
    cci_parser.parse_directives(PETE_ASSETS + "assets/directives/iiwa7_on_table.yaml")
    cci_plant = cci_parser.build_plant()
    cci_mplant = cci_plant.getMinimalPlant()
    cci_domain = cci.HPolyhedron()
    cci_domain.MakeBox(
        cci_plant.getPositionLowerLimits(), cci_plant.getPositionUpperLimits()
    )
    cci_objects = {
        "cci_plant": cci_plant,
        "cci_mplant": cci_mplant,
        "cci_domain": cci_domain,
    }

    cci_fei_opts = cci.FastEdgeInflationOptions()
    cci_fei_opts.num_particles = 10000
    cci_fei_opts.max_hyperplanes_per_iteration = 20
    cci_fei_opts.epsilon = 0.005
    cci_fei_opts.delta = 0.005
    cci_fei_opts.max_iterations = 30
    cci_fei_opts.mixing_steps = 60
    cci_fei_opts.configurataon_margin = 0.01
    cci_fei_opts.verbose = verbose

    edge_inflator = cci.CudaEdgeInflator(
        cci_objects["cci_mplant"],
        cci_objects["cci_plant"].getRobotGeometryIds(),
        cci_fei_opts,
        cci_objects["cci_domain"],
    )

    return edge_inflator, cci_objects


def get_drm_planner(cci_obj, vox=None):
    drm_pl_opts = cci.DrmPlannerOptions()
    drm_pl_opts.max_number_planning_attempts = 50
    drm_pl_opts.try_shortcutting = True
    drm_pl_opts.online_edge_step_size = 0.005

    drm_planner = cci.DrmPlanner(cci_obj["cci_plant"], drm_pl_opts)
    drm_planner.LoadRoadmap(
        MMT_GCS_ROOT
        + "/tmp/iiwa_hardware/iiwa_roadmap_1_0_0.01_0.2_50000_10_4.5_0.45.rm"
    )

    if vox is not None:
        online_voxel_observation = cci.Voxels(vox.T)
        drm_planner.BuildCollisionSet(online_voxel_observation)

    return drm_planner


def scs_trajopt(
    start, goal, drm_planner, cci_obj, edge_inflator, vox, vel_limits, acc_limits
):
    online_voxel_observation = cci.Voxels(vox)

    success, pwl_plan = drm_planner.Plan(
        start, goal, online_voxel_observation, ONLINE_VOXEL_RADIUS
    )

    regions, edges = CCI_inflate_edges_given_pwl_path(
        pwl_plan,
        edge_inflator,
        online_voxel_observation,
        ONLINE_VOXEL_RADIUS,
        verbose=True,
    )

    cci_checker = CCICollisionChecker(
        cci_obj["cci_mplant"],
        cci_obj["cci_plant"].getRobotGeometryIds(),
        online_voxel_observation,
        ONLINE_VOXEL_RADIUS,
    )

    vel_limits_reflected = [-vel_limits, vel_limits]
    acc_limits_reflected = [-acc_limits, acc_limits]
    traj, cost, timing_info, traj_col_free, first_solve_collision_free, collisions = (
        MintimeSCSWithPathFixing(
            start,
            goal,
            regions,
            edges,
            vel_limits_reflected,
            acc_limits_reflected,
            cci_checker,
            edge_inflator,
            online_voxel_observation,
            ONLINE_VOXEL_RADIUS,
        )
    )

    return traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario_path",
        type=str,
        required=True,
        help="Path to the scenario file. This must contain an iiwa model named 'iiwa' "
        "and a gripper model named 'wsg'.",
    )
    parser.add_argument(
        "--traj_parameter_path",
        type=Path,
        required=True,
        help="Path to the trajectory parameter folder. The folder must contain "
        + "'a_value.npy', 'b_value.npy', and 'q0_value.npy' or 'control_points.npy', "
        + "'knots.npy', and 'spline_order.npy'.",
    )
    parser.add_argument(
        "--save_data_path",
        type=Path,
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
        "--log_level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    scenario_path = args.scenario_path
    traj_parameter_path = args.traj_parameter_path
    save_data_path = args.save_data_path
    use_hardware = args.use_hardware
    time_horizon = args.time_horizon
    num_gripper_openings = args.num_gripper_openings

    builder = DiagramBuilder()
    scenario = LoadScenario(filename=scenario_path)
    assert (
        scenario.plant_config.time_step == 5e-3
    ), "Invalid time-step for position control mode."

    meshcat = StartMeshcat()

    station: RobotDiagram = builder.AddNamedSystem(
        "hardware_station",
        MakeHardwareStation(
            scenario=scenario,
            meshcat=meshcat,
            use_hardware=use_hardware,
            package_xmls=[os.path.abspath("models/package.xml")],
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
    logging_period = scenario.plant_config.time_step
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

    # Create cci planner.
    edge_inflator, cci_objects = get_cci_edge_inflator()
    drm_planner = get_drm_planner(cci_objects)

    gripper_closed = 0.0
    gripper_open = 0.1
    gripper_positions = np.linspace(gripper_closed, gripper_open, num_gripper_openings)

    for gripper_position in gripper_positions:
        # Move to starting position
        simulator = Simulator(diagram)
        ApplySimulatorConfig(scenario.simulator_config, simulator)
        simulator.set_target_realtime_rate(1.0)
        simulator.Initialize()
        current_positions = station.GetOutputPort("iiwa.position_measured").Eval(
            simulator.get_context()
        )
        traj = scs_trajopt(
            start=current_positions,
            goal=start_positions,
            drm_planner=drm_planner,
            cci_obj=cci_objects,
            edge_inflator=edge_inflator,
            vox=cci.Voxels(),
            vel_limits=np.ones(num_positions),
            acc_limits=np.ones(num_positions),
        )
        traj_source.UpdateTrajectory(traj)
        simulator.AdvanceTo(traj.end_time() + 1.0)

        # Excecute system ID trajectory
        simulator = Simulator(diagram)
        ApplySimulatorConfig(scenario.simulator_config, simulator)
        simulator.set_target_realtime_rate(1.0)
        simulator.Initialize()
        traj_source.UpdateTrajectory(excitation_traj)
        simulator.AdvanceTo(excitation_traj.end_time() + 1.0)

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
        data_start_time = 0.0
        excitation_traj_end_time = excitation_traj.end_time()
        excitation_traj_start_idx = np.argmax(sample_times_s >= data_start_time)
        excitation_traj_end_idx = np.argmax(sample_times_s >= excitation_traj_end_time)
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
        output_dir = save_data_path / f"gripper_position_{gripper_position:.2f}"
        np.save(output_dir / "joint_positions.npy", measured_position_data)
        np.save(output_dir / "joint_torques.npy", measured_torque_data)
        np.save(output_dir / "sample_times_s.npy", sample_times_s)

        print(
            f"Collected {len(sample_times_s)} data samples at gripper position "
            f"{gripper_position:.2f}."
        )


if __name__ == "__main__":
    main()
