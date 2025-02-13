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
)

from robot_payload_id.control.trajectory import FourierSeriesTrajectory
from robot_payload_id.utils import FourierSeriesTrajectoryAttributes
from manipulation.station import MakeHardwareStation, RobotDiagram


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

    gripper_closed = 0.0
    gripper_open = 0.1
    gripper_positions = np.linspace(gripper_closed, gripper_open, num_gripper_openings)

    for gripper_position in gripper_positions:
        # Set the gripper position.
        wsg_traj_source.UpdateTrajectory(
            PiecewisePolynomial.ZeroOrderHold([0.0, 1.0], [gripper_position] * 2)
        )

        # TODO: Move to starting position.
        simulator = Simulator(diagram)
        ApplySimulatorConfig(scenario.simulator_config, simulator)
        simulator.set_target_realtime_rate(1.0)
        simulator.Initialize()
        # start_positions
        # traj_source.UpdateTrajectory()
        # simulator.AdvanceTo()

        # Excecute system ID trajectory.
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
