import os
import argparse
import pickle
import time
from manipulation.station import MakeHardwareStation, LoadScenario
from planning.gcs import plan_unconstrained_gcs_path_start_to_goal
from iiwa_setup_dataclasses.bspline_trajectory import CompositeBezierCurveTrajectoryAttributes

import pydrake.planning as mut
from pydrake.common import RandomGenerator, use_native_cpp_logging
from pydrake.planning import (RobotDiagramBuilder,
                              SceneGraphCollisionChecker)
from pydrake.solvers import MosekSolver, GurobiSolver
from pydrake.geometry.optimization import IrisOptions, IrisInConfigurationSpace
from pydrake.geometry import (
    StartMeshcat,
)
from pydrake.systems.framework import DiagramBuilder


def get_seeded_region(models_path, q_nominal):
    print("getting seeded region around", q_nominal)
    builder = RobotDiagramBuilder()
    plant = builder.plant()
    builder.parser().AddModels(models_path)
    diagram = builder.Build()

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, q_nominal)

    iris_options = IrisOptions(require_sample_point_is_contained=True)
    region = IrisInConfigurationSpace(plant, plant_context, iris_options)
    print("region:", region)

    return region


def get_regions(models_path):

    use_native_cpp_logging()
    params = dict(edge_step_size=0.125)
    builder = RobotDiagramBuilder()
    builder.parser().AddModels(models_path)
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

        return sets
    else:
        print("No solvers available")

def make_trajectory_save_dirs(dirstr, traj_name):
    if not os.path.exists(dirstr+f"/{traj_name}"):
        os.makedirs(dirstr+f"/{traj_name}")
    if not os.path.exists(dirstr+f"/{traj_name}" + "/control_points"):
        os.makedirs(dirstr+f"/{traj_name}" + "/control_points")
    if not os.path.exists(dirstr+f"/{traj_name}" + "/start_times"):
        os.makedirs(dirstr+f"/{traj_name}" + "/start_times")
    if not os.path.exists(dirstr+f"/{traj_name}" + "/end_times"):
        os.makedirs(dirstr+f"/{traj_name}" + "/end_times")


def save_regions_pkl(sets, dirstr):
    pkl_path = dirstr + '/scanning_traj_region.pkl'

    with open(pkl_path, 'wb') as f:
        pickle.dump(sets, f)

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
        help="dmd.yaml file with scenario, used for generating iris regions",
        nargs='?',
    )
    args = parser.parse_args()
    scenario_path = args.scenario_path
    models_path=args.models_path
    savedir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'scanning_traj'))

    # Start the visualizer.
    meshcat = StartMeshcat()

    builder = DiagramBuilder()

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    filename = os.path.join(dir_path, os.path.join("scenario_datas", scenario_path))
    scenario = LoadScenario(filename=filename)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat, hardware=False))
    controller_plant = station.GetSubsystemByName(
        "iiwa.controller"
    ).get_multibody_plant_for_control()
    
    # Build diagram
    diagram = builder.Build()

    q_home = [0.0, 0.4, 0.0, -1.2, 0.0, 1.0, -1.57]
    q_1 = [0, 0.7, -0.2, -1.1, 0.2, 1.75, 2.9]
    q_2 = [2.5, 1.3, 1.8, 1.8, 0.2, -1.5, -1.4]
    q_3 = [0.64, -1.3, 1.34, 1.8, -0.2, -1.5, 1.3]
    
    times = []
    # make regions
    start = time.time()
    regions = get_regions(models_path)
    regions.append(get_seeded_region(models_path, q_home))
    regions.append(get_seeded_region(models_path, q_1))
    regions.append(get_seeded_region(models_path, q_2))
    regions.append(get_seeded_region(models_path, q_3))
    times.append(time.time() - start)
    print(f"Regions generated in {times[-1]} seconds")
    save_regions_pkl(savedir)

    # generate and save trajectories
    make_trajectory_save_dirs(savedir, "to1")
    start = time.time()
    traj1 = plan_unconstrained_gcs_path_start_to_goal(
        plant=controller_plant, q_start=q_home, q_goal=q_1, regions=regions, no_obstacles=False
    )
    times.append(time.time() - start)
    print(f"To 1 traj generated in {times[-1]} seconds")
    traj_attr = CompositeBezierCurveTrajectoryAttributes.from_composite_bezier_curve_trajectory(traj1)
    traj_attr.log(savedir + "/to1/")

    make_trajectory_save_dirs(savedir, "to2")
    start = time.time()
    traj2 = plan_unconstrained_gcs_path_start_to_goal(
        plant=controller_plant, q_start=q_1, q_goal=q_2, regions=regions, no_obstacles=False
    )
    times.append(time.time() - start)
    print(f"To 2 traj generated in {times[-1]} seconds")
    traj_attr = CompositeBezierCurveTrajectoryAttributes.from_composite_bezier_curve_trajectory(traj2)
    traj_attr.log(savedir + "/to2/")

    make_trajectory_save_dirs(savedir, "to3")
    start = time.time()
    traj3 = plan_unconstrained_gcs_path_start_to_goal(
        plant=controller_plant, q_start=q_2, q_goal=q_3, regions=regions, no_obstacles=False
    )
    times.append(time.time() - start)
    print(f"To 3 traj generated in {times[-1]} seconds")
    traj_attr = CompositeBezierCurveTrajectoryAttributes.from_composite_bezier_curve_trajectory(traj3)
    traj_attr.log(savedir + "/to3/")

    make_trajectory_save_dirs(savedir, "to_home")
    start = time.time()
    traj4 = plan_unconstrained_gcs_path_start_to_goal(
        plant=controller_plant, q_start=q_3, q_goal=q_home, regions=regions, no_obstacles=False
    )
    times.append(time.time() - start)
    print(f"To home traj generated in {times[-1]} seconds")
    traj_attr = CompositeBezierCurveTrajectoryAttributes.from_composite_bezier_curve_trajectory(traj4)
    traj_attr.log(savedir + "/to_home/")

    print("Times:", times)