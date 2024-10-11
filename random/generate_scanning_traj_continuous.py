import os
import argparse
import pickle
import time
import numpy as np
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
from pydrake.trajectories import CompositeTrajectory

iris_options = IrisOptions(require_sample_point_is_contained=True)
use_existing_regions_as_obstacles = True

def get_seeded_region(models_path, q_nominal):
    global iris_options

    print("getting seeded region around", q_nominal)
    builder = RobotDiagramBuilder()
    plant = builder.plant()
    builder.parser().AddModels(models_path)
    diagram = builder.Build()

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, q_nominal)

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
    if not os.path.exists(dirstr):
        os.makedirs(dirstr)
        
    pkl_path = dirstr + '/scanning_traj_region.pkl'

    with open(pkl_path, 'wb') as f:
        pickle.dump(sets, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scenario_path",
        default="scenario_data_grasping.yml",
        help="yaml file with scenario",
        nargs='?',
    )
    parser.add_argument(
        "models_path",
        default="scenario_datas/scenario_data_grasping.dmd.yaml",
        help="dmd.yaml file with scenario, used for generating iris regions",
        nargs='?',
    )
    args = parser.parse_args()
    scenario_path = args.scenario_path
    models_path=args.models_path
    savedir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'scanning_traj_continuous'))

    # Start the visualizer.
    meshcat = StartMeshcat()

    builder = DiagramBuilder()

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    filename = os.path.join(dir_path, os.path.join("scenario_datas", scenario_path))
    scenario = LoadScenario(filename=filename)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat, hardware=False))
    controller_plant = station.GetSubsystemByName(
       "iiwa_controller_plant_pointer_system"
    ).get()
    
    # Build diagram
    diagram = builder.Build()

    q_home = [0.0, 0.4, 0.0, -1.2, 0.0, 1.0, -1.57]
    q_top = [0.07368055, 0.05530896, -0.10634229, -1.25887253, 0.00770357, 1.82683227, 1.53041879]
    q_back_left = [2.63832054, 1.12806438, 1.53517895, 2.03334719, 0.08600897, -1.36597991, -1.50074236]
    q_back_right = [0.50201402, -1.09220736, 1.58270881, 2.03354345, -0.05672917, -1.36275102, 1.46086455]
    q_left = [1.54171116, 1.2877689, 1.68350761, 1.21460623, 0.24698678, -1.85501355, -1.39106498]
    q_right = [1.60222014, -1.27009492, 1.43085245, 1.21461796, -0.22635522, -1.85061546, 1.37198868]
    q_front_left = [1.20058908, 1.32614569, 1.70315449, 0.95582837, 0.30928626, -2.06720313, -1.29576344]
    q_front_right = [1.94295434, -1.3132019, 1.41193298, 0.95584696, -0.29072372, -2.06312362, 1.28077475]
    q_front = [0, 0.7, -0.2, -1.1, 0.2, 1.75, 2.9]
    q_back_back_left = [2.5, 1.3, 1.8, 1.8, 0.2, -1.5, -1.4]
    q_back_back_right = [0.64, -1.3, 1.34, 1.8, -0.2, -1.5, 1.3]
    
    times = []
    # # make regions
    # start = time.time()
    # regions = []
    # regions.append(get_seeded_region(models_path, q_home))
    # regions.append(get_seeded_region(models_path, q_top))
    # regions.append(get_seeded_region(models_path, q_back_back_left))
    # regions.append(get_seeded_region(models_path, q_back_back_right))
    # regions.append(get_seeded_region(models_path, q_back_left))
    # regions.append(get_seeded_region(models_path, q_back_right))
    # regions.append(get_seeded_region(models_path, q_left))
    # regions.append(get_seeded_region(models_path, q_right))
    # regions.append(get_seeded_region(models_path, q_front_left))
    # regions.append(get_seeded_region(models_path, q_front_right))
    # regions.append(get_seeded_region(models_path, q_front))
    # times.append(time.time() - start)
    # print(f"Regions generated in {times[-1]} seconds")
    # save_regions_pkl(regions, savedir)

    with open(savedir + '/scanning_traj_region.pkl', 'rb') as f:
        regions = pickle.load(f)

    # generate and save trajectories
    breakpoints = [1, 5, 9, 11]
    traj_names = ["to1", "to2", "to3", "to_home"]
    waypoints = [q_home, q_top, q_back_back_left, q_back_left, q_left, q_front_left, 
                q_front, q_front_right, q_right, q_back_right, q_back_back_right, q_top, q_home]
    
    control_points = []
    start_times = []
    end_times = []
    time_offset = 0
    traj_ind = 0
    for waypt_ind in range(len(waypoints)-1):
        start = time.time()
        traj = plan_unconstrained_gcs_path_start_to_goal(
            plant=controller_plant, q_start=waypoints[waypt_ind], q_goal=waypoints[waypt_ind+1], regions=regions, no_obstacles=False
        )
        times.append(time.time() - start)
        print(f"Traj {waypt_ind} generated in {times[-1]} seconds")
        
        for i in range(traj.get_number_of_segments()):
            segment = traj.segment(i)
            control_points.append(segment.control_points())
            start_times.append(segment.start_time() + time_offset)
            end_times.append(segment.end_time() + time_offset)
        
        time_offset = end_times[-1]
        
        # Start a new composite trajectory
        if waypt_ind in breakpoints:
            print(waypt_ind, savedir + "/" + traj_names[traj_ind])
            # Save this trajectory
            make_trajectory_save_dirs(savedir, traj_names[traj_ind])
            for i in range(len(start_times)):
                np.save(savedir + "/" + traj_names[traj_ind] + f"/control_points/{i}.npy", control_points[i])
                np.save(savedir + "/" + traj_names[traj_ind] + f"/start_times/{i}.npy", start_times[i])
                np.save(savedir + "/" + traj_names[traj_ind] + f"/end_times/{i}.npy", end_times[i])

            # Reset
            print(start_times, end_times)
            time_offset = 0
            control_points = []
            start_times = []
            end_times = []
            traj_ind += 1

    print("Times:", times)