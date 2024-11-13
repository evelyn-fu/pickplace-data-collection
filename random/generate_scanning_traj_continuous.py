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
from pydrake.geometry.optimization import (
    IrisOptions, 
    IrisInConfigurationSpace,
    LoadIrisRegionsYamlFile,
)
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
    savedir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'scanning_traj_continuous3'))

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
    q_top = [0.07368055462938518, 0.05530896177583773, -0.10634229062166067, -1.2588725332166288, 0.007703567486763913, 1.8268322666728602, 1.5304187858016194]
    q_back_left = [1.169654066480132, 0.30893919577771406, 0.4483862783720277, -2.0187414822238736, -0.723892255910587, 1.7621031681258894, 2.6614871227910735]
    q_back_right = [-1.2900862213017672, 0.29470243892151493, -0.3081726492589266, -2.0196267202918032, 0.7085820301181794, 1.7355730231045834, -2.628964934483996]
    q_left = [0.8368133658955407, 0.8143734336392138, 0.19208249948917935, -1.2146479770197423, -0.6783126668486744, 2.0374352481809517, 2.596435287950576]
    q_right = [-0.9041161427467723, 0.8079787763767405, -0.07241214960941236, -1.2146861343174455, 0.6269783191527105, 2.006281486796294, -2.5509098435585225]
    q_front_left = [0.8474033851221104, 0.9567105093538819, -0.2548647540604976, -0.9560565973091726, -0.41475963934323606, 2.09439510239, 2.436544250805479]
    q_front_right = [-0.847444042967782, 0.9567257952691892, 0.2549051611948841, -0.9560694576591842, 0.4148480265244533, 2.09439510239, -2.4364868300177367]
    q_front = [0, 0.7, -0.2, -1.1, 0.2, 1.75, 2.9]
    q_back_back_left = [0.8368133658955407, 0.8143734336392138, 0.19208249948917935, -1.2146479770197423, -0.6783126668486744, 2.0374352481809517, 2.596435287950576]
    q_back_back_right = [-0.9041161427467723, 0.8079787763767405, -0.07241214960941236, -1.2146861343174455, 0.6269783191527105, 2.006281486796294, -2.5509098435585225]
    q_top_left = [0.6262849237329107, 0.307410769068365, 0.056717958605464525, -1.393627869548887, -0.4174013108079192, 1.7728653830244612, 2.2324769105958406]
    q_top_right = [-0.48202318388431536, 0.3138245949660967, -0.2495415656052779, -1.3939909899646707, 0.4617628552816069, 1.791799833115145, 0.8656138505464283]

    
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

    regions_dict = LoadIrisRegionsYamlFile("../regions/gaze_constrained_scanning_regions_3.yaml")
    regions = list(regions_dict.values())

    # generate and save trajectories
    breakpoints = [0, 1, 3, 10, 11, 12]
    traj_names = ["to_prescan", "to1", "to2", "to3", "to_postscan", "to_home"]
    waypoints = [q_home, q_top, q_front, q_top_left, q_back_back_left, q_back_left, q_left, q_front_left,
     q_front_right, q_right, q_back_right, q_back_back_right, q_top, q_home]
    
    control_points = []
    start_times = []
    end_times = []
    time_offset = 0
    traj_ind = 0
    for waypt_ind in range(len(waypoints)-1):
        start = time.time()
        if waypt_ind == 0 or waypt_ind == len(waypoints)-2:
            traj = plan_unconstrained_gcs_path_start_to_goal(
                plant=controller_plant, q_start=waypoints[waypt_ind], q_goal=waypoints[waypt_ind+1], regions=None, no_obstacles=True
            )
        else:
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
            print("start_times", start_times)
            print("end_times", end_times)
            print(waypt_ind, savedir + "/" + traj_names[traj_ind])
            # Save this trajectory
            make_trajectory_save_dirs(savedir, traj_names[traj_ind])
            for i in range(len(start_times)):
                if i > 0:
                    if start_times[i] != end_times[i-1]:
                        print("ah fuck", start_times[i], end_times[i-1])
                np.save(savedir + "/" + traj_names[traj_ind] + f"/control_points/{i}.npy", control_points[i])
                np.save(savedir + "/" + traj_names[traj_ind] + f"/start_times/{i}.npy", start_times[i])
                np.save(savedir + "/" + traj_names[traj_ind] + f"/end_times/{i}.npy", end_times[i])

            # Reset
            time_offset = 0
            control_points = []
            start_times = []
            end_times = []
            traj_ind += 1

    print("Times:", times)