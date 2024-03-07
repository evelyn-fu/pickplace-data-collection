import argparse
import pickle

def start_scenario(
        dirstr = "temp", 
        scenario_path="scenario_data_grasping.yml", 
        trajectories = []
    ):
    
    meshcat.ResetRenderMode()

    builder = DiagramBuilder()

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    filename = os.path.join(dir_path, os.path.join("scenario_datas", scenario_path))
    scenario = LoadScenario(filename=filename)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat, hardware=False))
    if use_hardware:
        external_station = builder.AddSystem(MakeHardwareStation(scenario, meshcat, hardware=True))
    plant = station.GetSubsystemByName("plant")

    planner = builder.AddSystem(TrajectoryPlayer(trajectories))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scenario_path",
        default="scenario_data_grasping.yml",
        help="yaml file with scenario",
    )
    parser.add_argument(
        "trajectories_path",
        help="pkl file with trajectories",
    )
    parser.add_argument(
        "save_dir",
        default="temp",
        help="directory to save images in",
        nargs='?',
    )
    args = parser.parse_args()


    with open(args.trajectories_path, 'rb') as f:
        trajectories = pickle.load(f)

        start_scenario(args.scenario_path, trajectories, args.save_dir)