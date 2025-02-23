import pydrake.planning as mut
from pydrake.common import RandomGenerator, use_native_cpp_logging
from pydrake.planning import (RobotDiagramBuilder,
                              SceneGraphCollisionChecker)
from pydrake.solvers import MosekSolver, GurobiSolver
import os
import pickle
import time
import numpy as np
from pydrake.all import (
    StartMeshcat, MeshcatVisualizerParams, MeshcatVisualizer, LoadModelDirectives, ProcessModelDirectives
)

PETE_ASSETS = os.path.abspath(os.path.dirname(__file__)+"/../pete_assets/")

def get_regions():
    meshcat = StartMeshcat()
    use_native_cpp_logging()
    params = dict(edge_step_size=0.125)
    builder = RobotDiagramBuilder()
    directives_file = PETE_ASSETS+'/assets/directives/iiwa7_on_table_with_extra_cameras.yaml'
    builder.parser().package_map().Add("adaptive_decomp", PETE_ASSETS+"/assets")
    builder.parser().package_map().Add("iiwa_description", PETE_ASSETS+"/assets/iiwa")
    builder.parser().package_map().Add("wsg_description", PETE_ASSETS+"/assets/wsg_description")
    builder.parser().package_map().Add("tri_finray_gripper", PETE_ASSETS+"/assets/tri_finray_gripper")
    directives = LoadModelDirectives(directives_file)
    models = ProcessModelDirectives(directives, builder.plant(), builder.parser())
    iiwa_model_instance_index = builder.plant().GetModelInstanceByName("iiwa7")
    wsg_model_instance_index = builder.plant().GetModelInstanceByName("wsg")
    plant = builder.plant()

    viz_params = MeshcatVisualizerParams()
    viz_params.prefix = "planning"
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder.builder(), builder.scene_graph(), meshcat, viz_params
    )

    params["robot_model_instances"] = [iiwa_model_instance_index, wsg_model_instance_index]
    params["model"] = builder.Build()
    checker = SceneGraphCollisionChecker(**params)

    diagram_context = params["model"].CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)
    pos = plant.GetPositions(plant_context)
    pos[0] = -np.pi/2
    plant.SetPositions(plant_context, np.zeros(7))
    params["model"].ForcedPublish(diagram_context)

    while True:
        pass

    options = mut.IrisFromCliqueCoverOptions()
    options.num_points_per_coverage_check = 5000
    options.num_points_per_visibility_round = 1000
    options.minimum_clique_size = 16
    options.coverage_termination_threshold = 0.7

    generator = RandomGenerator(0)

    # if (MosekSolver().available() and MosekSolver().enabled()) or (
    #         GurobiSolver().available() and GurobiSolver().enabled()):
    #     # We need a MIP solver to be available to run this method.
    #     sets = mut.IrisInConfigurationSpaceFromCliqueCover(
    #         checker=checker, options=options, generator=generator,
    #         sets=[]
    #     )

    #     if len(sets) < 1:
    #         raise("No regions found")

    #     return sets
    # else:
    #     print("No solvers available")

def save_regions_pkl(sets, dirstr):
    pkl_path = dirstr + '/safe_region.pkl'

    with open(pkl_path, 'wb') as f:
        pickle.dump(sets, f)

if __name__ == "__main__":
    savedir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
    start = time.time()
    regions = get_regions()
    print(f"Regions generated in {time.time() - start} seconds")
    save_regions_pkl(regions, savedir)