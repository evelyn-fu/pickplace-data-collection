##### How to import CCI ################

import sys
import os 
# append path to pycuci to system path
PYCUCI_ROOT = os.path.dirname(__file__) + "/../../" + "cuciv0" 
sys.path.append(PYCUCI_ROOT+'/bazel-bin/cuci/src/pybind/pycuci')
import pycuci as cci

################ How to build the plant #####################
import numpy as np
from pydrake.all import (StartMeshcat,
                         RobotDiagramBuilder,
                         LoadModelDirectives,
                         ProcessModelDirectives,
                         VisualizationConfig,
                         ApplyVisualizationConfig,
                         Rgba,
                         HPolyhedron)


PETE_ASSETS =  os.path.dirname(__file__)+"/../pete_assets/"
directives_file = PETE_ASSETS+'assets/directives/iiwa7_on_table.yaml'
meshcat = StartMeshcat()
builder = RobotDiagramBuilder()
plant = builder.plant()
scene_graph = builder.scene_graph()
parser = builder.parser()

parser.package_map().Add("adaptive_decomp", PETE_ASSETS+"assets")
parser.package_map().Add("iiwa_description", PETE_ASSETS+"assets/iiwa")
parser.package_map().Add("wsg_description", PETE_ASSETS+"assets/wsg_description")
parser.package_map().Add("tri_finray_gripper", PETE_ASSETS+"assets/tri_finray_gripper")

directives = LoadModelDirectives(directives_file)
models = ProcessModelDirectives(directives, plant, parser)
plant.Finalize()

config = VisualizationConfig()
config.enable_alpha_sliders = False
config.publish_contacts=False
config.publish_inertia = False
config.default_proximity_color = Rgba(0.8,0,0,0.2)
ApplyVisualizationConfig(config, builder.builder(), meshcat=meshcat)

diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(diagram_context)
scene_graph_context = scene_graph.GetMyContextFromRoot(diagram_context)
diagram.ForcedPublish(diagram_context)
robot_model_instances = [plant.GetModelInstanceByName(m) for m in ['iiwa7', 
                                                                'wsg', 
                                                                'finray']]

drake_objects = {
    'plant': plant,
    'plant_context': plant_context,
    'diagram': diagram,
    'diagram_context': diagram_context,
    'scene_graph': scene_graph,
    'scene_graph_context': scene_graph_context,
    'robot_model_instances': robot_model_instances
}

meshcat.SetProperty('/drake/proximity', "visible", True)



min_corner = np.array([-0.4, -1.25, 0.])
max_corner = np.array([1.1, 1.25, 1.])

ws_corners_online_voxels = np.array([
    [min_corner[0], min_corner[1], min_corner[2]],
    [min_corner[0], min_corner[1], max_corner[2]],
    [min_corner[0], max_corner[1], min_corner[2]],
    [min_corner[0], max_corner[1], max_corner[2]],
    [max_corner[0], min_corner[1], min_corner[2]],
    [max_corner[0], min_corner[1], max_corner[2]],
    [max_corner[0], max_corner[1], min_corner[2]],
    [max_corner[0], max_corner[1], max_corner[2]]
])

aux_info = {
    'min_corner': min_corner,
    'max_corner': max_corner,
    'ws_corners_online_voxels': ws_corners_online_voxels,
}

cci_parser = cci.URDFParser()
cci_parser.register_package("adaptive_decomp", PETE_ASSETS+"assets")
cci_parser.register_package("iiwa_description", PETE_ASSETS+"assets/iiwa")
cci_parser.register_package("wsg_description", PETE_ASSETS+"assets/wsg_description")
cci_parser.register_package("tri_finray_gripper", PETE_ASSETS+"assets/tri_finray_gripper")
cci_parser.parse_directives(PETE_ASSETS+"assets/directives/iiwa7_on_table.yaml")
cci_plant = cci_parser.build_plant()
cci_mplant = cci_plant.getMinimalPlant()
cci_domain = cci.HPolyhedron()
cci_domain.MakeBox(cci_plant.getPositionLowerLimits(), 
                cci_plant.getPositionUpperLimits())
cci_objects = {
    'cci_plant' : cci_plant,
    'cci_mplant' : cci_mplant,
    'cci_domain' : cci_domain
}


###### How to generate regions #####

cci_fei_opts = cci.FastEdgeInflationOptions()
cci_fei_opts.num_particles = 10000
cci_fei_opts.max_hyperplanes_per_iteration = 20
cci_fei_opts.epsilon = 0.005
cci_fei_opts.delta = 0.005
cci_fei_opts.max_iterations = 30
cci_fei_opts.mixing_steps = 60
cci_fei_opts.configurataon_margin = 0.01
cci_fei_opts.verbose = True


edge_inflator = cci.CudaEdgeInflator(cci_objects['cci_mplant'], 
                                     cci_objects['cci_plant'].getRobotGeometryIds(), 
                                     cci_fei_opts, 
                                     cci_objects['cci_domain'])

cci_region : cci.HPolyhedron = edge_inflator.inflateEdge(np.zeros(7).reshape(7,1), 
                                                         0.3*np.ones(7).reshape(7,1), 
                                                         cci.Voxels(np.array([0,0,100]).reshape(3,1)), 
                                                         0.01, 
                                                         verbose=True)

region : HPolyhedron = HPolyhedron(cci_region.A(), cci_region.b()) 

print(region.A())
