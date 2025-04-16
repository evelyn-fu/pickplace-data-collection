from pydrake.all import (RobotDiagramBuilder,
                         LoadModelDirectives,
                         ProcessModelDirectives,
                         AddDefaultVisualization,
                         ApplyVisualizationConfig,
                         VisualizationConfig,
                         Rgba, 
                         RigidTransform,
                         SceneGraphCollisionChecker,
                         StartMeshcat,
                         IrisZo, 
                         IrisZoOptions, 
                         IrisInConfigurationSpace, 
                         IrisOptions,
                         HPolyhedron,
                         Hyperellipsoid,
                         Sphere,
                         RotationMatrix)
import time
import numpy as np
import yaml
import os

# from cspace_utils.plotting import plot_points, plot_triad
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import networkx as nx

num_samples_to_collect = 5.0*1e5
roadmap_size = 100000
batchsize = 5000
store_edges = False
max_neighbors = 10
max_configuration_distance = 4.5
max_task_space_distance = 0.45
offline_voxel_resolution = 0.03
use_kmeans = False
edge_step_size = 0.01
edge_storage_step_size = 0.2

rm_name = f'iiwa_roadmap_{1 if use_kmeans else 0}_{1 if store_edges else 0}_{edge_step_size}\
_{edge_storage_step_size}_{roadmap_size}_{max_neighbors}_{max_configuration_distance}_\
{max_task_space_distance}.rm'

def stretch_array_to_3d(arr, val=0.):
    if arr.shape[0] < 3:
        arr = np.append(arr, val * np.ones((3 - arr.shape[0])))
    return arr

def plot_point(point, meshcat_instance, name,
               color=Rgba(0.06, 0.0, 0, 1), radius=0.01):
    meshcat_instance.SetObject(name,
                               Sphere(radius),
                               color)
    meshcat_instance.SetTransform(name, RigidTransform(
        RotationMatrix(), stretch_array_to_3d(point)))

def plot_points(meshcat, points, name, size = 0.05, color = Rgba(0.06, 0.0, 0, 1)):
    if isinstance(color , list):
        for i, pt in enumerate(points):
            n_i = name+f"/pt{i}"
            plot_point(pt, meshcat, n_i, color = color[i], radius=size)
    else:
        for i, pt in enumerate(points):
            n_i = name+f"/pt{i}"
            plot_point(pt, meshcat, n_i, color = color, radius=size)
        

def write_graph_summary(g: nx.Graph):
    summary = []
    summary.append(f"Graph Summary for Dynamic Roadmap")
    summary.append(f"=====================================")
    
    # Basic graph information
    summary.append(f"Number of nodes: {g.number_of_nodes()}")
    summary.append(f"Number of edges: {g.number_of_edges()}")
    
    # Connected components
    components = list(nx.connected_components(g))
    summary.append(f"Number of connected components: {len(components)}")
    
    # Largest component details
    largest_component = max(components, key=len)
    summary.append(f"Largest component size: {len(largest_component)} nodes")
    
    # Degree information
    degrees = [d for n, d in g.degree()]
    avg_degree = np.mean(degrees)
    max_degree = max(degrees)
    min_degree = min(degrees)
    summary.append(f"Average node degree: {avg_degree:.2f}")
    summary.append(f"Maximum node degree: {max_degree}")
    summary.append(f"Minimum node degree: {min_degree}")
    # cut_vertices = list(nx.articulation_points(g))
    # summary.append(f"Number of articulation points (potential bottlenecks): {len(cut_vertices)}")
    # Isolated nodes
    isolated_nodes = list(nx.isolates(g))
    summary.append(f"Number of isolated nodes: {len(isolated_nodes)}")
    component_sizes = [len(c) for c in components]
    summary.append("Component size distribution:")
    summary.append(f"  Min: {min(component_sizes)}")
    summary.append(f"  Max: {max(component_sizes)}")
    summary.append(f"  Mean: {np.mean(component_sizes):.2f}")
    summary.append(f"  Median: {np.median(component_sizes):.2f}")
    print("\n".join(summary))



def select_subset_kmeans(data, n_samples, batch_size=5000, max_iter=20):
    """
    Select a subset of samples using k-means clustering.
    
    Args:
        data: numpy array of shape (dimensions, num_samples)
        n_samples: desired number of samples in subset
    
    Returns:
        indices of selected samples
    """
    # Transpose to (num_samples, dimensions) for sklearn
    X = data.T
    
    # Fit k-means
    kmeans = MiniBatchKMeans(
        n_clusters=n_samples,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=42
    )
    kmeans.fit(X)
    
    # Find nearest point to each centroid
    selected_indices = []
    for centroid in kmeans.cluster_centers_:
        distances = np.linalg.norm(X - centroid, axis=1)
        nearest_idx = np.argmin(distances)
        selected_indices.append(nearest_idx)
    
    return np.array(selected_indices)

import sys
from planning.utils.csdecomp_path import CSDECOMP_PATH
sys.path.append(f'{CSDECOMP_PATH}/bazel-bin/csdecomp/src/pybind/pycsdecomp')
import pycsdecomp as csd

PETE_ASSETS = os.path.abspath(os.path.dirname(__file__)+"/../pete_assets/")
root = os.path.abspath(os.path.dirname(__file__)+'/../')

directives_file = PETE_ASSETS+'/assets/directives/iiwa7_on_table.yaml'
meshcat = StartMeshcat()
builder = RobotDiagramBuilder()
plant = builder.plant()
scene_graph = builder.scene_graph()
parser = builder.parser()

parser.package_map().Add("adaptive_decomp", PETE_ASSETS+"/assets")
parser.package_map().Add("iiwa_description", PETE_ASSETS+"/assets/iiwa")
parser.package_map().Add("wsg_description", PETE_ASSETS+"/assets/wsg_description")
parser.package_map().Add("tri_finray_gripper", PETE_ASSETS+"/assets/tri_finray_gripper")

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
diagram.ForcedPublish(diagram_context)
robot_model_instances = [plant.GetModelInstanceByName(m) for m in ['iiwa7', 
                                                                   'wsg', 
                                                                   'finray']]
meshcat.SetProperty('/drake/proximity', "visible", True)
# Calculate the 8 corners of the workspace
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

plot_points(meshcat, 
            ws_corners_online_voxels, 
            'drm/online_voxel_range_markers', 
            0.03,
            Rgba(1,0,1,0.9))


parser.package_map().Add("adaptive_decomp", PETE_ASSETS+"/assets")
parser.package_map().Add("iiwa_description", PETE_ASSETS+"/assets/iiwa")
parser.package_map().Add("wsg_description", PETE_ASSETS+"/assets/wsg_description")
parser.package_map().Add("tri_finray_gripper", PETE_ASSETS+"/assets/tri_finray_gripper")

csd_parser = csd.URDFParser()
csd_parser.register_package("adaptive_decomp", PETE_ASSETS+"/assets")
csd_parser.register_package("iiwa_description", PETE_ASSETS+"/assets/iiwa")
csd_parser.register_package("wsg_description", PETE_ASSETS+"/assets/wsg_description")
csd_parser.register_package("tri_finray_gripper", PETE_ASSETS+"/assets/tri_finray_gripper")
csd_parser.parse_directives(PETE_ASSETS+"/assets/directives/iiwa7_on_table_with_extra_cameras.yaml")
csd_plant = csd_parser.build_plant()
csd_mplant = csd_plant.getMinimalPlant()
csd_domain = csd.HPolyhedron()
csd_domain.MakeBox(csd_plant.getPositionLowerLimits(), 
                   csd_plant.getPositionUpperLimits())

#build the roadmap
np.random.seed(1337)
rm_opts = csd.RoadmapOptions()
rm_opts.robot_map_size_x = 1.5
rm_opts.robot_map_size_y = 2.25
rm_opts.robot_map_size_z = 1.
rm_opts.map_center = np.array([0.4, 0, 0.5])
rm_opts.nodes_processed_before_debug_statement = 500
rm_opts.max_configuration_distance_between_nodes = max_configuration_distance
rm_opts.max_task_space_distance_between_nodes = max_task_space_distance
rm_opts.offline_voxel_resolution = offline_voxel_resolution
rm_opts.edge_step_size = edge_step_size
rm_builder = csd.RoadmapBuilder(csd_plant, "iiwa7::iiwa_link_ee", rm_opts)
cfree_samps = []
num_samps = 0

while True:
    samples = csd.UniformSampleInHPolyhedronCuda([csd_domain], 
                                                csd_domain.ChebyshevCenter(), 
                                                batchsize, 
                                                300,
                                                np.random.randint(0, 1000))[0]

    results = csd.CheckCollisionFreeCuda(samples, csd_mplant)        
    cfree_samps.append(samples[:, np.where(results)[0]])
    num_samps+= cfree_samps[-1].shape[1]
    print(f'num samples sofar {num_samps} delta {cfree_samps[-1].shape[1]}')
    if num_samps>=num_samples_to_collect:
        break
cfree_samps = np.concatenate(tuple(cfree_samps),axis = 1)
if use_kmeans:
    t1 = time.time()
    samples_for_drm_idx = select_subset_kmeans(cfree_samps, roadmap_size)
    drm_samples = cfree_samps[:, samples_for_drm_idx]
    t2 = time.time()
    print(f"time kmeans selection {t2-t1:.2f} num_input_points {num_samps}")
else:
    print("not using kmeans")
    drm_samples = cfree_samps[:, 0:roadmap_size]
print("adding nodes manually")
rm_builder.add_nodes_manual(drm_samples)
print(f"building roadmap with {roadmap_size} nodes")
rm_builder.build_roadmap(max_neighbors=max_neighbors)
sys.stdout.flush()
rm_builder.build_pose_map()
sys.stdout.flush()
rm_builder.build_collision_map()
sys.stdout.flush()
if store_edges:
    rm_builder.build_edge_collision_map(edge_storage_step_size)
    sys.stdout.flush()
drm = rm_builder.get_drm()
rm_builder.write(root+'/'+rm_name)
print('graph stored')
# G = nx.Graph()

# #Add edges to the graph
# edges_added = []
# for node, neighbors in drm.node_adjacency_map.items():
#     for neighbor in neighbors:
#         e_id = f"{np.min([node,neighbor])},{np.max([node,neighbor])}"
#         if e_id not in edges_added:
#             edges_added.append(e_id)
#             G.add_edge(node, neighbor)
# write_graph_summary(G)
print('done')