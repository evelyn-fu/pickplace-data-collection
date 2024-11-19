import multiprocessing as mp
import os.path
import time
from collections import OrderedDict
from typing import Dict
import logging

import numpy as np
import pydot
from IPython.display import SVG, display
from pydrake.common.value import AbstractValue
from pydrake.geometry import (
    Meshcat,
    MeshcatVisualizer,
    QueryObject,
    Rgba,
    Role,
    SceneGraph,
    Sphere,
    StartMeshcat,
)
from pydrake.geometry.optimization import (
    HPolyhedron,
    IrisInConfigurationSpace,
    IrisOptions,
    LoadIrisRegionsYamlFile,
    SaveIrisRegionsYamlFile,
)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.meshcat import JointSliders
from pydrake.multibody.parsing import PackageMap, Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.tree import Body
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.visualization import AddDefaultVisualization, ModelVisualizer
from pydrake.solvers import MosekSolver, GurobiSolver
from pydrake.common import use_native_cpp_logging

from manipulation import running_as_notebook
from manipulation.utils import FindDataResource

# logging.getLogger('drake').setLevel(logging.DEBUG)
models_path = "scenario_datas/scenario_data_no_object.dmd.yaml"
iris_options = IrisOptions()
iris_options.iteration_limit = 10
# increase num_collision_infeasible_samples to improve the (probabilistic)
# certificate of having no collisions.
iris_options.num_collision_infeasible_samples = 3
iris_options.require_sample_point_is_contained = True
iris_options.relative_termination_threshold = 0.01
iris_options.termination_threshold = -1

# If use_existing_regions_as_obstacles is True, then iris_regions will be
# shrunk by regions_as_obstacles_margin, and then passed to
# iris_options.configuration_obstacles.
use_existing_regions_as_obstacles = True
regions_as_obstacles_scale_factor = 0.95

# We can compute some regions in parallel.
num_parallel = mp.cpu_count()

def ScaleHPolyhedron(hpoly, scale_factor):
    # Shift to the center.
    xc = hpoly.ChebyshevCenter()
    A = hpoly.A()
    b = hpoly.b() - A @ xc
    # Scale
    b = scale_factor * b
    # Shift back
    b = b + A @ xc
    return HPolyhedron(A, b)


def _CheckNonEmpty(region):
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(region.ambient_dimension())
    region.AddPointInSetConstraints(prog, x)
    result = Solve(prog)
    assert result.is_success()


def _CalcRegion(name, seed):
    use_native_cpp_logging()
    builder = DiagramBuilder()
    plant = AddMultibodyPlantSceneGraph(builder, 0.0)[0]
    LoadRobot(plant, models_path)
    plant.Finalize()
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)
    plant.SetPositions(plant_context, seed)

    # Gaze constraint:
    ik = InverseKinematics(plant, plant_context)
    corners = [
        [0.3, 0.2, 0.07],
        [0.3, -0.2, 0.07],
        [0.5, 0.2, 0.07],
        [0.5, -0.2, 0.07],
        [0.3, 0.2, 0.23],
        [0.3, -0.2, 0.23],
        [0.5, 0.2, 0.23],
        [0.5, -0.2, 0.23]
    ]
    for c in corners:
        ik.AddGazeTargetConstraint(
            frameA=plant.GetFrameByName("iiwa_link_7"), 
            p_AS=np.array([0., 0., 0.09]),
            n_A=np.array([0., 0., 1.]),
            frameB=plant.GetFrameByName("world"),
            p_BT=np.array(c),
            cone_half_angle=np.pi*40/180)
    iris_options.prog_with_additional_constraints = ik.prog()

    if use_existing_regions_as_obstacles:
        configuration_obstacles = [
            ScaleHPolyhedron(r, regions_as_obstacles_scale_factor)
            for k, r in iris_regions.items()
            if k != name
        ]
        iris_options.configuration_obstacles = []
        for h in configuration_obstacles:
            _CheckNonEmpty(h)
            if not h.PointInSet(seed):
                iris_options.configuration_obstacles.append(h)
            else:
                display(f"Seed already in computed region: {name}")
    else:
        iris_options.configuration_obstacles = None
    display(f"Computing region for seed: {name}")
    start_time = time.time()
    print(plant.GetPositionLowerLimits())
    try:
        hpoly = IrisInConfigurationSpace(plant, plant_context, iris_options)
    except:
        print(f"full corners gaze constraint failed at seed {name}")
        ik_backup = InverseKinematics(plant, plant_context)
        ik_backup.AddGazeTargetConstraint(
            frameA=plant.GetFrameByName("iiwa_link_7"), 
            p_AS=np.array([0., 0., 0.09]),
            n_A=np.array([0., 0., 1.]),
            frameB=plant.GetFrameByName("world"),
            p_BT=np.array([0.4, 0, 0.1]),
            cone_half_angle=np.pi*20/180)
        iris_options.prog_with_additional_constraints = ik_backup.prog()
        hpoly = IrisInConfigurationSpace(plant, plant_context, iris_options)

    display(
        f"Finished seed {name}; Computation time: {(time.time() - start_time):.2f} seconds"
    )

    _CheckNonEmpty(hpoly)
    reduced = hpoly.ReduceInequalities()
    _CheckNonEmpty(reduced)

    return reduced


def GenerateRegion(name, seed):
    global iris_regions
    if name in iris_regions:
        display(f"Region already computed: {name}")
        return
    iris_regions[name] = _CalcRegion(name, seed)
    SaveIrisRegionsYamlFile("../regions/gaze_constrained_scanning_regions_2.yaml.autosave", iris_regions)


def GenerateRegions(seed_dict, verbose=True):
    if use_existing_regions_as_obstacles:
        # Then run serially
        for k, v in seed_dict.items():
            GenerateRegion(k, v)
        return

    loop_time = time.time()
    with mp.Pool(processes=num_parallel) as pool:
        new_regions = pool.starmap(_CalcRegion, [[k, v] for k, v in seed_dict.items()])

    if verbose:
        print("Loop time:", time.time() - loop_time)

    global iris_regions
    iris_regions.update(dict(list(zip(seed_dict.keys(), new_regions))))


def DrawRobot(query_object: QueryObject, meshcat_prefix: str, draw_world: bool = True):
    rgba = Rgba(0.7, 0.7, 0.7, 0.3)
    role = Role.kProximity
    # This is a minimal replication of the work done in MeshcatVisualizer.
    inspector = query_object.inspector()
    for frame_id in inspector.GetAllFrameIds():
        if frame_id == inspector.world_frame_id():
            if not draw_world:
                continue
            frame_path = meshcat_prefix
        else:
            frame_path = f"{meshcat_prefix}/{inspector.GetName(frame_id)}"
        frame_path.replace("::", "/")
        frame_has_any_geometry = False
        for geom_id in inspector.GetGeometries(frame_id, role):
            path = f"{frame_path}/{geom_id.get_value()}"
            path.replace("::", "/")
            meshcat.SetObject(path, inspector.GetShape(geom_id), rgba)
            meshcat.SetTransform(path, inspector.GetPoseInFrame(geom_id))
            frame_has_any_geometry = True

        if frame_has_any_geometry:
            X_WF = query_object.GetPoseInWorld(frame_id)
            meshcat.SetTransform(frame_path, X_WF)


def VisualizeRegion(region_name, num_to_draw=30, draw_illustration_role_once=True):
    """
    A simple hit-and-run-style idea for visualizing the IRIS regions:
    1. Start at the center. Pick a random direction and run to the boundary.
    2. Pick a new random direction; project it onto the current boundary, and run along it. Repeat
    """

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    LoadRobot(plant, models_path)
    plant.Finalize()
    if draw_illustration_role_once:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    scene_graph_context = scene_graph.GetMyContextFromRoot(context)

    global iris_regions
    region = iris_regions[region_name]

    q = region.ChebyshevCenter()
    plant.SetPositions(plant_context, q)
    diagram.ForcedPublish(context)

    query = scene_graph.get_query_output_port().Eval(scene_graph_context)
    DrawRobot(query, f"{region_name}/0", True)

    rng = np.random.default_rng()
    nq = plant.num_positions()
    prog = MathematicalProgram()
    qvar = prog.NewContinuousVariables(nq, "q")
    prog.AddLinearConstraint(region.A(), 0 * region.b() - np.inf, region.b(), qvar)
    cost = prog.AddLinearCost(np.ones((nq, 1)), qvar)

    for i in range(1, num_to_draw):
        direction = rng.standard_normal(nq)
        cost.evaluator().UpdateCoefficients(direction)

        result = Solve(prog)
        assert result.is_success()

        q = result.GetSolution(qvar)
        plant.SetPositions(plant_context, q)
        query = scene_graph.get_query_output_port().Eval(scene_graph_context)
        DrawRobot(query, f"{region_name}/{i}", False)


def VisualizeRegions():
    for k in iris_regions.keys():
        meshcat.Delete()
        VisualizeRegion(k)
        button_name = f"Visualizing {k}; Press for next region"
        meshcat.AddButton(button_name, "Enter")
        print("Press Enter to visualize the next region")
        while meshcat.GetButtonClicks(button_name) < 1:
            time.sleep(1.0)
        meshcat.DeleteButton(button_name)

def MyInverseKinematics(X_WE, plant=None, context=None, initial_guess=None):
    if not plant:
        plant = MultibodyPlant(0.0)
        LoadRobot(plant, models_path)
        plant.Finalize()
    if not context:
        context = plant.CreateDefaultContext()
    # E = ee_body.body_frame()
    E = plant.GetBodyByName("body").body_frame()

    ik = InverseKinematics(plant, context)

    ik.AddPositionConstraint(
        E, [0, 0, 0], plant.world_frame(), X_WE.translation() - 0.01, X_WE.translation() + 0.01
    )

    ik.AddOrientationConstraint(
        E, RotationMatrix(), plant.world_frame(), X_WE.rotation(), 0.01
    )

    prog = ik.get_mutable_prog()
    q = ik.q()

    q0 = plant.GetPositions(context)
    if initial_guess is None:
        initial_guess = q0
    prog.AddQuadraticErrorCost(np.identity(len(q)), initial_guess, q)
    prog.SetInitialGuess(q, initial_guess)
    result = Solve(ik.prog())
    if not result.is_success():
        print("IK failed")
        return None
    plant.SetPositions(context, result.GetSolution(q))
    return result.GetSolution(q)

def MyForwardKinematics(q, plant=None):
    if not plant:
        plant = MultibodyPlant(0.0)
        LoadRobot(plant, models_path)
        plant.Finalize()

    temp_context = plant.CreateDefaultContext()
    plant.SetPositions(temp_context, q)

    return plant.EvalBodyPoseInWorld(temp_context, plant.GetBodyByName("body"))

def get_seeded_region(models_path, com, rot, dims, q_nominal):
    object_area_urdf = """<?xml version="1.0"?>
    <robot name="object_area">
    <link name="object_area">
    <collision name="object_area">
        <origin rpy="0 0 0" xyz="0.4 0.0 0.15"/>
        <geometry>
        <box size="0.4 0.4 0.16"/>
        </geometry>
    </collision>
    </link>
    <joint name="fixed_link_weld" type="fixed">
    <parent link="world"/>
    <child link="object_area"/>
    </joint>
    </robot>
    """

    print("getting seeded region around", q_nominal)
    builder = RobotDiagramBuilder()
    plant = builder.plant()
    builder.parser().AddModels(models_path)
    builder.parser().AddModelsFromString(object_area_urdf, "urdf")
    diagram = builder.Build()

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, q_nominal)

    options = IrisOptions()
    # You'll see a few glancing collisions in the resulting region; increase this number
    # to reduce them (at the cost of IRIS running for longer)
    options.num_collision_infeasible_samples = 3
    options.random_seed = 1235
    options.require_sample_point_is_contained = True
    ik = InverseKinematics(plant, plant_context)
    ik.AddGazeTargetConstraint(
        frameA=plant.GetFrameByName("iiwa_link_7"), 
        p_AS=[0,0,0],
        n_A=[0,0,1],
        frameB=plant.GetFrameByName("world"),
        p_PT=[0.4, 0, 0.1],
        cone_half_angle=np.pi*40/180)
    options.prog_with_additional_constraints = ik.prog()
    region = IrisInConfigurationSpace(plant, plant_context, options)

    return region

def LoadRobot(plant: MultibodyPlant, models_path: str) -> Body:
    """Setup your plant, and return the body corresponding to your
    end-effector."""

    parser = Parser(plant)

    object_area_urdf = """<?xml version="1.0"?>
    <robot name="object_area">
    <link name="object_area">
    <collision name="object_area">
        <origin rpy="0 0 0" xyz="0.4 0.0 0.15"/>
        <geometry>
        <box size="0.4 0.4 0.16"/>
        </geometry>
    </collision>
    </link>
    <joint name="fixed_link_weld" type="fixed">
    <parent link="world"/>
    <child link="object_area"/>
    </joint>
    </robot>
    """

    parser.AddModels(models_path)
    parser.AddModelsFromString(object_area_urdf, "urdf")
    gripper = plant.GetModelInstanceByName("wsg")
    end_effector_body = plant.GetBodyByName("body", gripper)
    return end_effector_body

def get_default_position():
    plant = MultibodyPlant(0.0)
    LoadRobot(plant)
    plant.Finalize()
    context = plant.CreateDefaultContext()
    return plant.GetPositions(context)


# Note: The order of the seeds matters when we are using existing regions as
# configuration_obstacles.
seeds = OrderedDict()
print("Top")
seeds["Top"] = MyInverseKinematics(
    RigidTransform(RollPitchYaw(-np.pi/2, 0, 0), [0.4, 0, 0.7])
)
print(list(seeds["Top"]))

print("Back Left")
seeds["Back Left"] = MyInverseKinematics(
    RigidTransform(RollPitchYaw(-5*np.pi/6, 0, np.pi/4), [0.1, 0.3, 0.4]),
    initial_guess = seeds["Top"]
)
print(list(seeds["Back Left"]))
print("Back Right")
seeds["Back Right"] = MyInverseKinematics(
    RigidTransform(RollPitchYaw(np.pi/6, np.pi, -np.pi/4), [0.1, -0.3, 0.4]),
    initial_guess = seeds["Top"]
)
print(list(seeds["Back Right"]))

back_back_left_pose = MyForwardKinematics(np.array([2.5, 1.3, 1.8, 1.8, 0.2, -1.5, -1.4]))
back_back_right_pose = MyForwardKinematics(np.array([0.64, -1.3, 1.34, 1.8, -0.2, -1.5, 1.3]))

print("Back Back Left")
seeds["Back Back Left"] = MyInverseKinematics(
    back_back_left_pose,
    initial_guess = seeds["Back Left"]
)
print(list(seeds["Back Back Left"]))
print("Back Back Right")
seeds["Back Back Right"] = MyInverseKinematics(
    back_back_right_pose,
    initial_guess = seeds["Back Right"]
)
print(list(seeds["Back Back Right"]))

print("Left")
seeds["Left"] = MyInverseKinematics(
    RigidTransform(RollPitchYaw(-5*np.pi/6, 0, 0), [0.4, 0.4, 0.4]),
    initial_guess = seeds["Back Left"]
)
print(list(seeds["Left"]))
print("Right")
seeds["Right"] = MyInverseKinematics(
    RigidTransform(RollPitchYaw(np.pi/6, np.pi, 0), [0.4, -0.4, 0.4]),
    initial_guess = seeds["Back Right"]
)
print(list(seeds["Right"]))

print("Front Left")
seeds["Front Left"] = MyInverseKinematics(
    RigidTransform(RollPitchYaw(-5*np.pi/6, 0, -np.pi/10), [0.5, 0.35, 0.4]),
    initial_guess = seeds["Left"]
)
print(list(seeds["Front Left"]))
print("Front Right")
seeds["Front Right"] = MyInverseKinematics(
    RigidTransform(RollPitchYaw(np.pi/6, np.pi, np.pi/10), [0.5, -0.35, 0.4]),
    initial_guess = seeds["Right"]
)
print(list(seeds["Front Right"]))

print("Top Left")
seeds["Top Left"] = MyInverseKinematics(
    RigidTransform(RollPitchYaw(-2*np.pi/3, 0, 0), [0.4, 0.25, 0.55]),
    initial_guess = seeds["Top"]
)
print(list(seeds["Top Left"]))
print("Top Right")
seeds["Top Right"] = MyInverseKinematics(
    RigidTransform(RollPitchYaw(-1*np.pi/3, 0, 0), [0.4, -0.25, 0.55]),
    initial_guess = seeds["Top"]
)
print(list(seeds["Top Right"]))

# if MosekSolver().available() and MosekSolver().enabled():
#     print("mosek enabled")
#     iris_regions = dict()  # reset the iris regions
#     try:
#         # old_regions = LoadIrisRegionsYamlFile("../regions/gaze_constrained_scanning_regions_2.yaml.autosave")
#         # iris_regions["Top"] = old_regions["Top"]
#         # iris_regions["Front"] = old_regions["Front"]
#         # iris_regions["Top Left"] = old_regions["Top Left"]
#         # iris_regions["Top Right"] = old_regions["Top Right"]
#         # iris_regions["Back Left"] = old_regions["Back Left Top Seeded"]
#         # iris_regions["Back Right"] = old_regions["Back Right Top Seeded"]
#         iris_regions.update(LoadIrisRegionsYamlFile("../regions/gaze_constrained_scanning_regions_3.yaml"))
#     except:
#         pass
#     print(iris_regions)
#     GenerateRegions(seeds)

#     SaveIrisRegionsYamlFile("../regions/gaze_constrained_scanning_regions_3.yaml", iris_regions)

#     VisualizeRegions()
# elif GurobiSolver().available() and GurobiSolver().enabled():
#     print("gurobi enabled")
#     iris_regions = dict()  # reset the iris regions
#     GenerateRegions(seeds)

#     SaveIrisRegionsYamlFile("../regions/gaze_constrained_scanning_regions_3.yaml", iris_regions)

#     VisualizeRegions()
# else:
#     print("No solvers available")
