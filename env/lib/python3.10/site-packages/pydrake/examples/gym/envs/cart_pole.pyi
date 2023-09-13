from _typeshed import Incomplete
from pydrake.common import FindResourceOrThrow as FindResourceOrThrow
from pydrake.common.cpp_param import List as List
from pydrake.common.value import Value as Value
from pydrake.examples.gym.named_view_helpers import MakeNamedViewActuation as MakeNamedViewActuation, MakeNamedViewPositions as MakeNamedViewPositions, MakeNamedViewState as MakeNamedViewState
from pydrake.geometry import ClippingRange as ClippingRange, ColorRenderCamera as ColorRenderCamera, DepthRange as DepthRange, DepthRenderCamera as DepthRenderCamera, MakeRenderEngineVtk as MakeRenderEngineVtk, MeshcatVisualizer as MeshcatVisualizer, RenderCameraCore as RenderCameraCore, RenderEngineVtkParams as RenderEngineVtkParams
from pydrake.gym import DrakeGymEnv as DrakeGymEnv
from pydrake.math import RigidTransform as RigidTransform, RollPitchYaw as RollPitchYaw
from pydrake.multibody.math import SpatialForce as SpatialForce
from pydrake.multibody.parsing import Parser as Parser
from pydrake.multibody.plant import AddMultibodyPlant as AddMultibodyPlant, ExternallyAppliedSpatialForce_ as ExternallyAppliedSpatialForce_, MultibodyPlant as MultibodyPlant, MultibodyPlantConfig as MultibodyPlantConfig
from pydrake.systems.analysis import Simulator as Simulator
from pydrake.systems.drawing import plot_graphviz as plot_graphviz, plot_system_graphviz as plot_system_graphviz
from pydrake.systems.framework import DiagramBuilder as DiagramBuilder, EventStatus as EventStatus, LeafSystem as LeafSystem, PublishEvent as PublishEvent
from pydrake.systems.primitives import ConstantVectorSource as ConstantVectorSource, Multiplexer as Multiplexer, PassThrough as PassThrough
from pydrake.systems.sensors import CameraInfo as CameraInfo, RgbdSensor as RgbdSensor

sim_time_step: float
gym_time_step: float
controller_time_step: float
gym_time_limit: int
drake_contact_models: Incomplete
contact_model: Incomplete
drake_contact_solvers: Incomplete
contact_solver: Incomplete

def AddAgent(plant): ...
def make_sim(meshcat: Incomplete | None = ..., time_limit: int = ..., debug: bool = ..., obs_noise: bool = ..., monitoring_camera: bool = ..., add_disturbances: bool = ...): ...
def reset_handler(simulator, diagram_context, seed) -> None: ...
def DrakeCartPoleEnv(meshcat: Incomplete | None = ..., time_limit=..., debug: bool = ..., obs_noise: bool = ..., monitoring_camera: bool = ..., add_disturbances: bool = ...): ...
