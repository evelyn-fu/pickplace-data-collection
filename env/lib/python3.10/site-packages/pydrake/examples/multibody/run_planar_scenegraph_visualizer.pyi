from pydrake.common import temp_directory as temp_directory
from pydrake.examples import ManipulationStation as ManipulationStation
from pydrake.multibody.parsing import Parser as Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph as AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator as Simulator
from pydrake.systems.framework import DiagramBuilder as DiagramBuilder
from pydrake.systems.planar_scenegraph_visualizer import ConnectPlanarSceneGraphVisualizer as ConnectPlanarSceneGraphVisualizer

def run_pendulum_example(args) -> None: ...
def run_manipulation_example(args) -> None: ...
def main() -> None: ...
