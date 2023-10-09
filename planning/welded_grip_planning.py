import numpy as np
import os
import copy
from PIL import Image
import matplotlib.pyplot as plt
from pydrake.geometry import (
    StartMeshcat,
    RenderLabel,
    Role,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.visualization import (
    MeshcatPoseSliders,
)
from pydrake.systems.sensors import (
    ImageRgba8U,
    ImageDepth16U,
    ImageLabel16I,
)
from pydrake.math import (
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
)
from pydrake.common.value import Value

from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK, ExtractBodyPose
from manipulation.station import MakeHardwareStation, load_scenario

class ImageSaver(LeafSystem):
    def __init__(self, dirstr = "test3"):
        super().__init__()

        self.dirstr = dirstr
        self.DeclareAbstractInputPort(name="rgb_in",
                                      model_value=Value(ImageRgba8U()))
        self.DeclareAbstractInputPort(name="depth_in",
                                      model_value=Value(ImageDepth16U()))
        self.DeclareAbstractInputPort(name="label_in",
                                      model_value=Value(ImageLabel16I()))

        # Calling `ForcePublish()` will trigger the callback.
        self.DeclareForcedPublishEvent(self.Publish)

        # Publish once every second.
        self.DeclarePeriodicPublishEvent(period_sec=0.03,
                                         offset_sec=0,
                                         publish=self.Publish)
        
    def Publish(self, context):
        time_ms = int(context.get_time() * 1000)
        timestr = f"{time_ms:06d}"
        
        color = self.GetInputPort("rgb_in").Eval(context).data

        depth = copy.deepcopy(
            self.GetInputPort("depth_in").Eval(context).data.squeeze()
        )

        label_image = copy.deepcopy(
            self.GetInputPort("label_in").Eval(context).data.squeeze()
        )

        color = color[:, :, :3]
        color_pil = Image.fromarray(color)
        color_pil.save(self.dirstr+"/rgb/"+timestr+".png")
        
        object_labels = np.unique(label_image)
        masks = [
            np.uint8(np.where(label_image == label, 255, 0)) for label in object_labels
        ]

        mask_pil = Image.fromarray(masks[0])
        mask_pil.save(self.dirstr+"/masks/"+timestr+".png")

        depth[depth > 3000] = 3000
        depth_pil = Image.fromarray(depth)
        depth_pil.save(self.dirstr+"/depth/"+timestr+".png")


def teleop_with_camera(dirstr = "test3"):
    meshcat.ResetRenderMode()

    builder = DiagramBuilder()

    cwd = os.path.abspath(os.path.dirname(__file__))
    full_file_path = os.path.join(cwd, "scenario_data_welded.yml")
    scenario = load_scenario(filename=full_file_path)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))

    # TODO(russt): Replace with station.AddDiffIk(...)
    controller_plant = station.GetSubsystemByName(
        "iiwa.controller"
    ).get_multibody_plant_for_control()
    # Set up differential inverse kinematics.
    differential_ik = AddIiwaDifferentialIK(
        builder,
        controller_plant,
        frame=controller_plant.GetFrameByName("iiwa_link_7"),
    )
    builder.Connect(
        differential_ik.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.state_estimated"),
        differential_ik.GetInputPort("robot_state"),
    )

    # Set up teleop widgets.
    meshcat.DeleteAddedControls()
    default_pose = RigidTransform(RotationMatrix(RollPitchYaw(3.14, 0.24, 1.57)), [0, -0.44, 0.3])
    teleop = builder.AddSystem(
        MeshcatPoseSliders(
            meshcat,
            lower_limit=[0, -0.5, -np.pi, -0.6, -0.8, 0.0],
            upper_limit=[2 * np.pi, np.pi, np.pi, 0.8, 0.3, 1.1],
            initial_pose=default_pose
        )
    )
    # teleop.SetPose(RigidTransform(RotationMatrix(RollPitchYaw()), [0, 0, 0.3]))
    builder.Connect(
        teleop.get_output_port(), differential_ik.GetInputPort("X_WE_desired")
    )
    # Note: This is using "Cheat Ports". For it to work on hardware, we would
    # need to construct the initial pose from the HardwareStation outputs.
    plant = station.GetSubsystemByName("plant")
    ee_pose = builder.AddSystem(
        ExtractBodyPose(
            station.GetOutputPort("body_poses"),
            plant.GetBodyByName("iiwa_link_7").index(),
        )
    )
    builder.Connect(
        station.GetOutputPort("body_poses"), ee_pose.get_input_port()
    )
    builder.Connect(ee_pose.get_output_port(), teleop.get_input_port())
    wsg_teleop = builder.AddSystem(WsgButton(meshcat))
    builder.Connect(
        wsg_teleop.get_output_port(0), station.GetInputPort("wsg.position")
    )

    # initialize image writer and save directories
    sensor = station.GetSubsystemByName("rgbd_sensor_camera0")
    K = sensor.color_camera_info().intrinsic_matrix()
    if not os.path.exists(dirstr):
        os.makedirs(dirstr)
    if not os.path.exists(dirstr+"/rgb/"):
        os.makedirs(dirstr+"/rgb/")
    if not os.path.exists(dirstr+"/depth/"):
        os.makedirs(dirstr+"/depth/")
    if not os.path.exists(dirstr+"/masks/"):
        os.makedirs(dirstr+"/masks/")
    np.savetxt(dirstr+"/cam_K.txt", K)

    
    img_saver = builder.AddSystem(ImageSaver(dirstr))
    builder.Connect(station.GetOutputPort("camera0.rgb_image"), img_saver.GetInputPort("rgb_in"))
    builder.Connect(station.GetOutputPort("camera0.depth_image_16u"), img_saver.GetInputPort("depth_in"))
    builder.Connect(station.GetOutputPort("camera0.label_image"), img_saver.GetInputPort("label_in"))

    # Build diagram
    diagram = builder.Build()

    # Simulate
    simulator = Simulator(diagram)
    simulator_context = simulator.get_mutable_context()

    # Remove labels of anything but mustard
    scene_graph = station.GetSubsystemByName("scene_graph")
    source_id = plant.get_source_id()
    scene_graph_context = scene_graph.GetMyMutableContextFromRoot(simulator_context)
    query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
    inspector = query_object.inspector()
    for geometry_id in inspector.GetAllGeometryIds():
        properties = copy.deepcopy(inspector.GetPerceptionProperties(geometry_id))
        if properties is None:
            continue
        frame_id = inspector.GetFrameId(geometry_id)
        body = plant.GetBodyFromFrameId(frame_id)
        if body.model_instance() == plant.GetModelInstanceByName("mustard_bottle"):
            properties.UpdateProperty("label", "id", RenderLabel(0)) # Make mustard label 0
        else:
            properties.UpdateProperty("label", "id", RenderLabel.kDontCare)
        scene_graph.RemoveRole(scene_graph_context, source_id, geometry_id, Role.kPerception)
        scene_graph.AssignRole(scene_graph_context, source_id, geometry_id, properties)


    simulator.set_target_realtime_rate(1.0)

    meshcat.AddButton("Stop Simulation", "Escape")
    print("Press Escape to stop the simulation")
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.03)
    meshcat.DeleteButton("Stop Simulation")


if __name__ == "__main__":
    # Start the visualizer.
    meshcat = StartMeshcat()
    teleop_with_camera()