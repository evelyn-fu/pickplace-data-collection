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

# Each motion given by axis of motion (X, Y, Z, y, p, r), direction (1, -1), amount (radians or m), and time it should take (s)
default_motion_primitives = [
    ("y", 1, np.pi, 5.0), 
    # ("p", -1, np.pi * 110.0 / 180.0, 3.0),
    # ("p", 1, np.pi * 130.0 / 180.0, 4.0),
    # ("p", -1, np.pi * 20.0 / 180.0, 1.0),
    ("y", -1, np.pi * 1.5, 10.0),
    ("y", 1, np.pi * 0.5, 5.0),
    # ("p", 1, np.pi * 110.0 / 180.0, 3.0),
    # ("p", -1, np.pi * 130.0 / 180.0, 4.0),
    # ("p", 1, np.pi * 20.0 / 180.0, 1.0)
]

# [x, y, z, y, p, r, initialization time]
default_start_pose = [0.0, -0.5, 0.35, np.pi/2, 0.0, np.pi, 3.0]

axis_index = {"X": 0, "Y": 1, "Z": 2, "y": 3, "p": 4, "r": 5}

class MotionPrimitives(LeafSystem):
    def __init__(self, motion_map=default_motion_primitives, start_pose=default_start_pose):
        super().__init__()

        self.times = [0.0, start_pose[6]]
        self.X = [start_pose[0], start_pose[0]]
        self.Y = [start_pose[1], start_pose[1]]
        self.Z = [start_pose[2], start_pose[2]]
        self.y = [start_pose[3], start_pose[3]]
        self.p = [start_pose[4], start_pose[4]]
        self.r = [start_pose[5], start_pose[5]]

        all_checkpoints = [self.X, self.Y, self.Z, self.y, self.p, self.r]

        cur_pose = copy.deepcopy(start_pose)
        cur_time = start_pose[6]
        for motion in motion_map:
            cur_time += motion[3]
            self.times.append(cur_time)

            cur_pose[axis_index[motion[0]]] += motion[1] * motion[2]
            for i in range(len(all_checkpoints)):
                all_checkpoints[i].append(cur_pose[i])

        self.DeclareAbstractOutputPort(name="pose_out",
                                        alloc=lambda: Value(RigidTransform()),
                                        calc=self.PoseOut)
        
        self.DeclareVectorOutputPort(
            "wsg_position", 1, self.GripperOut
        )
    
    def PoseOut(self, context, output):
        t = context.get_time()

        pose_out = RigidTransform()
        if (t > self.times[-1]):
            pose_out = RigidTransform(RotationMatrix(RollPitchYaw(self.r[-1], self.p[-1], self.y[-1])), [self.X[-1], self.Y[-1], self.Z[-1]])
        else:
            X = np.interp(t, self.times, self.X)
            Y = np.interp(t, self.times, self.Y)
            Z = np.interp(t, self.times, self.Z)
            y = np.interp(t, self.times, self.y)
            p = np.interp(t, self.times, self.p)
            r = np.interp(t, self.times, self.r)

            pose_out = RigidTransform(RotationMatrix(RollPitchYaw(r, p, y)), [X, Y, Z])

        output.set_value(pose_out)

    def GripperOut(self, context, output):
        t = context.get_time()

        position = 0.107  # open
        if (t >= self.times[1]):
            position = 0.002  # close

        output.SetAtIndex(0, position)


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

        # Publish at 33 fps
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

        # remove alpha
        color = color[:, :, :3]
        color_pil = Image.fromarray(color)
        color_pil.save(self.dirstr+"/rgb/"+timestr+".png")
        
        # get mask for bottle
        object_labels = np.unique(label_image)
        masks = [
            np.uint8(np.where(label_image == label, 255, 0)) for label in object_labels
        ]
        mask_pil = Image.fromarray(masks[0])
        mask_pil.save(self.dirstr+"/masks/"+timestr+".png")

        # cap depth at 3000mm
        depth[depth > 3000] = 3000
        depth_pil = Image.fromarray(depth)
        depth_pil.save(self.dirstr+"/depth/"+timestr+".png")


def motion_primitives_with_camera(dirstr = "test3", scenario_data_filename="scenario_data_welded.yml"):
    meshcat.ResetRenderMode()

    builder = DiagramBuilder()

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    full_file_path = os.path.join(dir_path, os.path.join("scenario_datas", scenario_data_filename))
    scenario = load_scenario(filename=full_file_path)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))

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

    # Set up motion primitives
    motion_primitives = builder.AddSystem(MotionPrimitives())
    
    builder.Connect(
        motion_primitives.GetOutputPort("pose_out"), differential_ik.GetInputPort("X_WE_desired")
    )
    builder.Connect(
        motion_primitives.GetOutputPort("wsg_position"), station.GetInputPort("wsg.position")
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
    plant = station.GetSubsystemByName("plant")
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
    motion_primitives_with_camera(dirstr="tests/test_welded_short_grip", scenario_data_filename="scenario_data_welded_short_axis_grip.yml")