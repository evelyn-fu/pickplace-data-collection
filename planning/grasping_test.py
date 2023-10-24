import numpy as np
import os
import copy
from PIL import Image
from grasp import GraspListener
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
from pydrake.common.value import Value
from pydrake.perception import (
    Concatenate,
    DepthImageToPointCloud
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

default_start_pose = [0.0, -0.5, 0.5, np.pi/2, 0.0, np.pi, 3.0]

class GraspSelector(LeafSystem):
    def __init__(self, start_pose=default_start_pose):
        super().__init__()

        X = start_pose[0]
        Y = start_pose[1]
        Z = start_pose[2]
        y = start_pose[3]
        p = start_pose[4]
        r = start_pose[5]
        self.pose_out = RigidTransform(RotationMatrix(RollPitchYaw(r, p, y)), [X, Y, Z])

        self.DeclareAbstractOutputPort(name="pose_out",
                                        alloc=lambda: Value(RigidTransform()),
                                        calc=self.PoseOut)
    
    def PoseOut(self, context, output):
        output.set_value(self.pose_out)


class ImageSaver(LeafSystem):
    def __init__(self, dirstr = "test2"):
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


def process_point_cloud(diagram, station, context, cameras):
    plant = station.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)

    pcd = []
    for i in range(3):
        cloud = diagram.GetOutputPort(f"{cameras[i]}_point_cloud").Eval(
            context
        )

        # Crop to region of interest.
        pcd.append(cloud.Crop(lower_xyz=[-0.5, -1.0, 0.01], upper_xyz=[0.5, -0.3, 0.2]))
        # Estimate normals
        pcd[i].EstimateNormals(radius=0.1, num_closest=30)

        # Flip normals toward camera
        camera = plant.GetModelInstanceByName(cameras[i])
        body = plant.GetBodyByName("base", camera)
        X_C = plant.EvalBodyPoseInWorld(plant_context, body)
        pcd[i].FlipNormalsTowardPoint(X_C.translation())

    # Merge point clouds.
    merged_pcd = Concatenate(pcd)

    # Voxelize down-sample.  (Note that the normals still look reasonable)
    return merged_pcd.VoxelizedDownSample(voxel_size=0.005)

def start_scenario(dirstr = "test4"):
    meshcat.ResetRenderMode()

    builder = DiagramBuilder()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dir_path, "scenario_data_grasping.yml")
    scenario = load_scenario(filename=filename)
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

    # Set up teleop widgets.
    meshcat.DeleteAddedControls()
    grasp_selector = builder.AddSystem(GraspSelector())
    
    builder.Connect(
        grasp_selector.GetOutputPort("pose_out"), differential_ik.GetInputPort("X_WE_desired")
    )
    wsg_teleop = builder.AddSystem(WsgButton(meshcat))
    builder.Connect(
        wsg_teleop.get_output_port(0), station.GetInputPort("wsg.position")
    )

    # initialize image writer and save directories
    camera0 = station.GetSubsystemByName("rgbd_sensor_camera0")
    camera1 = station.GetSubsystemByName("rgbd_sensor_camera1")
    camera2 = station.GetSubsystemByName("rgbd_sensor_camera2")
    K = camera0.color_camera_info().intrinsic_matrix()
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

    plant = station.GetSubsystemByName("plant")
    # initialize point cloud output ports
    camera0_pcd = builder.AddSystem(DepthImageToPointCloud(camera0.depth_camera_info()))
    camera1_pcd = builder.AddSystem(DepthImageToPointCloud(camera1.depth_camera_info()))
    camera2_pcd = builder.AddSystem(DepthImageToPointCloud(camera2.depth_camera_info()))

    builder.Connect(station.GetOutputPort("camera0.depth_image"), camera0_pcd.GetInputPort("depth_image"))
    # builder.Connect(station.GetOutputPort("camera0.rgb_image"), camera0_pcd.color_image_input_port())
    camera_pose0 = builder.AddSystem(
        ExtractBodyPose(
            plant.get_body_poses_output_port(), plant.GetBodyIndices(plant.GetModelInstanceByName("camera_main"))[0]
        )
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        camera_pose0.get_input_port(),
    )
    builder.Connect(
        camera_pose0.get_output_port(),
        camera0_pcd.GetInputPort("camera_pose"),
    )

    builder.Connect(station.GetOutputPort("camera1.depth_image"), camera1_pcd.GetInputPort("depth_image"))
    # builder.Connect(station.GetOutputPort("camera1.rgb_image"), camera1_pcd.color_image_input_port())
    camera_pose1 = builder.AddSystem(
        ExtractBodyPose(
            plant.get_body_poses_output_port(), plant.GetBodyIndices(plant.GetModelInstanceByName("camera_1"))[0]
        )
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        camera_pose1.get_input_port(),
    )
    builder.Connect(
        camera_pose1.get_output_port(),
        camera1_pcd.GetInputPort("camera_pose"),
    )

    builder.Connect(station.GetOutputPort("camera2.depth_image"), camera2_pcd.GetInputPort("depth_image"))
    # builder.Connect(station.GetOutputPort("camera2.rgb_image"), camera2_pcd.color_image_input_port())
    camera_pose2 = builder.AddSystem(
        ExtractBodyPose(
            plant.get_body_poses_output_port(), plant.GetBodyIndices(plant.GetModelInstanceByName("camera_2"))[0]
        )
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        camera_pose2.get_input_port(),
    )
    builder.Connect(
        camera_pose2.get_output_port(),
        camera2_pcd.GetInputPort("camera_pose"),
    )

    # Expore point cloud output ports
    builder.ExportOutput(
        camera0_pcd.GetOutputPort("point_cloud"), "camera_main_point_cloud"
    )
    builder.ExportOutput(
        camera1_pcd.GetOutputPort("point_cloud"), "camera_1_point_cloud"
    )
    builder.ExportOutput(
        camera1_pcd.GetOutputPort("point_cloud"), "camera_2_point_cloud"
    )

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

    grasp_btn_presses = 0
    grasp_node = GraspListener()
    meshcat.AddButton("Stop Simulation", "Escape")
    meshcat.AddButton("Compute Grasps")
    print("Press Escape to stop the simulation")
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.03)
        
        if (meshcat.GetButtonClicks("Compute Grasps") > grasp_btn_presses):
            pcd = process_point_cloud(diagram, station, simulator_context, ["camera_main", "camera_1", "camera_2"])
            meshcat.SetObject("cloud", pcd, point_size=0.001)

            grasp_node.compute_candidate_grasps(pcd)
            grasps = grasp_node.get_best_grasps(candidate_num=10)

            print(grasps)
            
            # get end effector pose from grasp pose
            X_GE = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 0)), [0, 0, -0.09])

            ee_grasps = [X_WG.multiply(X_GE) for X_WG in grasps]

            print("end effector poses:", ee_grasps)
            grasp_selector.pose_out = ee_grasps[0] # lol i havent made this collision free traj yet

        grasp_btn_presses = meshcat.GetButtonClicks("Compute Grasps")
    meshcat.DeleteButton("Stop Simulation")


if __name__ == "__main__":
    # Start the visualizer.
    meshcat = StartMeshcat()

    save_dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'tests', 'test_grasping'))
    start_scenario(save_dir_path)