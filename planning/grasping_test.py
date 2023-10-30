import numpy as np
import os
import copy
from PIL import Image
from scipy.spatial.transform import Rotation as R
from grasp import GraspListener
from trajectories import (
    MakeGripperCommandTrajectory,
    MakeGripperFrames,
    MakeGripperPoseTrajectory,
)
from pydrake.geometry import (
    StartMeshcat,
    RenderLabel,
    Role,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.systems.sensors import (
    ImageRgba8U,
    ImageDepth16U,
    ImageLabel16I,
)
from pydrake.common.value import Value
from pydrake.perception import (
    Concatenate,
    DepthImageToPointCloud,
    PointCloud
)
from pydrake.math import (
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
)
from pydrake.common.value import (
    Value,
    AbstractValue
)
from pydrake.trajectories import (
    PiecewisePose,
    PiecewisePolynomial
)

from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import AddIiwaDifferentialIK, ExtractBodyPose
from manipulation.station import MakeHardwareStation, load_scenario
from enum import Enum

class PlannerState(Enum):
    WAIT_FOR_OBJECTS_TO_SETTLE = 1
    GRASP1 = 2
    GRASP2 = 3

default_home_pose = RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, np.pi/2)), [0.0, -0.5, 0.5]) # arm out of the way of depth cameras

default_display_traj = []

default_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, np.pi/2)), [0.0, -0.5, 0.4]))
default_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, np.pi/4)), [0.0, -0.5, 0.4]))
default_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, np.pi/2)), [0.0, -0.5, 0.4]))
default_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, np.pi)), [0.0, -0.5, 0.4]))
default_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, 3 * np.pi / 2)), [0.0, -0.5, 0.4]))
default_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, 7 * np.pi / 4)), [0.0, -0.5, 0.4]))
default_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, 3 * np.pi / 2)), [0.0, -0.5, 0.4]))
default_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, np.pi)), [0.0, -0.5, 0.4]))
default_display_traj.append(RigidTransform(RotationMatrix(RollPitchYaw(np.pi, 0.0, np.pi/2)), [0.0, -0.5, 0.4]))


class Planner(LeafSystem):
    def __init__(
            self, 
            plant, 
            camera_body_indices,
            meshcat
        ):
        LeafSystem.__init__(self)

        model_point_cloud = AbstractValue.Make(PointCloud(0))
        self.DeclareAbstractInputPort("cloud0_W", model_point_cloud)
        self.DeclareAbstractInputPort("cloud1_W", model_point_cloud)
        self.DeclareAbstractInputPort("cloud2_W", model_point_cloud)
        self._camera_body_indices = camera_body_indices

        self._gripper_body_index = plant.GetBodyByName("body").index()
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )

        self._mode_index = self.DeclareAbstractState(
            AbstractValue.Make(PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE)
        )
        self._traj_X_G_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePose())
        )
        self._traj_wsg_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )
        self._times_index = self.DeclareAbstractState(
            AbstractValue.Make({"initial": 0.0})
        )

        self.DeclareAbstractOutputPort(
            "X_WG",
            lambda: AbstractValue.Make(RigidTransform()),
            self.CalcGripperPose,
        )
        self.DeclareVectorOutputPort("wsg_position", 1, self.CalcWsgPosition)

        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)

        self.grasp_node = GraspListener()
        self.meshcat = meshcat

    def Update(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        current_time = context.get_time()
        times = context.get_abstract_state(int(self._times_index)).get_value()

        if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            if current_time - times["initial"] > 1.0:
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.GRASP1)
                self.Plan(context, state)
            return
        if mode == PlannerState.GRASP1:
            traj_X_G = context.get_abstract_state(
                int(self._traj_X_G_index)
            ).get_value()
            if traj_X_G.get_number_of_segments() > 0 and (not traj_X_G.is_time_in_range(context.get_time())):
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(PlannerState.GRASP2)
                self.Plan(context, state)
            return

    def Plan(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()

        X_G = {
            "initial": self.get_input_port(3).Eval(context)[
                int(self._gripper_body_index)
            ],
            "end": default_home_pose
        }

        # Get pcd and select grasp
        body_poses = self.get_input_port(3).Eval(context)
        pcd = []
        for i in range(3):
            cloud = self.get_input_port(i).Eval(context)

            # Crop to region of interest.
            pcd.append(cloud.Crop(lower_xyz=[-0.5, -1.0, 0.01], upper_xyz=[0.5, -0.3, 0.2]))
            # Estimate normals
            pcd[i].EstimateNormals(radius=0.1, num_closest=30)

            # Flip normals toward camera
            X_WC = body_poses[self._camera_body_indices[i]]
            pcd[i].FlipNormalsTowardPoint(X_WC.translation())
        merged_pcd = Concatenate(pcd)

        down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)
        self.meshcat.SetObject("cloud", down_sampled_pcd, point_size=0.001)

        pcd_points = down_sampled_pcd.xyzs().T
        principal_component, secondary_component, minor_component = compute_principal_minor_components(pcd_points)

        # visualize axes, principal axis is z axis (blue), minor axis is x axis (red)
        z_axis, x_axis = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
        rot_principal_component_to_axes, _ = R.align_vectors(
            np.array([z_axis, x_axis]), np.stack([principal_component, minor_component])
        )
        com = np.mean(pcd_points, axis=0)
        AddMeshcatTriad(self.meshcat, "principal axis", 
                        X_PT=RigidTransform(RotationMatrix(rot_principal_component_to_axes.as_matrix().T),
                        [com[0], com[1], com[2]]))

        if mode == PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE:
            self.grasp_node.compute_candidate_grasps(down_sampled_pcd, align_grasp_axis=principal_component)
        else:
            self.grasp_node.compute_candidate_grasps(down_sampled_pcd, align_grasp_axis=secondary_component)

        grasps = self.grasp_node.get_best_grasps(candidate_num=1)

        print(grasps)
        
        # get end effector pose from grasp pose
        X_GE = RigidTransform(RotationMatrix(RollPitchYaw(0, 0, 0)), [0, 0, -0.09])

        ee_grasps = [X_WG.multiply(X_GE) for X_WG in grasps]

        X_G["pick"] = ee_grasps[0]

        X_G["display_traj"] = default_display_traj
        X_G, times = MakeGripperFrames(X_G, t0=context.get_time())
        print(
            f"Planned {times['postplace'] - times['initial']} second trajectory in mode {mode} at time {context.get_time()}."
        )
        state.get_mutable_abstract_state(int(self._times_index)).set_value(
            times
        )

        if False:  # Useful for debugging
            AddMeshcatTriad(self.meshcat, "X_Oinitial", X_PT=X_O["initial"])
            AddMeshcatTriad(self.meshcat, "X_Gprepick", X_PT=X_G["prepick"])
            AddMeshcatTriad(self.meshcat, "X_Gpick", X_PT=X_G["pick"])
            AddMeshcatTriad(self.meshcat, "X_Gplace", X_PT=X_G["place"])

        traj_X_G = MakeGripperPoseTrajectory(X_G, times)
        traj_wsg_command = MakeGripperCommandTrajectory(times)

        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(
            traj_X_G
        )
        state.get_mutable_abstract_state(int(self._traj_wsg_index)).set_value(
            traj_wsg_command
        )

    def start_time(self, context):
        return (
            context.get_abstract_state(int(self._traj_X_G_index))
            .get_value()
            .start_time()
        )

    def end_time(self, context):
        return (
            context.get_abstract_state(int(self._traj_X_G_index))
            .get_value()
            .end_time()
        )

    def CalcGripperPose(self, context, output):
        context.get_abstract_state(int(self._mode_index)).get_value()

        traj_X_G = context.get_abstract_state(
            int(self._traj_X_G_index)
        ).get_value()
        if traj_X_G.get_number_of_segments() > 0 and traj_X_G.is_time_in_range(
            context.get_time()
        ):
            # Evaluate the trajectory at the current time, and write it to the
            # output port.
            output.set_value(
                context.get_abstract_state(int(self._traj_X_G_index))
                .get_value()
                .GetPose(context.get_time())
            )
            return

        # Command the current position (note: this is not particularly good if the velocity is non-zero)
        output.set_value(
            default_home_pose
        )

    def CalcWsgPosition(self, context, output):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        opened = np.array([0.107])
        np.array([0.0])

        traj_wsg = context.get_abstract_state(
            int(self._traj_wsg_index)
        ).get_value()
        if traj_wsg.get_number_of_segments() > 0 and traj_wsg.is_time_in_range(
            context.get_time()
        ):
            # Evaluate the trajectory at the current time, and write it to the
            # output port.
            output.SetFromVector(traj_wsg.value(context.get_time()))
            return

        # Command the open position
        output.SetFromVector([opened])


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

def compute_principal_minor_components(pcd):
    cov = np.cov(pcd.T)
    eigval, eigvec = np.linalg.eig(cov)

    order = eigval.argsort()
    principal_component = eigvec[:, order[-1]]
    secondary_component = eigvec[:, order[1]]
    minor_component = eigvec[:, order[0]]

    return principal_component, secondary_component, minor_component

def start_scenario(dirstr = "test4"):
    meshcat.ResetRenderMode()

    builder = DiagramBuilder()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(dir_path, "scenario_data_grasping.yml")
    scenario = load_scenario(filename=filename)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    plant = station.GetSubsystemByName("plant")

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

    # Set up planner
    planner = builder.AddSystem(Planner(
            plant, 
            camera_body_indices=[
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera_main"))[
                    0
                ],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera_1"))[
                    0
                ],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera_2"))[
                    0
                ]
            ],
            meshcat=meshcat,))
    
    builder.Connect(planner.GetOutputPort("X_WG"), differential_ik.get_input_port(0))
    builder.Connect(
        planner.GetOutputPort("wsg_position"),
        station.GetInputPort("wsg.position"),
    )
    builder.Connect(
        camera0_pcd.GetOutputPort("point_cloud"),
        planner.GetInputPort("cloud0_W"),
    )
    builder.Connect(
        camera1_pcd.GetOutputPort("point_cloud"),
        planner.GetInputPort("cloud1_W"),
    )
    builder.Connect(
        camera2_pcd.GetOutputPort("point_cloud"),
        planner.GetInputPort("cloud2_W"),
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        planner.GetInputPort("body_poses"),
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

    meshcat.AddButton("Stop Simulation", "Escape")
    print("Press Escape to stop the simulation")
    while meshcat.GetButtonClicks("Stop Simulation") < 1:
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.03)

    meshcat.DeleteButton("Stop Simulation")


if __name__ == "__main__":
    # Start the visualizer.
    meshcat = StartMeshcat()

    save_dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'tests', 'test_grasping'))
    start_scenario(save_dir_path)