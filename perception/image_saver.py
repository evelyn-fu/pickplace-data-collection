import numpy as np
import copy
from PIL import Image
from pydrake.systems.framework import LeafSystem
from pydrake.systems.sensors import CameraInfo
from pydrake.systems.sensors import (
    ImageRgba8U,
    ImageDepth16U,
    ImageLabel16I,
    ImageDepth32F,
    ConvertDepth32FTo16U
)
from pydrake.common.value import Value
from pydrake.common.value import (
    Value
)
from pydrake.math import (
    RigidTransform,
)
import pyrealsense2 as rs
import sys
import tty
import termios
from planning.two_grasp_display_planner import PlannerState

class ImageSaver(LeafSystem):
    def __init__(
            self, 
            depth_format="16U", 
            dirstr = "test2", 
            labels=False, 
            camera_info=False, 
            ob_in_cam=False, 
            object_index=None, 
            camera_index=None
        ):
        super().__init__()

        self.depth_format = depth_format
        self.dirstr = dirstr
        self.labels = labels
        self.ob_in_cam = ob_in_cam
        self.object_index = object_index
        self.camera_index = camera_index
        self.DeclareAbstractInputPort(name="rgb_in",
                                      model_value=Value(ImageRgba8U()))
        
        if depth_format == "32F":
            self.DeclareAbstractInputPort(name="depth_in",
                                        model_value=Value(ImageDepth32F()))
        else:
            self.DeclareAbstractInputPort(name="depth_in",
                                        model_value=Value(ImageDepth16U()))
        
        if labels:
            self.DeclareAbstractInputPort(name="label_in",
                                        model_value=Value(ImageLabel16I()))
        
        if camera_info:
            camera_info_model = CameraInfo(width=640, height=480, fov_y=np.pi / 4.0)
            self.DeclareAbstractInputPort(name="rgb_info_in",
                                        model_value=Value(camera_info_model))
            self.DeclareAbstractInputPort(name="depth_info_in",
                                        model_value=Value(camera_info_model))
            self.DeclareInitializationPublishEvent(self.Initialize)
        
        if ob_in_cam:
            self.DeclareAbstractInputPort(
                "body_poses", model_value=Value([RigidTransform()])
            )
        
        self.DeclareAbstractInputPort("planner_state", model_value=Value(PlannerState.START))

        # Calling `ForcePublish()` will trigger the callback.
        self.DeclareForcedPublishEvent(self.Publish)

        # Publish at 33 fps
        self.DeclarePeriodicPublishEvent(period_sec=0.03,
                                         offset_sec=0,
                                         publish=self.Publish)
        
    def Publish(self, context):
        no_save_states = [
            PlannerState.WAIT_FOR_OBJECTS_TO_SETTLE,
            PlannerState.START,
            PlannerState.SCANNING1, 
            PlannerState.SCANNING2]
        if self.GetInputPort("planner_state").Eval(context) in no_save_states:
            return
        
        time_ms = int(context.get_time() * 1000)
        timestr = f"{time_ms:06d}"
        
        # color
        color = self.GetInputPort("rgb_in").Eval(context).data
        color_pil = Image.fromarray(color)
        color_pil.save(self.dirstr+"/rgb_alpha/"+timestr+".png")

        # remove alpha
        color_no_alpha = color[:, :, :3]
        color_no_alpha_pil = Image.fromarray(color)
        color_no_alpha_pil.save(self.dirstr+"/rgb/"+timestr+".png")

        # depth
        if self.depth_format != "16U":
            depth_32f = self.GetInputPort("depth_in").Eval(context)

            depth_16u = ImageDepth16U()
            ConvertDepth32FTo16U(depth_32f, depth_16u)
        else:
            depth_16u = self.GetInputPort("depth_in").Eval(context)

        depth = copy.deepcopy(depth_16u.data.squeeze())

        # cap depth at 3000mm
        depth[depth > 3000] = 3000
        depth_pil = Image.fromarray(depth)
        depth_pil.save(self.dirstr+"/depth/"+timestr+".png")

        # labels
        if self.labels:
            label_image = copy.deepcopy(
                self.GetInputPort("label_in").Eval(context).data.squeeze()
            )
        
            # get mask for bottle
            object_labels = np.unique(label_image)
            masks = [
                np.uint8(np.where(label_image == label, 255, 0)) for label in object_labels
            ]
            mask_pil = Image.fromarray(masks[0])
            mask_pil.save(self.dirstr+"/masks/"+timestr+".png")

            gripper_mask_pil = Image.fromarray(masks[1])
            gripper_mask_pil.save(self.dirstr+"/gripper_masks/"+timestr+".png")
        
        # ob_in_cam pose
        if self.ob_in_cam:
            o2w = self.GetInputPort("body_poses").Eval(context)[int(self.object_index)]
            c2w = self.GetInputPort("body_poses").Eval(context)[int(self.camera_index)]

            o2c = c2w.inverse() @ o2w
            T = o2c.GetAsMatrix4()
            np.savetxt(self.dirstr+"/ob_in_cam/"+timestr+".txt", T)

    def Initialize(self, context):
        rgb_info = self.GetInputPort("rgb_info_in").Eval(context)
        depth_info = self.GetInputPort("depth_info_in").Eval(context)

        K_color = rgb_info.intrinsic_matrix()
        K_depth = depth_info.intrinsic_matrix()

        np.savetxt(self.dirstr+"/cam_K.txt", K_color)
        np.savetxt(self.dirstr+"/cam_K_depth.txt", K_depth)


class RealsenseImageSaver(LeafSystem):
    def __init__(self, dirstr = "test_realsense"):
        super().__init__()

        self.dirstr = dirstr

        # Create a pipeline
        self.pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.rgb8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        # Start streaming
        profile = self.pipeline.start(config)
        intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        ppx = intr.ppx
        ppy = intr.ppy
        fx = intr.fx
        fy = intr.fy
        K = np.array([[fx, 0.0, ppx],
                                    [0.0, fy, ppy],
                                    [0.0, 0.0, 1.0]])
        np.savetxt(dirstr+"/cam_K.txt", K)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters = 3 # meters

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
        except:
            raise Exception("RealsenseImageSaver init failed.")

        # Calling `ForcePublish()` will trigger the callback.
        self.DeclareForcedPublishEvent(self.Publish)

        # Publish at 33 fps
        self.DeclarePeriodicPublishEvent(period_sec=0.03,
                                         offset_sec=0,
                                         publish=self.Publish)
    
    def __del__(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        self.pipeline.stop()

    def Publish(self, context):
        time_ms = int(context.get_time() * 1000)
        timestr = f"{time_ms:06d}"

        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        color_pil = Image.fromarray(color_image, mode='RGB')
        color_pil.save(self.dirstr+"/rgb/"+timestr+".png")

        depth_pil = Image.fromarray(depth_image)
        depth_pil.save(self.dirstr+"/depth/"+timestr+".png")

