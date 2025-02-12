import numpy as np
import copy
import pyrealsense2 as rs
import json
import tqdm
import multiprocessing as mp
from functools import partial
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
import os
import time
import termios
from planning.two_grasp_display_planner import PlannerState

def save_image_pair(args, base_dir):
    rgb, depth, timestamp = args
    timestr = f"{timestamp:06d}"

    # Save RGB image
    color_pil = Image.fromarray(rgb, mode="RGB")
    color_pil.save(f"{base_dir}/rgb/{timestr}.png")

    # Save depth image
    depth_pil = Image.fromarray(depth)
    depth_pil.save(f"{base_dir}/depth/{timestr}.png")

class ImageSaver(LeafSystem):
    def __init__(
            self, 
            depth_format="16U", 
            dirstr = "temp", 
            num_objects = 1, 
            labels=False, 
            camera_info=False, 
            ob_in_cam=False, 
            object_index=None, 
            camera_index=None,
            use_hardware=False,
            camera_num=0,
            config_path=None
        ):
        super().__init__()

        self.timestamps = []
        self.color_images = []
        self.depth_images = []
        self.depth_format = depth_format
        self.dirstr = dirstr
        self.labels = labels
        self.ob_in_cam = ob_in_cam
        self.object_index = object_index
        self.camera_index = camera_index
        self.use_hardware = use_hardware
        self.object_saved = False

        for i in range(num_objects):
            combined_dir = os.path.join(dirstr, f"obj_{i}")
            if not os.path.exists(combined_dir):
                os.makedirs(combined_dir)
            if not os.path.exists(combined_dir+"/rgb/"):
                os.makedirs(combined_dir+"/rgb/")
            if not os.path.exists(combined_dir+"/depth/"):
                os.makedirs(combined_dir+"/depth/")
            if not use_hardware:
                if not os.path.exists(combined_dir+"/ob_in_cam/"):
                    os.makedirs(combined_dir+"/ob_in_cam/")
                if not os.path.exists(combined_dir+"/masks/"):
                    os.makedirs(combined_dir+"/masks/")
                if not os.path.exists(combined_dir+"/gripper_masks/"):
                    os.makedirs(combined_dir+"/gripper_masks/")
        self.object_ind = 0

        if not use_hardware:
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
        else:
            # Create context object to get all devices connected
            ctx = rs.context()

            # Query all devices
            devices = ctx.query_devices()

            print(f"Number of devices found: {len(devices)}")
            i = 0
            for device in devices:
                print(f"{i}th Device serial number: {device.get_info(rs.camera_info.serial_number)}")
                i += 1

            # Create a pipeline
            self.pipeline = rs.pipeline()

            # Serial number of the desired camera
            self.camera_serial = devices[camera_num].get_info(
                rs.camera_info.serial_number
            ) 

            # Create a config and configure the pipeline to stream
            #  different resolutions of color and depth streams
            config = rs.config()
            config.enable_device(str(self.camera_serial))

            device_name = devices[0].get_info(rs.camera_info.name)
            if "L5" in device_name:
                config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
            else:
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
                # config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

            # Start streaming
            profile = self.pipeline.start(config)

            # Load JSON configuration if provided
            json_config = None
            if config_path is not None:
                with open(config_path, "r") as f:
                    json_config = json.load(f)
                print("loaded config")

            # Apply the specific config
            if config_path is not None:
                # Check device type
                device_name = devices[0].get_info(rs.camera_info.name)

                if "L5" in device_name:
                    # For L515, apply settings directly to the sensor
                    sensors = devices[0].query_sensors()
                    for sensor in sensors:
                        if "Depth" in sensor.get_info(rs.camera_info.name):
                            # Apply parameters from the config
                            for param_name, param_value in json_config["parameters"].items():
                                try:
                                    # Convert parameter name to pyrealsense2 format
                                    option_name = param_name.lower().replace(" ", "_")
                                    option = getattr(rs.option, option_name)
                                    sensor.set_option(option, float(param_value))
                                    print(f"Set {option_name} to {param_value}")
                                except Exception as e:
                                    print(f"Failed to set option {param_name}: {e}")
                    print("Loaded JSON configuration for L515")
                else:
                    try:
                        # For D400 series, use advanced mode
                        advanced_mode = rs.rs400_advanced_mode(devices[0])
                        advanced_mode.load_json(str(json_config))
                        print("Loaded JSON configuration for D400 series")
                    except Exception as e:
                        print(f"Failed to load advanced mode configuration: {e}")

            intr = (
                profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            )
            self.w = intr.width
            self.h = intr.height
            self.ppx = intr.ppx
            self.ppy = intr.ppy
            self.fx = intr.fx
            self.fy = intr.fy
            K = np.array([[self.fx, 0.0, self.ppx], [0.0, self.fy, self.ppy], [0.0, 0.0, 1.0]])
            np.savetxt(dirstr + "/cam_K.txt", K)

            color_intr = (
                profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            )
            self.color_w = color_intr.width
            self.color_h = color_intr.height

            # Getting the depth sensor's depth scale (see rs-align example for explanation)
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print("Depth Scale is: ", self.depth_scale)

            # We will be removing the background of objects more than
            #  clipping_distance_in_meters meters away
            clipping_distance_in_meters = 3  # meters
            clipping_distance = clipping_distance_in_meters / self.depth_scale

            # Create an align object
            # rs.align allows us to perform alignment of depth frames to others frames
            # The "align_to" is the stream type to which we plan to align depth frames.
            align_to = rs.stream.color
            self.align = rs.align(align_to)

            # Check if can connect to camera.
            frames = self.pipeline.wait_for_frames()
            print("Connected to camera")

            self.old_settings = termios.tcgetattr(sys.stdin)

            try:
                tty.setcbreak(sys.stdin.fileno())
            except:
                raise Exception("ImageSaver on hardware init failed.")

        
        self.DeclareAbstractInputPort("planner_state", model_value=Value(PlannerState.START))

        # Calling `ForcePublish()` will trigger the callback.
        self.DeclareForcedPublishEvent(self.Publish)

        # Publish at 33 fps
        self.DeclarePeriodicPublishEvent(period_sec=0.03,
                                         offset_sec=0,
                                         publish=self.Publish)
        
    def Publish(self, context):
        save_states = [
            # PlannerState.GO_TO_PREGRASP1,
            PlannerState.GRASP1,
            # PlannerState.GO_HOME1,
            # PlannerState.GO_TO_PREGRASP2,
            PlannerState.GRASP2,
            # PlannerState.GO_HOME2
        ]
        mode = self.GetInputPort("planner_state").Eval(context)
        if mode not in save_states:
            if mode == PlannerState.RESET and not self.object_saved:
                self.save_imgs()
                self.object_saved = True
            if mode == PlannerState.START:
                self.object_saved = False
            return

        time_ms = int(context.get_time() * 1000)
        timestr = f"{time_ms:06d}"
        
        if not self.use_hardware:
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
        else:
            # Get frameset of color and depth
            frames = self.pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            aligned_depth_frame = (
                aligned_frames.get_depth_frame()
            )  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                return

            depth_image = np.asanyarray(aligned_depth_frame.get_data()).copy()
            # Convert depth to millimeters
            depth_image = (depth_image * self.depth_scale * 1000).astype(np.uint16)
            color_image = np.asanyarray(color_frame.get_data()).copy()

            self.timestamps.append(time_ms)
            self.color_images.append(color_image)
            self.depth_images.append(depth_image)


    def Initialize(self, context):
        if not self.use_hardware:
            rgb_info = self.GetInputPort("rgb_info_in").Eval(context)
            depth_info = self.GetInputPort("depth_info_in").Eval(context)

            K_color = rgb_info.intrinsic_matrix()
            K_depth = depth_info.intrinsic_matrix()

            np.savetxt(self.dirstr+"/cam_K.txt", K_color)
            np.savetxt(self.dirstr+"/cam_K_depth.txt", K_depth)
    
    def __del__(self):
        if not self.use_hardware:
            return
        
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        self.pipeline.stop()

    def save_imgs(self):
        print(f"Saving {len(self.color_images)} images for object {self.object_ind}")
        combined_dir = os.path.join(self.dirstr, f"obj_{self.object_ind}")
        # Create a partial function with the base directory
        save_func = partial(save_image_pair, base_dir=combined_dir)

        # Create a process pool
        with mp.Pool(processes=mp.cpu_count()) as pool:
            # Create iterator of image pairs
            image_pairs = zip(self.color_images, self.depth_images, self.timestamps)

            # Use imap to process images in parallel with progress bar
            list(
                tqdm.tqdm(
                    pool.imap(save_func, image_pairs),
                    total=len(self.color_images),
                    desc="Saving images",
                )
            )
        self.color_images = []
        self.depth_images = []
        self.timestamps = []

        # Save metadata
        with open(combined_dir + "/metadata.json", "w") as f:
            json.dump(
                {
                    "timestamps": self.timestamps,
                    "camera_params": {
                        "depth_scale": self.depth_scale,
                        "intrinsics": {
                            "width": self.w,
                            "height": self.h,
                            "ppx": self.ppx,
                            "ppy": self.ppy,
                            "fx": self.fx,
                            "fy": self.fy,
                        },
                    },
                    "camera_config": {
                        "serial_number": self.camera_serial,
                        "streams": {
                            "depth": {
                                "resolution": (self.w, self.h),
                                "format": "z16",
                                "fps": 30,
                            },
                            "color": {
                                "resolution": (
                                    self.color_w,
                                    self.color_h,
                                ),
                                "format": "rgb8",
                                "fps": 30,
                            },
                        },
                    },
                    "current_time": time.time(),
                },
                f,
            )
        self.object_ind += 1


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

