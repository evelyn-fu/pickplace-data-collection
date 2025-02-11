import pyrealsense2 as rs
import numpy as np
from PIL import Image
import argparse
import time
import os

import sys
import select
import tty
import termios
import shutil
import json
import tqdm
import multiprocessing as mp
from functools import partial


def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


def save_image_pair(args, base_dir):
    rgb, depth, timestamp = args
    timestr = f"{timestamp:06d}"

    # Save RGB image
    color_pil = Image.fromarray(rgb, mode="RGB")
    color_pil.save(f"{base_dir}/rgb/{timestr}.png")

    # Save depth image
    depth_pil = Image.fromarray(depth)
    depth_pil.save(f"{base_dir}/depth/{timestr}.png")


def realsense(dirstr="temp", serial="810512062206", config_path=None):
    if os.path.exists(dirstr):
        shutil.rmtree(dirstr)

    os.makedirs(dirstr)
    os.makedirs(dirstr + "/rgb/")
    os.makedirs(dirstr + "/depth/")

    # Create context object to get all devices connected
    ctx = rs.context()

    # Query all devices
    devices = ctx.query_devices()

    print(f"Number of devices found: {len(devices)}")
    for device in devices:
        print(f"Device serial number: {device.get_info(rs.camera_info.serial_number)}")

    # Create a pipeline
    pipeline = rs.pipeline()

    # Serial number of the 5th camera (get this from the output of the previous script)
    fourth_camera_serial = devices[0].get_info(
        rs.camera_info.serial_number
    )  # Assumes 5th camera is at index 3

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_device(str(fourth_camera_serial))

    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)

    # Start streaming
    profile = pipeline.start(config)

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
    w = intr.width
    h = intr.height
    ppx = intr.ppx
    ppy = intr.ppy
    fx = intr.fx
    fy = intr.fy
    K = np.array([[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]])
    np.savetxt(dirstr + "/cam_K.txt", K)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 3  # meters
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Check if can connect to camera.
    frames = pipeline.wait_for_frames()
    print("Connected to camera")

    # Streaming loop
    start_t = time.time()
    print("Press esc to exit, press any other key to pause/unpause")
    save_on = True
    old_settings = termios.tcgetattr(sys.stdin)
    timestamps = []
    color_images = []
    depth_images = []
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            time_ms = int((time.time() - start_t) * 1000)
            timestr = f"{time_ms:06d}"
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = (
                aligned_frames.get_depth_frame()
            )  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data()).copy()
            color_image = np.asanyarray(color_frame.get_data()).copy()

            if save_on:
                # Saving directly is too slow and would reduce the fps.
                timestamps.append(time_ms)
                color_images.append(color_image)
                depth_images.append(depth_image)

            if isData():
                c = sys.stdin.read(1)
                if c == "\x1b":  # x1b is ESC
                    break
                else:
                    save_on = not save_on
                    if save_on:
                        print("unpaused")
                    else:
                        print("paused")

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        pipeline.stop()

        print(f"Saving {len(color_images)} images")
        # Create a partial function with the base directory
        save_func = partial(save_image_pair, base_dir=dirstr)

        # Create a process pool
        with mp.Pool(processes=mp.cpu_count()) as pool:
            # Create iterator of image pairs
            image_pairs = zip(color_images, depth_images, timestamps)

            # Use imap to process images in parallel with progress bar
            list(
                tqdm.tqdm(
                    pool.imap(save_func, image_pairs),
                    total=len(color_images),
                    desc="Saving images",
                )
            )

        # Save metadata
        with open(dirstr + "/metadata.json", "w") as f:
            json.dump(
                {
                    "timestamps": timestamps,
                    "camera_params": {
                        "depth_scale": depth_scale,
                        "intrinsics": {
                            "width": w,
                            "height": h,
                            "ppx": ppx,
                            "ppy": ppy,
                            "fx": fx,
                            "fy": fy,
                        },
                    },
                    "camera_config": {
                        "serial_number": fourth_camera_serial,
                        "streams": {
                            "depth": {
                                "resolution": (intr.width, intr.height),
                                "format": "z16",
                                "fps": 30,
                            },
                            "color": {
                                "resolution": (
                                    color_frame.get_width(),
                                    color_frame.get_height(),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "save_dir",
        default="temp",
        help="directory to save images in",
        nargs="?",
    )
    parser.add_argument(
        "serial",
        default="810512062206",
        help="serial number of camera to save from",
        nargs="?",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="path to JSON configuration file",
    )
    args = parser.parse_args()
    save_dir_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "tests_realsense", args.save_dir)
    )
    realsense(save_dir_path, args.serial, args.config_path)
