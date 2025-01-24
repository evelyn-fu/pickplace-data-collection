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

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


def realsense(dirstr="temp", serial="811313020078"):
    if os.path.exists(dirstr):
        shutil.rmtree(dirstr)
        
    os.makedirs(dirstr)
    os.makedirs(dirstr+"/rgb/")
    os.makedirs(dirstr+"/depth/")

    # Create context object to get all devices connected
    ctx = rs.context()

    # Query all devices
    devices = ctx.query_devices()

    print(f"Number of devices found: {len(devices)}")
    for device in devices:
        print(f"Device serial number: {device.get_info(rs.camera_info.serial_number)}")

    # Create a pipeline
    pipeline = rs.pipeline()
    
    # Serial number of the 4th camera (get this from the output of the previous script)
    fourth_camera_serial = devices[0].get_info(rs.camera_info.serial_number)  # Assumes 4th camera is at index 3

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_device(str(fourth_camera_serial))

    # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)

    # Start streaming
    profile = pipeline.start(config)
    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    w = intr.width
    h = intr.height
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
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    start_t = time.time()
    print("Press esc to exit, press any other key to pause/unpause")
    save_on = True
    old_settings = termios.tcgetattr(sys.stdin)
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
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if save_on:
                color_pil = Image.fromarray(color_image, mode='RGB')
                color_pil.save(dirstr+"/rgb/"+timestr+".png")

                depth_pil = Image.fromarray(depth_image)
                depth_pil.save(dirstr+"/depth/"+timestr+".png")

            if isData():
                c = sys.stdin.read(1)
                if c == '\x1b':         # x1b is ESC
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "save_dir",
        default="temp",
        help="directory to save images in",
        nargs='?',
    )
    parser.add_argument(
        "serial",
        default="811313020078",
        help="serial number of camera to save from",
        nargs='?',
    )
    args = parser.parse_args()
    save_dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'tests_realsense', args.save_dir))
    realsense(save_dir_path, args.serial)