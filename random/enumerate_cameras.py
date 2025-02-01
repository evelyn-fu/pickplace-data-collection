import pyrealsense2 as rs

# Create context object to get all devices connected
ctx = rs.context()

# Query all devices
devices = ctx.query_devices()

print(f"Number of devices found: {len(devices)}")
for device in devices:
    print(f"Device serial number: {device.get_info(rs.camera_info.serial_number)}")
    
# Serial number of the 4th camera (get this from the output of the previous script)
fourth_camera_serial = devices[3].get_info(rs.camera_info.serial_number)  # Assumes 4th camera is at index 3

# Now set up a pipeline to access the 4th camera
pipeline = rs.pipeline()
config = rs.config()

# Specify the 4th camera by its serial number
config.enable_device(fourth_camera_serial)

# Start the pipeline
pipeline.start(config)

# Capture frames
try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        # Do something with the frames, e.g., process the depth or color data
        print(f"Capturing frames from camera {fourth_camera_serial}")
except KeyboardInterrupt:
    print("Program interrupted.")
finally:
    pipeline.stop()
