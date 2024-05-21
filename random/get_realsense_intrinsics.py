import pyrealsense2 as rs
import numpy as np

if __name__ == "__main__":
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    profile = pipeline.start(config)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    w = intr.width
    h = intr.height
    ppx = intr.ppx
    ppy = intr.ppy
    fx = intr.fx
    fy = intr.fy
    K = np.array([[fx, 0.0, ppx],
                    [0.0, fy, ppy],
                    [0.0, 0.0, 1.0]])
    
    print("rgb intrinsics", K)
    print("rgb width, height", w, h)

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
    
    print("depth intrinsics", K)
    print("depth width, height", w, h)
    pipeline.stop()