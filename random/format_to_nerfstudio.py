# Script to format image data and object pose in camera transforms to nerfstudio format in object frame

from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import sys
from PIL import Image 
import json
from pathlib import Path

def transform_pose_opengl_to_opencv(pose):
    # Inverse transform is identical
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    return pose @ flip_yz

def main(argv):
    data_path = Path(argv[0])
    transforms_path = os.path.join(data_path, "ob_in_cam/")
    rgbs_path = os.path.join(data_path, "rgb/")
    camera_intrinsics = os.path.join(data_path, "cam_K.txt")

    K = np.loadtxt(camera_intrinsics, delimiter=' ')

    transforms_json = {
        "camera_model": "OPENCV",
        "fl_x": K[0, 0], 
        "fl_y": K[1, 1], 
        "cx": K[0, 2],
        "cy": K[1, 2],
        "w": 0, 
        "h": 0,
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "aabb_scale": 1.0,
        "frames": []
    }
        
    count = 0
    for image_path in sorted(os.listdir(rgbs_path)):
        if count == 0:
            img = Image.open(os.path.join(rgbs_path, image_path))
            transforms_json["w"] = img.width
            transforms_json["h"] = img.height

        transforms_filename = os.path.splitext(image_path)[0] + ".txt"
        
        pose_opencv = np.loadtxt(os.path.join(transforms_path, transforms_filename))
        pose_opencv = np.linalg.inv(pose_opencv)
        pose_opengl = transform_pose_opengl_to_opencv(pose_opencv)
        
        frame = {
            "file_path": "rgb/" + image_path,
            "depth_file_path": "depth/" + image_path,
            "mask_path": "masks/" + image_path,
            "transform_matrix": pose_opengl.tolist()
        }

        transforms_json["frames"].append(frame)
        count += 1
    
    print(f"Processed {count} frames")

    with open(os.path.join(data_path, "transforms.json"), 'w') as fp:
        json.dump(transforms_json, fp)

if __name__ == "__main__":
   main(sys.argv[1:]) # arguments: [data directory]
