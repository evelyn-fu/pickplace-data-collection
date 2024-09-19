# Script to format image data and object pose in camera transforms to nerfstudio format in object frame

from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import sys
from PIL import Image 
import json
from pathlib import Path

def main(argv):
    '''
    Takes in path to directory of images of same (h,w) as first argument, desired width to scale to as
    second argument, desired height as third argument. Optionally takes in output directory name as 4th argument.
    If 4th argument is not provided, will create a new directory called [input directory]_scaled
    '''
    data_path = Path(argv[0])
    output_w = int(argv[1])
    output_h = int(argv[2])
    input_w = 0
    input_h = 0

    # determine path to save scaled imgs
    if len(argv) < 4:
        output_image_dir = argv[0]
        if output_image_dir[-1] == "/":
            output_image_dir = output_image_dir[:-1]
        output_image_dir += "_scaled"
    else:
        output_image_dir = argv[3]

    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    count = 0
    for image_path in sorted(os.listdir(data_path)):
        if count == 0:
            img = Image.open(os.path.join(data_path, image_path))
            input_w = img.width
            input_h = img.height

        img = Image.open(os.path.join(data_path, image_path))
    
        # Resize the image to the new dimensions
        scaled_img = img.resize((output_w, output_h), Image.Resampling.LANCZOS)
        
        # Save the scaled image
        scaled_img.save(os.path.join(output_image_dir, image_path))

        count += 1
    

if __name__ == "__main__":
   main(sys.argv[1:]) # arguments: [data directory]
