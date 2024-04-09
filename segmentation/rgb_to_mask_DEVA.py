import sys
import os
from pathlib import Path
import shutil 
from PIL import Image
import numpy as np

def main(argv):
    '''
    Takes in path to DEVA repository as first argument, text prompt as second argument, 
    path to RGB images as third argument, optionally takes in path to put masked images as fourth argument.
    If the fourth argument is not given, creates a masks directory in the parent directory of the RGB images directory
    '''
    deva_path = argv[0]
    text_prompt = argv[1]
    print("text prompt:", text_prompt)

    rgb_path = Path(argv[2])
    parent_path =  rgb_path.parent.absolute()
    new_rgb_dir = argv[2] + "_filtered"
    if not os.path.exists(new_rgb_dir):
        os.makedirs(new_rgb_dir)
    
    # determine path to save masks
    if len(argv) < 4:
        mask_dir_path = os.path.join(parent_path, "masks")
    else:
        mask_dir_path = argv[3]

    os.system(f"python {deva_path}/demo/demo_with_text.py --chunk_size 4 --img_path {rgb_path} --amp --temporal_setting semionline --size 480 --output {parent_path}/temp --prompt \"{text_prompt}\" --max_num_objects 1")
    shutil.move(f"{parent_path}/temp/Annotations", mask_dir_path)
    shutil.rmtree(f"{parent_path}/temp")

    for image_path in os.listdir(mask_dir_path):
        input_path = os.path.join(mask_dir_path, image_path)
        label_image = np.array(Image.open(input_path))
        mask = np.uint8(np.where(np.sum(label_image, axis=-1) == 0, 0, 255))
        if np.any(mask): # filter out images where segmentation failed
            mask_pil = Image.fromarray(mask)
            mask_pil.save(input_path)
            old_rgb_path = os.path.join(rgb_path, image_path)
            new_rgb_path = os.path.join(new_rgb_dir, image_path)
            shutil.copy2(old_rgb_path, new_rgb_path)

if __name__ == "__main__":
   main(sys.argv[1:])