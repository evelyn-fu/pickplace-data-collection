import sys
import os
from pathlib import Path
import shutil 

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
    
    # determine path to save masks
    if len(argv) < 4:
        mask_dir_path = os.path.join(parent_path, "masks")
    else:
        mask_dir_path = argv[3]

    os.system(f"python {deva_path}/demo/demo_with_text.py --chunk_size 4 --img_path {rgb_path} --amp --temporal_setting semionline --size 480 --output {parent_path}/temp --prompt \"{text_prompt}\" --max_num_objects 1")
    shutil.move(f"{parent_path}/temp/Annotations", mask_dir_path)
    shutil.rmtree(f"{parent_path}/temp")

if __name__ == "__main__":
   main(sys.argv[1:])