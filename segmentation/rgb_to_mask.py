from PIL import Image
from lang_sam import LangSAM
import sys
import os
import numpy as np
from pathlib import Path

def save_mask(mask_np, filename):
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_image.save(filename)

def main(argv):
    '''
    Takes in text prompt as first argument, path to RGB images as second argument, 
    optionally takes in path to put masked images as third argument.
    If the third argument is not given, creates a masks directory in the parent directory of the RGB images directory
    '''
    model = LangSAM()

    text_prompt = argv[0]
    rgb_directory = os.fsencode(argv[1])
    
    # determine path to save masks
    if len(argv) < 3:
        rgb_path = Path(argv[1])
        parent_path =  rgb_path.parent.absolute()
        mask_dir_path = os.path.join(parent_path, "masks_sam")
    else:
        mask_dir_path = argv[2]

    # make directory if not exists
    Path(mask_dir_path).mkdir(parents=True, exist_ok=True)

    for file in os.listdir(rgb_directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"): 
            image_pil = Image.open(filename).convert("RGB")
            masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
            if len(masks) == 0:
                raise Exception(f"No objects of the '{text_prompt}' prompt detected in the image {filename}.")
            else:
                # Convert masks to numpy arrays
                masks_np = [mask.squeeze().cpu().numpy() for mask in masks]

                # Save the first mask
                mask_path = os.path.join(mask_dir_path, filename)
                save_mask(masks_np[0], mask_path)

if __name__ == "__main__":
   main(sys.argv[1:])