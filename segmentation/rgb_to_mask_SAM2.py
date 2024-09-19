import sys
import os
from pathlib import Path
import shutil 
from PIL import Image
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

# Function to convert PNG to JPG
def convert_png_to_jpg(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            # Open the PNG image
            png_image_path = os.path.join(input_dir, filename)
            img = Image.open(png_image_path)

            # Convert the PNG to RGB (to remove alpha channel)
            rgb_img = img.convert("RGB")

            # Save the image as JPG in the output directory
            jpg_image_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.jpg")
            rgb_img.save(jpg_image_path, "JPEG")
            print(f"Converted {filename} to JPG and saved as {jpg_image_path}")

def save_mask(mask_dir_path, frame_names, frame_idx, masks):
    mask_path = os.path.join(mask_dir_path, f"{frame_names[frame_idx]}.png")
    mask = (masks[0] > 0.0).cpu().numpy()
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * 255
    mask_image = np.squeeze(mask_image, axis=2)
    mask_pil = Image.fromarray(mask_image.astype(np.uint8))
    mask_pil.save(mask_path)

def main(argv):
    '''
    Takes in path to checkpoint as first argument, path to RGB images as second argument, 
    x pixel sampling location as third argument, y pixel sampling location as fourth argument, 
    optionally takes in path to put masked images as fifth argument.
    If the fifth argument is not given, creates a masks directory in the parent directory of the RGB images directory
    '''
    checkpoint = argv[0]
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    rgb_dir = argv[1]
    if rgb_dir[-1] == '/':
        rgb_dir = rgb_dir[:-1]
    x = int(argv[2])
    y = int(argv[3])
    points = np.array([[x, y]], dtype=np.float32)
    labels = np.array([1], np.int32)
    
    # determine path to save masks
    if len(argv) < 5:
        rgb_path = Path(rgb_dir)
        parent_path =  rgb_path.parent.absolute()
        mask_dir_path = os.path.join(parent_path, "masks")
    else:
        mask_dir_path = argv[4]

    if not os.path.exists(mask_dir_path):
        os.makedirs(mask_dir_path)

    video_dir = rgb_dir + "_jpg"

    # convert to jpg
    convert_png_to_jpg(rgb_dir, video_dir)

    frame_names = [
        os.path.splitext(p)[0] for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(p))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_dir)

        # add new prompts and instantly get the output on the same frame
        frame_idx, object_ids, masks = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=points,
            labels=labels
        )
        save_mask(mask_dir_path, frame_names, frame_idx, masks)

        # propagate the prompts to get masklets throughout the video
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            save_mask(mask_dir_path, frame_names, frame_idx, masks)
    
    # remove jpgs
    shutil.rmtree(video_dir)

if __name__ == "__main__":
   main(sys.argv[1:])