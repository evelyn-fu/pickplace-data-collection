import numpy as np
import os
import sys
from PIL import Image 
from pathlib import Path

def main(argv):
    data_path = Path(argv[0])
    masks_path = os.path.join(data_path, "masks/")
    rgbs_path = os.path.join(data_path, "images/")
    outs_path = os.path.join(data_path, "masked_rgb/")
    if not os.path.exists(outs_path):
        os.makedirs(outs_path)
    
    for image_path in sorted(os.listdir(rgbs_path)):
        mask_name = "mask" + image_path[-8:-3] + "npy"
        mask_path = os.path.join(masks_path, mask_name)
        rgb_path = os.path.join(rgbs_path, image_path)
        out_path = os.path.join(outs_path, image_path)
        label_image = np.load(mask_path)
        rgb_image = np.array(Image.open(rgb_path))

        mask = np.zeros_like(rgb_image)
        for i in range(3): 
            mask[:,:,i] = label_image > 0
        masked_image = np.concatenate((rgb_image, label_image[:, :, np.newaxis]), axis=-1)
        # masked_image = np.uint8(np.where(mask, rgb_image, np.zeros(rgb_image.shape)))
        mask_pil = Image.fromarray(masked_image)
        mask_pil.save(out_path)

if __name__ == "__main__":
   main(sys.argv[1:]) # arguments: [data directory]