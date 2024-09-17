#!/bin/python3

"""Script that adds an alpha channel based on binary masks."""

import argparse
import os
import fnmatch
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="The path to the folder containing the images.",
    )
    parser.add_argument(
        "--masks",
        type=str,
        required=True,
        help="The path to the folder containing the binary masks.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="The path to the folder where the images with alpha channel are written to.",
    )
    args = parser.parse_args()
    image_dir_path = Path(args.images)
    mask_dir_path = Path(args.masks)
    out_dir_path = Path(args.out)

    num_images = len(fnmatch.filter(os.listdir(image_dir_path), "*.png"))
    assert num_images > 0, f"No images found in {image_dir_path}"

    out_dir_path.mkdir(exist_ok=True)

    for image_path in sorted(os.listdir(image_dir_path)):
        img_path = os.path.join(image_dir_path, image_path)
        img = np.asarray(Image.open(img_path).convert("RGB"))

        mask_path = os.path.join(mask_dir_path, image_path)
        mask = np.asarray(Image.open(mask_path))

        img_w_alpha = np.concatenate((img, mask[:, :, np.newaxis]), axis=-1)

        out_path = out_dir_path / image_path
        img_w_alpha_pil = Image.fromarray(img_w_alpha)
        img_w_alpha_pil.save(out_path)


if __name__ == "__main__":
    main()