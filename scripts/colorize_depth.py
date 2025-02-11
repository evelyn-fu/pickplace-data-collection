import os
import cv2
import numpy as np
import argparse
from tqdm.contrib.concurrent import process_map
from functools import partial


def load_and_colorize_depth(image_path, output_folder, colormap=cv2.COLORMAP_JET):
    """Loads a depth image, applies colormap, and saves it."""
    try:
        # Load depth image (assuming it's stored as 16-bit PNG or similar)
        depth = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"Failed to load {image_path}")
            return

        # Enhance closer range differences
        depth = np.sqrt(
            depth
        )  # Applying a non-linear transformation to emphasize closer depth differences

        # Normalize depth to 8-bit range
        depth_normalized = cv2.normalize(
            depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        # Apply colormap
        colorized = cv2.applyColorMap(depth_normalized, colormap)

        # Save colorized image
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, colorized)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def process_depth_images(input_folder, output_folder, num_workers=os.cpu_count()):
    """Processes all depth images in the input folder using multiprocessing."""
    os.makedirs(output_folder, exist_ok=True)

    # Get list of depth images
    image_paths = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith((".png", ".tiff", ".exr"))
    ]

    # Process in parallel
    process_map(
        partial(load_and_colorize_depth, output_folder=output_folder),
        image_paths,
        max_workers=num_workers,
        chunksize=10,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Colorize depth images using a colormap and save them."
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the input folder containing depth images.",
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Path to the output folder where colorized images will be saved.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers (default: all cores).",
    )

    args = parser.parse_args()
    process_depth_images(args.input_folder, args.output_folder, args.num_workers)
