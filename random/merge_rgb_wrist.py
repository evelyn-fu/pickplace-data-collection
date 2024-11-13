import os
import shutil
import sys

def main(argv):
    top_dir = argv[0]

    # Define paths
    rgb_dir = os.path.join(top_dir, 'rgb')
    rgb_wrist_dir = os.path.join(top_dir, 'rgb_wrist')
    rgb_merged_dir = os.path.join(top_dir, 'rgb_merged')
    masks_dir = os.path.join(top_dir, 'masks')
    masks_wrist_dir = os.path.join(top_dir, 'masks_wrist')
    masks_merged_dir = os.path.join(top_dir, 'masks_merged')
    depth_dir = os.path.join(top_dir, 'depth')
    depth_wrist_dir = os.path.join(top_dir, 'depth_wrist')
    depth_merged_dir = os.path.join(top_dir, 'depth_merged')
    
    # Create the merged directory if it doesn't exist
    os.makedirs(rgb_merged_dir, exist_ok=True)
    os.makedirs(masks_merged_dir, exist_ok=True)
    os.makedirs(depth_merged_dir, exist_ok=True)
    
    # Copy and rename files from rgb directory with '0' prefix
    if os.path.exists(rgb_dir):
        for filename in os.listdir(rgb_dir):
            src_path = os.path.join(rgb_dir, filename)
            if os.path.isfile(src_path):
                dst_filename = f"0{filename}"
                dst_path = os.path.join(rgb_merged_dir, dst_filename)
                shutil.copy(src_path, dst_path)
    
    # Copy and rename files from rgb_wrist directory with '1' prefix
    if os.path.exists(rgb_wrist_dir):
        for filename in os.listdir(rgb_wrist_dir):
            src_path = os.path.join(rgb_wrist_dir, filename)
            if os.path.isfile(src_path):
                dst_filename = f"1{filename}"
                dst_path = os.path.join(rgb_merged_dir, dst_filename)
                shutil.copy(src_path, dst_path)
    
    # Copy and rename files from masks directory with '0' prefix
    if os.path.exists(masks_dir):
        for filename in os.listdir(masks_dir):
            src_path = os.path.join(masks_dir, filename)
            if os.path.isfile(src_path):
                dst_filename = f"0{filename}"
                dst_path = os.path.join(masks_merged_dir, dst_filename)
                shutil.copy(src_path, dst_path)
    
    # Copy and rename files from masks_wrist directory with '1' prefix
    if os.path.exists(masks_wrist_dir):
        for filename in os.listdir(masks_wrist_dir):
            src_path = os.path.join(masks_wrist_dir, filename)
            if os.path.isfile(src_path):
                dst_filename = f"1{filename}"
                dst_path = os.path.join(masks_merged_dir, dst_filename)
                shutil.copy(src_path, dst_path)
    
    # Copy and rename files from depth directory with '0' prefix
    if os.path.exists(depth_dir):
        for filename in os.listdir(depth_dir):
            src_path = os.path.join(depth_dir, filename)
            if os.path.isfile(src_path):
                dst_filename = f"0{filename}"
                dst_path = os.path.join(depth_merged_dir, dst_filename)
                shutil.copy(src_path, dst_path)
    
    # Copy and rename files from depth_wrist directory with '1' prefix
    if os.path.exists(depth_wrist_dir):
        for filename in os.listdir(depth_wrist_dir):
            src_path = os.path.join(depth_wrist_dir, filename)
            if os.path.isfile(src_path):
                dst_filename = f"1{filename}"
                dst_path = os.path.join(depth_merged_dir, dst_filename)
                shutil.copy(src_path, dst_path)

    print(f"Files merged and renamed in '{rgb_merged_dir}', '{masks_merged_dir}', '{depth_merged_dir}'")

if __name__ == "__main__":
   main(sys.argv[1:]) # arguments: [data directory]