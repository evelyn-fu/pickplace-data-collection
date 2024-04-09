import os
import sys
import shutil
from pathlib import Path

def main(argv):
    new_dir = argv[0] + "_halved"
    original_path = Path(argv[0])
    print(new_dir)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        
    count = 0
    for image_path in sorted(os.listdir(original_path)):
        input_path = os.path.join(original_path, image_path)
        if count % 2 == 0:
            shutil.copy2(input_path, new_dir)
        count += 1

if __name__ == "__main__":
   main(sys.argv[1:])