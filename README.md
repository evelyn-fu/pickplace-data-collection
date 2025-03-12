# pickplace_data_collection
Automatic pick + place for visual and inertial data collection

## Usage
Run the pick and place pipeline using `scripts/grasping_test.py`.  Use the `--num_objects` flag to specify how many loops to run the pick and place for. Use the `--use_hardware` flag to run on hardware and omit to run in simulation. Use the `--save_imgs` flag to automatically save visual data and omit to skip saving images. Use the `--save_dir` flag to provide a name for a directory that will be created under `\tests\` to which the data will be saved.

### Example Usage
```
python scripts/grasping_test.py --num_objects=5 --save_dir=5_objects --use_hardware --save_imgs
```
