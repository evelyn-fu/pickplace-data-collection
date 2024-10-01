import numpy as np
import argparse
import json
import time
from planning.misc.sdf_tools import SignedDensityField

# SDF for boxes
def sdf_box_batch(points, centers, half_extents):
    d = np.abs(points[:, np.newaxis, :] - centers) - half_extents
    
    outside_dist = np.maximum(d, 0)  # Euclidean distance for points outside
    outside = np.linalg.norm(outside_dist, axis=-1)

    inside_dist = np.min(d, axis=-1)  # Negative distance inside
    
    # If inside, return the inside distance (negative); if outside, return the outside distance
    sdf_values = np.where(np.all(d <= 0, axis=-1), inside_dist, outside)
    
    return sdf_values

# Smooth union operation
def smooth_union(d1, d2, k=16.0):
    # Perform smooth union
    smooth_d = -np.log(np.exp(-k * d1) + np.exp(-k * d2)) / k
    
    # Ensure distances outside the shape remain non-negative
    return np.maximum(smooth_d, np.minimum(d1, d2))

# Vectorized combination using smooth union for multiple boxes
def combined_sdf_batch_smooth(points, boxes, k=16.0):
    centers = np.array([center for center, _ in boxes])
    half_extents = np.array([he for _, he in boxes])

    # Compute SDF for all boxes and points at once
    sdf_values = sdf_box_batch(points, centers, half_extents)
    
    # Initialize with the first SDF, then apply smooth union for the rest
    combined_sdf = sdf_values[:, 0]
    for i in range(1, sdf_values.shape[1]):
        combined_sdf = smooth_union(combined_sdf, sdf_values[:, i], k=k)
    
    return combined_sdf


# Generate the SDF for the object
def generate_sdf(delta, origin, boxes, k=16.0):
    '''
    Parameters:
        delta: grid spacing between sdf values
        origin: length 3 array of (x,y,z) point of origin
        boxes: list of tuples (center, half_extents)
    Returns: SignedDensityField
    '''
    if len(origin) != 3:
        raise Exception("Origin should be length 3")

    bounds = np.zeros((3,2))
    for box in boxes:
        box_min = box[0]-box[1]
        box_max = box[0]+box[1]

        bounds[:,0] = np.minimum(bounds[:,0], box_min)
        bounds[:,1] = np.maximum(bounds[:,1], box_max)

    grid_size_x = int((bounds[0][1] - bounds[0][0]) / delta)
    grid_size_y = int((bounds[1][1] - bounds[1][0]) / delta)
    grid_size_z = int((bounds[2][1] - bounds[2][0]) / delta)

    print(grid_size_z, grid_size_y, grid_size_z)

    # Create a 3D grid within the specified bounds
    x = np.linspace(bounds[0][0], bounds[0][1], grid_size_x)
    y = np.linspace(bounds[1][0], bounds[1][1], grid_size_y)
    z = np.linspace(bounds[2][0], bounds[2][1], grid_size_z)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    sdf_grid = combined_sdf_batch_smooth(grid_points, boxes, k=k)

    # Reshape back to 3D grid
    sdf_grid = sdf_grid.reshape((grid_size_x, grid_size_y, grid_size_z))
    
    sdf = SignedDensityField(sdf_grid, bounds[:,0] - np.array(origin), delta)

    return sdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "gripper_description",
        default="gripper_description.json",
        help="json file with gripper dimensions",
    )
    parser.add_argument(
        "pkl_file",
        default="gripper_sdf.pkl",
        help="pkl file path to save sdf to",
    )
    args = parser.parse_args()

    start = time.time()
    description_path = args.gripper_description
    with open(description_path) as f:
        description = json.load(f)

        fw = description["finger_width"]
        fh = description["finger_height"]
        fl = description["finger_length"]
        fs = description["finger_spacing"]
        bw = description["base_width"]
        bh = description["base_height"]
        bd = description["base_depth"]
        delta = description["delta"] # spacing between grid points

        # Construct gripper components
        boxes = [
            (np.array([0, 0, 0]), np.array([bw/2, bd/2, bh/2])),
            (np.array([-fs/2, bd/2 + fl/2, 0]), np.array([fw/2, fl/2, fh/2])),
            (np.array([fs/2, bd/2 + fl/2, 0]), np.array([fw/2, fl/2, fh/2]))
        ]

        if "buffer" in description:
            buf = description["buffer"]
            if buf != 0:
                # add box for gripper buffer
                boxes.append(
                    (np.array([0, (bd + buf)/2, 0]), np.array([bw/2, buf/2, bh/2]))
                )
        
        # add extra boxes
        if "extras" in description:
            for extra in description["extras"]:
                box_center = extra["box"][:3]
                box_dims = extra["box"][3:]
                boxes.append(
                    (np.array(box_center), np.array(box_dims))
                )

        # Generate the SDF
        sdf = generate_sdf(delta, description["origin"], boxes)

        sdf.dump(args.pkl_file)
    print("SDF computation time", time.time()-start)

    sdf = SignedDensityField.from_pkl(args.pkl_file)
    # Query the SDF at an arbitrary point
    point = np.array([0.0, -0.049133, 0.025])
    start = time.time()
    distance_at_point = sdf.get_distance(point)
    print("SDF lookup time", time.time()-start)
    print(f"Signed distance at point {point}: {distance_at_point}")

    sdf.visualize()