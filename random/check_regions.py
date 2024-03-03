import pickle
import argparse
from pydrake.geometry.optimization import HPolyhedron

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pickle_path",
        help="pkl file with regions",
    )
    args = parser.parse_args()

    initial = [-6.360e-09 , 4.000e-01,  4.171e-08, -1.200e+00 , 1.249e-07 , 1.000e+00, 1.570e+00]
    final = [-0.558 , 0.672,  0.872, -1.054 ,-0.352 , 1.575 ,-1.186]

    filepath = args.pickle_path
    with open(filepath, 'rb') as f:
        regions = pickle.load(f)
        print(regions)

        initial_contained = False
        final_contained = False

        for r in regions:
            if r.PointInSet(initial):
                initial_contained = True
            if r.PointInSet(final):
                final_contained = True

        if not initial_contained:
            print("initial point not in regions")
        if not final_contained:
            print("final point not in regions")
        
        if initial_contained and final_contained:
            print("they both there")