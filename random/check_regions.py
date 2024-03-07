import pickle
import argparse
from pydrake.geometry.optimization import HPolyhedron, ConvexSet
from pydrake.common import RandomGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pickle_path",
        help="pkl file with regions",
    )
    args = parser.parse_args()

    initial = [-6.360e-09,  4.000e-01,  4.171e-08, -1.200e+00 , 1.249e-07 , 1.000e+00,  1.570e+00]
    final = [-0.633 , 0.059 , 0.6 ,  -1.745 , 0.031 , 1.074, -1.336]

    filepath = args.pickle_path
    with open(filepath, 'rb') as f:
        regions = pickle.load(f)
        print(len(regions), "regions")

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

        # generator = RandomGenerator()
        # pairs_with_high_overlap = 0
        # max_overlap = 0
        # for i in range(len(regions)-1):
        #     for j in range(i, len(regions)):

        #         if not regions[i].IntersectsWith(regions[j]):
        #             continue

        #         # Do the overlap check
        #         intersection = regions[i].Intersection(regions[j])

        #         intersection_volume = intersection.CalcVolumeViaSampling(generator, 0.05, 10000).volume
        #         region_i_volume = regions[i].CalcVolumeViaSampling(generator, 0.05, 10000).volume
        #         region_j_volume = regions[j].CalcVolumeViaSampling(generator, 0.05, 10000).volume

        #         overlap = intersection_volume / (region_i_volume + region_j_volume)
        #         if overlap > 0.4:
        #             print("high overlap:", overlap)
        #             pairs_with_high_overlap += 1
        #         if overlap > max_overlap:
        #             print("new max overlap:", overlap)
        #             max_overlap = overlap
        
        # print("pairs with >0.4 overlap:", pairs_with_high_overlap)
        # print("max overlap:", max_overlap)