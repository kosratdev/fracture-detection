import cv2

from preprocessors import Filter
from bone_fracture import FractureDetector, Ml, DatasetType
from print import print_table


def main():
    filters = [[Filter.gaussian, Filter.adaptive_hist, Filter.canny]]
    data = []
    for _filter in filters:
        detector = FractureDetector(_filter, Ml.svm,
                                    dataset=DatasetType.perfect)
        data.append(detector.accuracy())
        detector = FractureDetector(_filter, Ml.decision_tree,
                                    dataset=DatasetType.perfect)
        data.append(detector.accuracy())
        detector = FractureDetector(_filter, Ml.naive_bayes,
                                    dataset=DatasetType.perfect)
        data.append(detector.accuracy())
        detector = FractureDetector(_filter, Ml.random_forest,
                                    dataset=DatasetType.perfect)
        data.append(detector.accuracy())
        detector = FractureDetector(_filter, Ml.nearest_neighbors,
                                    dataset=DatasetType.perfect)
        data.append(detector.accuracy())

    print("ML Comparison Table")
    print_table(data)


if __name__ == "__main__":
    main()
