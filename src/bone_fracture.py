from enum import Enum, auto, unique

import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from feature import FeatureExtractor
from preprocessors import Preprocessors, Filter
from reader import ImageReader


@unique
class DatasetType(Enum):
    original = auto()
    edited = auto()
    augmented = auto()


@unique
class Ml(Enum):
    svm = auto()
    decision_tree = auto()
    naive_bayes = auto()
    random_forest = auto()
    nearest_neighbors = auto()

    def __str__(self):
        return ' '.join(word.title() for word in self.name.split('_'))


class FractureDetector:

    def __init__(self, filters: [Filter], ml: Ml, dataset=DatasetType.original):
        self._filters = filters
        self._ml = ml

        # Read image from the provided images to prepare the dataset.
        if dataset == DatasetType.augmented:
            reader = ImageReader(train_path="../images/augmented_train",
                                 test_path="../images/augmented_test")
        elif dataset == DatasetType.edited:
            reader = ImageReader(train_path="../images/edited_train",
                                 test_path="../images/edited_test")
        else:
            reader = ImageReader()

        o_train_images, o_train_labels, o_test_images, o_test_labels = reader.read()

        # Apple preprocessors on the images.
        p_train_images = Preprocessors(o_train_images).process(filters)
        p_test_images = Preprocessors(o_test_images).process(filters)

        features = FeatureExtractor(p_train_images, o_train_labels,
                                    p_test_images, o_test_labels)
        self._train_images, self._train_labels, self._test_images, self._test_labels = features.glcm_feature_extraction()
        self._clf = self._get_clf()

    def _get_clf(self):
        if self._ml == Ml.svm:
            clf = SVC(kernel="rbf", C=10000)
            return clf.fit(self._train_images, self._train_labels)
        elif self._ml == Ml.decision_tree:
            clf = DecisionTreeClassifier(min_samples_split=40)
            return clf.fit(self._train_images, self._train_labels)
        elif self._ml == Ml.naive_bayes:
            clf = GaussianNB()
            return clf.fit(self._train_images, self._train_labels)
        elif self._ml == Ml.random_forest:
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            return clf.fit(self._train_images, self._train_labels)
        elif self._ml == Ml.nearest_neighbors:
            clf = KNeighborsClassifier(n_neighbors=15)
            return clf.fit(self._train_images, self._train_labels)

    def predict(self, img_path: str, img_size=128):
        # Reading color images
        read_img = cv2.imread(img_path, 0)
        # Resize images
        resize_img = cv2.resize(read_img, (img_size, img_size))
        processed_img = Preprocessors([resize_img]).process(self._filters)
        features = FeatureExtractor(processed_img)
        img_features = features.single_glcm_feature_extraction()
        predict = self._clf.predict(img_features)
        predict_label = ['Fractured', 'Non-Fractured']

        data = []
        data.extend(self._filters)
        data.append(self._ml)
        data.append(predict_label[predict[0]])
        return data

    def accuracy(self):
        data = []
        data.extend(self._filters)
        data.append(self._ml)
        data.append(self._clf.score(self._test_images, self._test_labels))
        return data
