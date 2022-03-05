from enum import Enum, auto, unique

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from feature import FeatureExtractor
from preprocessors import Preprocessors, Filter
from reader import ImageReader


@unique
class ML(Enum):
    svm = auto()
    decision_tree = auto()
    naive_bayes = auto()
    random_forest = auto()
    nearest_neighbors = auto()

    def __str__(self):
        return ' '.join(word.title() for word in self.name.split('_'))


class FractureDetector:

    def __init__(self, filters: [Filter], ml: ML):
        self._filters = filters
        self._ml = ml
        # Read image from train and test folders to prepare the dataset.
        reader = ImageReader()
        o_train_images, o_train_labels, o_test_images, o_test_labels = reader.read()

        # Apple preprocessors on the images.
        p_train_images, p_test_images = Preprocessors(o_train_images,
                                                      o_test_images).process(
            filters)

        features = FeatureExtractor(p_train_images, o_train_labels,
                                    p_test_images, o_test_labels)
        self._train_images, self._train_labels, self._test_images, self._test_labels = features.glcm_feature_extraction()
        self._clf = self._get_clf()

    def _get_clf(self):
        if self._ml == ML.svm:
            clf = SVC(kernel="rbf", C=10000)
            return clf.fit(self._train_images, self._train_labels)
        elif self._ml == ML.decision_tree:
            clf = DecisionTreeClassifier(min_samples_split=40)
            return clf.fit(self._train_images, self._train_labels)
        elif self._ml == ML.naive_bayes:
            clf = GaussianNB()
            return clf.fit(self._train_images, self._train_labels)
        elif self._ml == ML.random_forest:
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            return clf.fit(self._train_images, self._train_labels)
        elif self._ml == ML.nearest_neighbors:
            clf = KNeighborsClassifier(n_neighbors=15)
            return clf.fit(self._train_images, self._train_labels)

    def accuracy(self):
        data = []
        data.extend(self._filters)
        data.append(self._ml)
        data.append(self._clf.score(self._test_images, self._test_labels))
        return data
