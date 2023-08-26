import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn import preprocessing

from src.models.glcm import GLCM


class FeatureExtractor:

    def __init__(self, *args):
        if len(args) == 2:
            self.train_images = np.array(args[0])  # np.array(train_images)
            self.train_labels = np.array(args[1])  # np.array(train_labels)
            self._encode_labels()
        else:
            self._image = np.array(np.array(args[0]))

    def _encode_labels(self):
        # Encode labels from text (folder names) to integers.
        le = preprocessing.LabelEncoder()

        le.fit(self.train_labels)
        train_labels_encoded = le.transform(self.train_labels)
        self.train_labels = train_labels_encoded

    def glcm_feature_extraction(self):
        image_features = []
        for image in self.train_images:
            image_features.append(GLCM(image).glcm_all())
        return image_features, self.train_labels

    def single_glcm_feature_extraction(self):
        return GLCM(self._image).glcm_all()
