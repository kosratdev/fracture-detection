import glob
import os

import cv2
import numpy as np


class ImageReader:

    def __init__(self, train_path="../images/ml_eeeh_dataset", extension="jpg"):
        self.__train_path = train_path
        self.__extension = extension

    def read(self):
        tr_images, tr_labels = self._read_images_labels(self.__train_path,
                                                        self.__extension)
        return tr_images, tr_labels

    @staticmethod
    def _read_images_labels(path, extension):
        images = []
        labels = []
        # for directory_path in glob.glob("cell_images/original_train/*"):
        for directory_path in glob.glob(path + "/*"):
            label = directory_path.split("\\")[-1]
            for img_path in glob.glob(
                    os.path.join(directory_path, "*." + extension)
            ):
                # Reading color images
                img = cv2.imread(img_path, 0)
                # Resize images
                img = cv2.resize(img, (256, 512))
                images.append(np.array(img))
                labels.append(label)

        return images, labels
