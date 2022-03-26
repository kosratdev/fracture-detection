import glob
import os

import cv2


class ImageReader:

    def __init__(self, train_path="../images/original_images"):
        self.__train_path = train_path

    def read(self):
        tr_images, tr_labels = self._read_images_labels(self.__train_path)
        return tr_images, tr_labels

    @staticmethod
    def _read_images_labels(path):
        images = []
        labels = []
        # for directory_path in glob.glob("cell_images/original_train/*"):
        for directory_path in glob.glob(path + "/*"):
            label = directory_path.split("\\")[-1]
            for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
                # Reading color images
                img = cv2.imread(img_path, 0)
                # Resize images
                img = cv2.resize(img, (256, 512))
                images.append(img)
                labels.append(label)

        return images, labels
