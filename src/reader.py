import glob
import os

import cv2


class ImageReader:

    def __init__(self, train_path="../images/train", test_path="../images/test",
                 image_size=128):
        self.__train_path = train_path
        self.__test_path = test_path
        self.__image_size = image_size
        # self._train_images = []
        # self._train_labels = []
        # self._test_images = []
        # self._test_labels = []

    def read(self):

        tr_images, tr_labels = self.__read_images_labels(self.__train_path)
        te_images, te_labels = self.__read_images_labels(self.__test_path)
        return tr_images, tr_labels, te_images, te_labels

    def __read_images_labels(self, path):

        images = []
        labels = []
        # for directory_path in glob.glob("cell_images/train/*"):
        for directory_path in glob.glob(path + "/*"):
            label = directory_path.split("\\")[-1]
            for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
                # Reading color images
                img = cv2.imread(img_path, 0)
                # Resize images
                img = cv2.resize(img, (self.__image_size, self.__image_size))
                images.append(img)
                labels.append(label)

        # self._train_images = np.array(self._train_images)
        # self._train_labels = np.array(self._train_labels)
        return images, labels
