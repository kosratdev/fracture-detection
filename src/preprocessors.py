from enum import Enum, unique, auto

import cv2


@unique
class Filter(Enum):
    median = auto()
    gaussian = auto()
    equalize_hist = auto()
    adaptive_hist = auto()
    adjust_contrast = auto()
    canny = auto()
    sobel = auto()

    def __str__(self):
        return ' '.join(word.title() for word in self.name.split('_'))


class Preprocessors:

    def __init__(self, train_images, test_images):
        self.__train_images = train_images
        self.__test_images = test_images

    def _median(self):
        for index, img in enumerate(self.__train_images):
            self.__train_images[index] = cv2.medianBlur(img, 3)

        for index, img in enumerate(self.__test_images):
            self.__test_images[index] = cv2.medianBlur(img, 3)

        return self

    def _gaussian(self):
        for index, img in enumerate(self.__train_images):
            self.__train_images[index] = cv2.GaussianBlur(img, (3, 3), 0)

        for index, img in enumerate(self.__test_images):
            self.__test_images[index] = cv2.GaussianBlur(img, (3, 3), 0)

        return self

    def _equalize_hist(self):
        for index, img in enumerate(self.__train_images):
            self.__train_images[index] = cv2.equalizeHist(img)

        for index, img in enumerate(self.__test_images):
            self.__test_images[index] = cv2.equalizeHist(img)

        return self

    def _adaptive_hist(self):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        for index, img in enumerate(self.__train_images):
            self.__train_images[index] = clahe.apply(img)

        for index, img in enumerate(self.__test_images):
            self.__test_images[index] = clahe.apply(img)

        return self

    def _adjust_contrast(self, brightness=-20, contrast=40):
        for index, img in enumerate(self.__train_images):
            self.__train_images[index] = self._apply_contrast(img, brightness,
                                                              contrast)

        for index, img in enumerate(self.__test_images):
            self.__test_images[index] = self._apply_contrast(img, brightness,
                                                             contrast)

        return self

    def _apply_contrast(self, img, brightness, contrast):
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            buf = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
        else:
            buf = img.copy()

        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    def _canny(self):
        for index, img in enumerate(self.__train_images):
            self.__train_images[index] = cv2.Canny(img, 100, 150)

        for index, img in enumerate(self.__test_images):
            self.__test_images[index] = cv2.Canny(img, 100, 150)

        return self

    def _sobel(self):
        for index, img in enumerate(self.__train_images):
            self.__train_images[index] = cv2.Sobel(img, ddepth=cv2.CV_8U, dx=1,
                                                   dy=0, ksize=3)

        for index, img in enumerate(self.__test_images):
            self.__test_images[index] = cv2.Sobel(img, ddepth=cv2.CV_8U, dx=1,
                                                  dy=1, ksize=5)

        return self

    def process(self, filters: [Filter]):
        for filter in filters:
            if filter == Filter.median:
                self._median()
            elif filter == Filter.gaussian:
                self._gaussian()
            elif filter == Filter.equalize_hist:
                self._equalize_hist()
            elif filter == Filter.adaptive_hist:
                self._adaptive_hist()
            elif filter == Filter.adjust_contrast:
                self._adjust_contrast()
            elif filter == Filter.canny:
                self._canny()
            elif filter == Filter.sobel:
                self._sobel()

        return self.__train_images, self.__test_images
