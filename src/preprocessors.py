import cv2


class Preprocessors:

    def __init__(self, train_images, test_images):
        self.__train_images = train_images
        self.__test_images = test_images

    def median(self):
        for index, img in enumerate(self.__train_images):
            self.__train_images[index] = cv2.medianBlur(img, 3)

        for index, img in enumerate(self.__test_images):
            self.__test_images[index] = cv2.medianBlur(img, 3)

        return self

    def gaussian(self):
        for index, img in enumerate(self.__train_images):
            self.__train_images[index] = cv2.GaussianBlur(img, (3, 3), 0)

        for index, img in enumerate(self.__test_images):
            self.__test_images[index] = cv2.GaussianBlur(img, (3, 3), 0)

        return self

    def equalize_hist(self):
        for index, img in enumerate(self.__train_images):
            self.__train_images[index] = cv2.equalizeHist(img)

        for index, img in enumerate(self.__test_images):
            self.__test_images[index] = cv2.equalizeHist(img)

        return self

    def adaptive_hist(self):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        for index, img in enumerate(self.__train_images):
            self.__train_images[index] = clahe.apply(img)

        for index, img in enumerate(self.__test_images):
            self.__test_images[index] = clahe.apply(img)

        return self

    def adjust_contrast(self, brightness=-20, contrast=40):
        for index, img in enumerate(self.__train_images):
            self.__train_images[index] = self.__apply_contrast(img, brightness,
                                                               contrast)

        for index, img in enumerate(self.__test_images):
            self.__test_images[index] = self.__apply_contrast(img, brightness,
                                                              contrast)

        return self

    def __apply_contrast(self, img, brightness, contrast):
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

    def canny(self):
        for index, img in enumerate(self.__train_images):
            self.__train_images[index] = cv2.Canny(img, 100, 150)

        for index, img in enumerate(self.__test_images):
            self.__test_images[index] = cv2.Canny(img, 100, 150)

        return self

    def sobel(self):
        for index, img in enumerate(self.__train_images):
            self.__train_images[index] = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1,
                                                   dy=1, ksize=5)

        for index, img in enumerate(self.__test_images):
            self.__test_images[index] = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1,
                                                  dy=1, ksize=5)

        return self

    def process(self):
        return self.__train_images, self.__test_images
