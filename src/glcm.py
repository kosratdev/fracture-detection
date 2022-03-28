import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops


class GLCM:
    def __init__(self, image):
        distance = [1, 2, 3]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        # self.image = img_as_ubyte(image.astype('int64'))
        self.glcm_mat = graycomatrix(image, distances=distance,
                                     angles=angles, symmetric=True, normed=True)
        self.properties = ['correlation', 'homogeneity', 'contrast', 'energy']

    def correlation(self):
        return graycoprops(self.glcm_mat, 'correlation').flatten()

    def homogeneity(self):
        return graycoprops(self.glcm_mat, 'homogeneity').flatten()

    def contrast(self):
        return graycoprops(self.glcm_mat, 'contrast').flatten()

    def energy(self):
        return graycoprops(self.glcm_mat, 'energy').flatten()

    def glcm_all(self):
        return np.hstack([graycoprops(self.glcm_mat, props).ravel() for props in
                          self.properties])


# img = cv2.imread("images/sample1.jpg", 0)
# print(img)
# feats = GLCM(np.array(img))
# _all = feats.glcm_all()
# print(len(_all))
