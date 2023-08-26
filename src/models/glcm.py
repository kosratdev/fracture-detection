import numpy as np
from skimage.feature import graycomatrix, graycoprops


class GLCM:
    def __init__(self, image):
        distance = [1, 3, 5, 8]
        angles = [
            0,  # 0
            np.pi / 4,  # 45
            np.pi / 2,  # 90
            3 * np.pi / 4,  # 135
            np.pi,  # 180
            # 5 * np.pi / 4,    # 225
            # 3 * np.pi / 2,    # 270
            # 7 * np.pi / 4     # 315
        ]
        self.glcm_mat = graycomatrix(image, distances=distance, angles=angles)
        self.properties = ['energy', 'correlation', 'dissimilarity',
                           'homogeneity', 'contrast']

    def correlation(self):
        return graycoprops(self.glcm_mat, 'correlation').flatten()

    def homogeneity(self):
        return graycoprops(self.glcm_mat, 'homogeneity').flatten()

    def contrast(self):
        return graycoprops(self.glcm_mat, 'contrast').flatten()

    def energy(self):
        return graycoprops(self.glcm_mat, 'energy').flatten()

    def dissimilarity(self):
        return graycoprops(self.glcm_mat, 'dissimilarity').flatten()

    def asm(self):
        return graycoprops(self.glcm_mat, 'ASM').flatten()

    def glcm_all(self):
        return np.hstack([graycoprops(self.glcm_mat, props).ravel() for props in
                          self.properties])
