import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn import preprocessing
import cv2

from src.glcm import GLCM


class FeatureExtractor:

    def __init__(self, *args):
        if len(args) == 2:
            self.train_images = np.array(args[0])  # np.array(train_images)
            self.train_labels = np.array(args[1])  # np.array(train_labels)
            self._encode_labels()
        else:
            self._image = np.array(args[0])

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
        # return _glcm(self.train_images), self.train_labels

    def single_glcm_feature_extraction(self):
        return _glcm(self._image)


def _glcm(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):  # iterate through each file
        # Temporary data frame to capture information for each loop.
        df = pd.DataFrame()
        # Reset dataframe to blank after each loop.

        img = dataset[image, :, :]
        ################################################################
        # START ADDING DATA TO THE DATAFRAME

        GLCM = graycomatrix(img, [1], [0])
        GLCM_Energy = graycoprops(GLCM, 'energy')[0]
        df['Energy'] = GLCM_Energy
        GLCM_corr = graycoprops(GLCM, 'correlation')[0]
        df['Corr'] = GLCM_corr
        GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
        df['Diss_sim'] = GLCM_diss
        GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
        df['Homogen'] = GLCM_hom
        GLCM_contr = graycoprops(GLCM, 'contrast')[0]
        df['Contrast'] = GLCM_contr
        GLCM_asm = graycoprops(GLCM, 'ASM')[0]
        df['ASM'] = GLCM_asm

        GLCM2 = graycomatrix(img, [3], [0])
        GLCM_Energy2 = graycoprops(GLCM2, 'energy')[0]
        df['Energy2'] = GLCM_Energy2
        GLCM_corr2 = graycoprops(GLCM2, 'correlation')[0]
        df['Corr2'] = GLCM_corr2
        GLCM_diss2 = graycoprops(GLCM2, 'dissimilarity')[0]
        df['Diss_sim2'] = GLCM_diss2
        GLCM_hom2 = graycoprops(GLCM2, 'homogeneity')[0]
        df['Homogen2'] = GLCM_hom2
        GLCM_contr2 = graycoprops(GLCM2, 'contrast')[0]
        df['Contrast2'] = GLCM_contr2
        GLCM_asm2 = graycoprops(GLCM2, 'ASM')[0]
        df['ASM2'] = GLCM_asm2

        GLCM3 = graycomatrix(img, [5], [0])
        GLCM_Energy3 = graycoprops(GLCM3, 'energy')[0]
        df['Energy3'] = GLCM_Energy3
        GLCM_corr3 = graycoprops(GLCM3, 'correlation')[0]
        df['Corr3'] = GLCM_corr3
        GLCM_diss3 = graycoprops(GLCM3, 'dissimilarity')[0]
        df['Diss_sim3'] = GLCM_diss3
        GLCM_hom3 = graycoprops(GLCM3, 'homogeneity')[0]
        df['Homogen3'] = GLCM_hom3
        GLCM_contr3 = graycoprops(GLCM3, 'contrast')[0]
        df['Contrast3'] = GLCM_contr3
        GLCM_asm3 = graycoprops(GLCM3, 'ASM')[0]
        df['ASM3'] = GLCM_asm3

        GLCM4 = graycomatrix(img, [0], [np.pi / 4])
        GLCM_Energy4 = graycoprops(GLCM4, 'energy')[0]
        df['Energy4'] = GLCM_Energy4
        GLCM_corr4 = graycoprops(GLCM4, 'correlation')[0]
        df['Corr4'] = GLCM_corr4
        GLCM_diss4 = graycoprops(GLCM4, 'dissimilarity')[0]
        df['Diss_sim4'] = GLCM_diss4
        GLCM_hom4 = graycoprops(GLCM4, 'homogeneity')[0]
        df['Homogen4'] = GLCM_hom4
        GLCM_contr4 = graycoprops(GLCM4, 'contrast')[0]
        df['Contrast4'] = GLCM_contr4
        GLCM_asm4 = graycoprops(GLCM4, 'ASM')[0]
        df['ASM4'] = GLCM_asm4

        GLCM5 = graycomatrix(img, [0], [np.pi / 2])
        GLCM_Energy5 = graycoprops(GLCM5, 'energy')[0]
        df['Energy5'] = GLCM_Energy5
        GLCM_corr5 = graycoprops(GLCM5, 'correlation')[0]
        df['Corr5'] = GLCM_corr5
        GLCM_diss5 = graycoprops(GLCM5, 'dissimilarity')[0]
        df['Diss_sim5'] = GLCM_diss5
        GLCM_hom5 = graycoprops(GLCM5, 'homogeneity')[0]
        df['Homogen5'] = GLCM_hom5
        GLCM_contr5 = graycoprops(GLCM5, 'contrast')[0]
        df['Contrast5'] = GLCM_contr5
        GLCM_asm5 = graycoprops(GLCM5, 'ASM')[0]
        df['ASM5'] = GLCM_asm5

        # # CANNY EDGE
        # edges = cv2.Canny(img, 100, 200)  # Image, min and max values
        # edges1 = edges.reshape(-1)
        # df['Canny Edge'] = edges1  # Add column to original dataframe
        #
        # from skimage.filters import roberts, sobel, scharr, prewitt
        #
        # # ROBERTS EDGE
        # edge_roberts = roberts(img)
        # edge_roberts1 = edge_roberts.reshape(-1)
        # df['Roberts'] = edge_roberts1
        #
        # # SOBEL
        # edge_sobel = sobel(img)
        # edge_sobel1 = edge_sobel.reshape(-1)
        # df['Sobel'] = edge_sobel1
        #
        # # SCHARR
        # edge_scharr = scharr(img)
        # edge_scharr1 = edge_scharr.reshape(-1)
        # df['Scharr'] = edge_scharr1
        #
        # # PREWITT
        # edge_prewitt = prewitt(img)
        # edge_prewitt1 = edge_prewitt.reshape(-1)
        # df['Prewitt'] = edge_prewitt1
        #
        # # GAUSSIAN with sigma=3
        # from scipy import ndimage as nd
        # gaussian_img = nd.gaussian_filter(img, sigma=3)
        # gaussian_img1 = gaussian_img.reshape(-1)
        # df['Gaussian s3'] = gaussian_img1
        #
        # # GAUSSIAN with sigma=7
        # gaussian_img2 = nd.gaussian_filter(img, sigma=7)
        # gaussian_img3 = gaussian_img2.reshape(-1)
        # df['Gaussian s7'] = gaussian_img3
        #
        # # MEDIAN with sigma=3
        # median_img = nd.median_filter(img, size=3)
        # median_img1 = median_img.reshape(-1)
        # df['Median s3'] = median_img1

        # Add more filters as needed
        # entropy = shannon_entropy(img)
        # df['Entropy'] = entropy

        # Append features from current image to the dataset
        image_dataset = pd.concat(
            [image_dataset, df])  # image_dataset.append(df)

    return image_dataset
