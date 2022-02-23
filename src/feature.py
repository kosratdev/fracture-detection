import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn import preprocessing


class FeatureExtractor:

    def __init__(self, train_images, train_labels, test_images, test_labels):
        self.train_images = np.array(train_images)
        self.train_labels = np.array(train_labels)
        self.test_images = np.array(test_images)
        self.test_labels = np.array(test_labels)
        self.__encode_labels()

    def __encode_labels(self):
        # Encode labels from text (folder names) to integers.
        le = preprocessing.LabelEncoder()
        le.fit(self.test_labels)
        test_labels_encoded = le.transform(self.test_labels)
        self.test_labels = test_labels_encoded

        le.fit(self.train_labels)
        train_labels_encoded = le.transform(self.train_labels)
        self.train_labels = train_labels_encoded

    def glcm_feature_extraction(self):
        return self.__glcm(self.train_images), self.train_labels, \
               self.__glcm(self.test_images), self.test_labels

    def __glcm(self, dataset):
        image_dataset = pd.DataFrame()
        for image in range(dataset.shape[0]):  # iterate through each file
            # Temporary data frame to capture information for each loop.
            df = pd.DataFrame()
            # Reset dataframe to blank after each loop.

            img = dataset[image, :, :]
            ################################################################
            # START ADDING DATA TO THE DATAFRAME

            # Full image
            # GLCM = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
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

            # Add more filters as needed
            # entropy = shannon_entropy(img)
            # df['Entropy'] = entropy

            # Append features from current image to the dataset
            image_dataset = pd.concat(
                [image_dataset, df])  # image_dataset.append(df)

        return image_dataset
