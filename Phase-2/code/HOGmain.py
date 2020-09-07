import cv2
import pandas as pd
import numpy as np
import os
from skimage.feature import hog
from ModelBase import ModelBase


class HOG(ModelBase):
    """ HOG class to hold information on HOG """

    def __init__(self, imagename):
        """ Initialize values for HOG class """
        super().__init__(imagename)
        head, tail = os.path.split(imagename)
        self.resultFile = "../output/HOG/" + tail[:len(tail) - 4] + ".csv"
        self.hogFile = "../output/HOG_" + tail[:len(tail) - 4] + ".png"

    def getFeatureDescriptors(self):
        if os.path.exists(self.resultFile):
            return []

        # Read the image
        img = cv2.imread(self.imgLoc)

        image_resized = cv2.resize(img, (160, 120))
        feature_descriptors = hog(image_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                  visualize=False,
                                  multichannel=True)
        feature_descriptors.resize(266, 36)

        return feature_descriptors

    def createFeatureOutputFile(self, feature_descriptors):
        if not os.path.exists(self.resultFile):
            df = pd.DataFrame(feature_descriptors)
            # Writing data to csv file
            df.to_csv(self.resultFile, index=False, header=False)

    def compareDescriptors(self, outFile2):
        # Reading csv files containing descriptors
        des1 = pd.read_csv(self.resultFile, sep=',', header=None)
        des2 = pd.read_csv(outFile2, sep=',', header=None)

        """ Code for calculating Euclidean Distance """
        dist = ((((des1.subtract(des2)) ** 2).sum(axis=1)) ** 0.5).values.sum()

        return dist

    def compareImages(self, imgLoc):
        obj2 = HOG(imgLoc)
        # checking if comparison is not between same image files
        if self.resultFile == obj2.resultFile:
            return -1
        # checking if descriptor already calculated
        if not os.path.exists(self.resultFile):
            des = self.getFeatureDescriptors()
            self.createFeatureOutputFile(des)
        if not os.path.exists(obj2.resultFile):
            des = obj2.getFeatureDescriptors()
            obj2.createFeatureOutputFile(des)
        return self.compareDescriptors(obj2.resultFile)
