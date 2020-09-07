from typing import Any, List
import cv2
import pandas as pd
import numpy as np
import os
from skimage.feature import local_binary_pattern
from ModelBase import ModelBase


class LBP(ModelBase):
    """ LBP class to hold information on LBP """
    def __init__(self, imagename):
        """ Initialize values for LBP class """
        super().__init__(imagename)
        head, tail = os.path.split(imagename)
        self.resultFile = "../output/LBP/" + tail[:len(tail)-4] + ".csv"
        self.grayFile = "../output/LBP_" + tail[:len(tail)-4] + ".png"
        self.w = 1600
        self.h = 1200
        self.b_dimensions = 100
        self.radius = 1
        self.no_points = 8 * self.radius

    def getFeatureDescriptors(self):
        if os.path.exists(self.resultFile):
            return []

        # Read the image
        img = cv2.imread(self.imgLoc)

        """ Converts image to Gray scale """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if not os.path.exists(self.grayFile):
            cv2.imwrite(self.grayFile, img_gray)

        """ Splits the Gray scale image into blocks of 100 by 100 pixels each.
                    This results in 192 blocks for the image. """
        gray_blocks = np.array([img_gray[i:i + self.b_dimensions, j:j + self.b_dimensions]
                                for j in range(0, self.w, self.b_dimensions)
                                for i in range(0, self.h, self.b_dimensions)])

        feature_descriptors: List[List[Any]] = []
        j = 0
        """ We have 192 blocks that we have split the image into. Now iterate through them. """
        while j < len(gray_blocks):
            j_block = gray_blocks[j]
            """ Local Binary Pattern  """
            lbp = local_binary_pattern(j_block, self.no_points, self.radius, method='uniform')
            """ Calculate the histogram """
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.no_points + 3),
                                     range=(0, self.no_points + 2))
            """ Add the LBP data for the current image block to the list containing all LBP data for the image. """
            feature_descriptors.append(list([x.item() for x in hist]))
            j += 1

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

        """ Code for calculating Cosine Similarity """
        init_flat = des1.values.flatten()
        cmp_flat = des2.values.flatten()
        """ Compute cosine similarity of the target image to each other image. """
        rank = np.dot(init_flat, cmp_flat) / (np.linalg.norm(init_flat) * np.linalg.norm(cmp_flat))

        """ Code for calculating Euclidean Distance """
        # squared_diff = np.square(np.subtract(des1.values, des2.values))
        # rank = np.sqrt(squared_diff.sum(0).sum(0))

        return rank

    def compareImages(self, imgLoc):
        obj2 = LBP(imgLoc)
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
