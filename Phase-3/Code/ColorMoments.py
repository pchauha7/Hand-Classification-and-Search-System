from typing import Any, Union, List

import cv2
import pandas as pd
import numpy as np
import os
from scipy.stats import skew
from ModelBase import ModelBase

class CM(ModelBase):
    def __init__(self, imagname):
        super().__init__(imagname)
        head, tail = os.path.split(imagname)
        self.resultFile = "../output/CM/" + tail[:len(tail)-4] + ".csv"
        self.yuvFile = "../output/CM_" + tail[:len(tail) - 4] + ".png"

    def getFeatureDescriptors(self):
        # if (os.path.exists(self.resultFile)):
        #     return []

        # Read the image
        img = cv2.imread(self.imgLoc)

        # Converting the image into YUV format
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        if (not os.path.exists(self.yuvFile)):
            # Writing the yuv conversion the image file
            cv2.imwrite(self.yuvFile, img_yuv)

        img_yuv_array = np.array(img_yuv)

        row = int(len(img_yuv_array) / 100)
        col = int(len(img_yuv_array[0]) / 100)

        feature_descriptors: List[List[Any]] = []
        for i in range(row):
            for j in range(col):
                # 100X100 slice of the image
                img_slice = img_yuv_array[i * 100:i * 100 + 100, j * 100:j * 100 + 100]
                img_slice_copy = np.copy(img_slice)
                # Calculate the sum of each Y U V in the array
                img_slice_copy = img_slice_copy.sum(0).sum(0)
                # Divide it to get the Ist moment ie Mean
                moment1 = np.true_divide(img_slice_copy, 10000)

                img_slice_copy = np.copy(img_slice)
                # Calculate the square of their(YUV) difference with mean to calculate SD
                img_slice_copy = np.square(np.subtract(img_slice_copy, moment1))
                # Calculate the sum of each SD of (Y U V) in the array
                img_slice_copy = img_slice_copy.sum(0).sum(0)
                moment2 = np.true_divide(img_slice_copy, 10000)
                moment2 = np.sqrt(moment2)

                #Calculating the skew
                img_slice_copy = np.copy(img_slice)
                y_skew = skew(np.concatenate(img_slice_copy[:, :, 0]))
                u_skew = skew(np.concatenate(img_slice_copy[:, :, 1]))
                v_skew = skew(np.concatenate(img_slice_copy[:, :, 2]))
                final_moment = np.concatenate((moment1, moment2, [y_skew, u_skew, v_skew]))

                feature_descriptors.append(list(final_moment))

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

        #Assigning weight for calculation of matching score
        weight = np.array([2,1,1,2,1,1,2,1,1])
        des1 = np.multiply(des1.values, weight)
        des2 = np.multiply(des2.values, weight)
        #Applying manhattan distance calculation
        res = np.abs(np.subtract(des1, des2))
        rank = res.sum(0).sum(0)
        return rank

    def compareImages(self, imgLoc):
        obj2 = CM(imgLoc)
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




