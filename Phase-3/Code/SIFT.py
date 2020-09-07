import cv2
import pandas as pd
import numpy as np
import csv
import os
from queue import PriorityQueue
from ModelBase import ModelBase


class SIFT(ModelBase):
    def __init__(self, imagname):
        super().__init__(imagname)
        head, tail = os.path.split(imagname)
        self.resultFile = "./output/SIFT/" + tail[:len(tail) - 4] + ".csv"
        self.labelFile = "../output/SIFT_" + tail[:len(tail) - 4] + ".jpg"

    def getFeatureDescriptors(self, vector_size=70):
        if os.path.exists(self.resultFile):
            return []

        #Reading the image
        img = cv2.imread(self.imgLoc)
        try:
            # Changing it to GRAY format
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            # Computing key points
            keypoints = sift.detect(gray, None)
            # Number of keypoints is varies depend on image size and color pallet
            # Sorting them based on keypoint response value(bigger is better)
            kps = sorted(keypoints, key=lambda x: -x.response)[:vector_size]
            # computing descriptors vector
            kps, dsc = sift.compute(img, kps)
            # Making descriptor of same size
            # Descriptor vector size is 128
            if len(dsc) < vector_size:
                # if we have less the 50 descriptors then just adding zeros at the
                # end of our feature vector
                concat = vector_size - len(dsc)
                dsc = np.concatenate((dsc, np.zeros((concat, 128))), axis=0)
            descriptor = dsc.tolist()

        except cv2.error as e:
            print('Error: ', e)
            return None

        return descriptor


    def createFeatureOutputFile(self, feature_descriptor):
        if not os.path.exists(self.resultFile):
            df = pd.DataFrame(feature_descriptor)
            # Writing data to csv file
            df.to_csv(self.resultFile, index=False, header=False)

    def compareDescriptors(self, output2):
        # Reading csv files containing descriptors
        des1 = pd.read_csv(self.resultFile, sep=',', header=None)
        des2 = pd.read_csv(output2, sep=',', header=None)
        des1 = des1.values
        des2 = des2.values

        all_dist = []
        Best_Matches = []

        for des1_val in des1:
            #Comparing with all the descriptors

            for des2_val in des2:
                #calculating euclidean distance
                distance = (sum([(a - b) ** 2 for a, b in zip(des1_val, des2_val)])) ** 0.5
                all_dist.append(distance)
            min_dist = min(all_dist)
            Best_Matches.append(min_dist)
        distance_ = int(sum(Best_Matches) / 70)

        return distance_

    def compareImages(self, imgLoc):
        obj2 = SIFT(imgLoc)
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

