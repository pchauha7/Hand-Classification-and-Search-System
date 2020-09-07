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
        self.resultFile = "../output/SIFT/" + tail[:len(tail) - 4] + ".csv"
        self.labelFile = "../output/SIFT_" + tail[:len(tail) - 4] + ".jpg"

    def getFeatureDescriptors(self):
        if os.path.exists(self.resultFile):
            return []

        #Reading the image
        img = cv2.imread(self.imgLoc)
        # Changing it to GRAY format
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        # Computing key points
        keypoints = sift.detect(gray, None)
        # Drawing keypoints to an image
        img = cv2.drawKeypoints(img, keypoints, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(self.labelFile, img)
        #creating feature descriptors based on key points
        keypoints, feature_descriptor = sift.detectAndCompute(gray, None)
        print(keypoints)
        return feature_descriptor

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

        similarity_value = []
        for des1_val in des1:
            #Comparing with all the descriptors
            key_arr = PriorityQueue()
            for des2_val in des2:
                #calculating euclidean distance
                euc_dis = np.square(np.subtract(des1_val, des2_val))
                val = np.sqrt(euc_dis.sum(0))
                #Adding distance to priority queues
                key_arr.put(val)
            dis1 = key_arr.get()
            dis2 = key_arr.get()
            # Decide if this keypoint is valid or not
            if dis1/dis2 < 0.7:
                similarity_value.append(dis1)

        similarity_value
        #print(similarity_value)
        return len(similarity_value)

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

