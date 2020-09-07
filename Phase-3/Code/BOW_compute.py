from LocalBinaryPatterns import LBP
from ColorMoments import CM
from SIFT import SIFT
from HOGmain import HOG
import Constants as const
import pymongo
import numpy as np

client = pymongo.MongoClient('localhost', const.MONGODB_PORT)
imagedb = client["imagedb"]
cluster_centers = imagedb["centroids"]

def BOW(img_path, model):
    print(img_path)
    if model == "CM":
        md = CM(img_path)
        lst = md.getFeatureDescriptors()
        label = 0

    elif model == "LBP":
        md = LBP(img_path)
        lst = md.getFeatureDescriptors()
        label = 3
    elif model == "SIFT":
        md = SIFT(img_path)
        lst = md.getFeatureDescriptors()
        label = 2

    elif model == "HOG":
        md = HOG(img_path)
        lst = md.getFeatureDescriptors()
        label = 1

    centers = imagedb.centroids.find()[label][model]

    bag = np.zeros((40,), dtype=int)
    for desc in lst:
        all_dist = []
        for c in centers:
            euc_dis = np.square(np.subtract(desc, c))
            dist = np.sqrt(euc_dis.sum())
            all_dist.append(dist)
        index = all_dist.index(min(all_dist))
        bag[index] += 1

    return bag



