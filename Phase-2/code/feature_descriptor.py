from LocalBinaryPatterns import LBP
from ColorMoments import CM
from SIFT import SIFT
from HOGmain import HOG
import os
import glob
import pymongo
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import Constants as const

# import dbtask

client = pymongo.MongoClient('localhost', const.MONGODB_PORT)
imagedb = client["imagedb"]
mydb = imagedb["image_models"]
cluster_centers = imagedb["centroids"]
subjects = imagedb["subjects"]

dict = {}
list_subject_id = []
dict_subjects = {}


def subjectMeta():
    print("enter")
    subjectID = imagedb.ImageMetadata.distinct("SubjectID")
    for id in subjectID:
        dorsal_left = imagedb.ImageMetadata.find({"SubjectID": id, "aspectOfHand": "dorsal", "Orientation": "left"})
        dorsal_left = [item["imageName"] for item in dorsal_left]
        dorsal_right = imagedb.ImageMetadata.find({"SubjectID": id, "aspectOfHand": "dorsal", "Orientation": "right"})
        dorsal_right = [item["imageName"] for item in dorsal_right]
        palmar_left = imagedb.ImageMetadata.find({"SubjectID": id, "aspectOfHand": "palmar", "Orientation": "left"})
        palmar_left = [item["imageName"] for item in palmar_left]
        palmar_right = imagedb.ImageMetadata.find({"SubjectID": id, "aspectOfHand": "palmar", "Orientation": "right"})
        palmar_right = [item["imageName"] for item in palmar_right]
        dict_subjects["_id"] = id
        dict_subjects["dorsal_left"] = dorsal_left
        dict_subjects["dorsal_right"] = dorsal_right
        dict_subjects["palmar_left"] = palmar_left
        dict_subjects["palmar_right"] = palmar_right
        print(dict_subjects)
        rec = imagedb.subjects.insert_one(dict_subjects)


def createKMeans(model, k):
    feature_desc = None
    for descriptor in imagedb.image_models.find():
        if feature_desc is None:
            feature_desc = pd.DataFrame(descriptor[model])
        else:
            feature_desc = [feature_desc, pd.DataFrame(descriptor[model])]
            feature_desc = pd.concat(feature_desc, axis=0, sort=False)

    feature_desc = feature_desc.values
    ret = KMeans(n_clusters=k, max_iter=1000).fit(feature_desc)
    centers = ret.cluster_centers_
    centroids_dict = {}
    centroids_dict[model] = centers.tolist()
    imagedb.centroids.insert_one(centroids_dict)

    for item in imagedb.image_models.find():
        img = pd.DataFrame(item[model])
        x = ret.predict(img)
        bag = np.zeros((k,), dtype=int)
        for z in x:
            bag[z-1] += 1
        imageID = item["_id"]
        bag = bag.tolist()
        imagedb.image_models.update_one({"_id": imageID}, {"$set": {"bag_"+model: bag}})


def calculate_fd(path):
    for image in glob.glob(os.path.join(path, "*.jpg")):
        dict = {}
        dict["_id"] = image[-16:]

        md = CM(image)
        lst = md.getFeatureDescriptors()
        #print(lst)
        dict["CM"] = lst

        md = LBP(image)
        lst = md.getFeatureDescriptors()
        dict["LBP"] = lst

        md = SIFT(image)
        lst = md.getFeatureDescriptors()
        dict["SIFT"] = lst

        md = HOG(image)
        lst = md.getFeatureDescriptors()
        lst = lst.tolist()
        dict["HOG"] = lst

        #dict["HOG"] = lst.tolist()
        #print(type(lst.tolist()))

        rec = imagedb.image_models.insert_one(dict)


# Main
# imagedb.image_models.drop()
# imagedb.subjects.drop()
# path = input("Enter Path: ")
# calculate_fd(path)
#
# createKMeans("CM", 40)
# createKMeans("HOG", 40)
# createKMeans("SIFT", 40)
# createKMeans("LBP", 40)
# subjectMeta()
