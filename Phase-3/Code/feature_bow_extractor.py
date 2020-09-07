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
imagedb14 = client["imagedb14"]
image_models = imagedb["image_models"]
label_unlabel_bow = imagedb14["label_unlabel_bow"]
# unlabelled_fd = imagedb["unlabelled_imagedb"]
# unlabelled_bow_centroid = imagedb["unlabelled_bow_centroids"]

dict = {}
list_subject_id = []

def calculate_Bow_unlabelled_imgs( img_list, model, k=40):
        feature_desc = None
        for descriptor in imagedb.image_models.find():
            if descriptor["_id"] in img_list:
                if feature_desc is None:
                    feature_desc = pd.DataFrame(descriptor[model])
                else:
                    feature_desc = [feature_desc, pd.DataFrame(descriptor[model])]
                    feature_desc = pd.concat(feature_desc, axis=0, sort=False)

        feature_desc = feature_desc.values
        ret = KMeans(n_clusters=k, max_iter=1000).fit(feature_desc)
        # centers = ret.cluster_centers_
        # centroids_dict = {}
        # centroids_dict[model] = centers.tolist()
        # imagedb.unlabelled_bow_centroids.insert_one(centroids_dict)
        dict = {}

        for item in imagedb.image_models.find():
            if item["_id"] in img_list:
                dict = {}
                img = pd.DataFrame(item[model])
                x = ret.predict(img)
                bag = np.zeros((k,), dtype=int)
                for z in x:
                    bag[z-1] += 1
                # imageID = item["_id"]
                bag = bag.tolist()
                dict["_id"] = item["_id"]
                dict["bag_" + model] = bag
                label_unlabel_bow.insert_one(dict)


def calculate_fd_unlabelled_imgs (folder, model):
        for image in glob.glob(os.path.join(folder, "*.jpg")):
            dict = {}
            dict["_id"] = image[-16:]
            if model == "CM":
                md = CM(image)
                lst = md.getFeatureDescriptors()
                dict["CM"] = lst

            elif model == "LBP":
                md = LBP(image)
                lst = md.getFeatureDescriptors()
                dict["LBP"] = lst

            elif model == "SIFT":
                md = SIFT(image)
                lst = md.getFeatureDescriptors()
                dict["SIFT"] = lst

            elif model == "HOG":
                md = HOG(image)
                lst = md.getFeatureDescriptors()
                lst = lst.tolist()
                dict["HOG"] = lst

            rec = imagedb.unlabelled_imagedb.insert_one(dict)


