import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pymongo
import Visualizer as vz
import Constants as const
from feature_bow_extractor import *

client = pymongo.MongoClient('localhost', const.MONGODB_PORT)
imagedb = client["imagedb"]
imagedb14 = client["imagedb14"]
label_unlabel_bow = imagedb14["label_unlabel_bow"]
dorsal_centroids = imagedb14["dorsal_centroids"]
palmar_centroids = imagedb14["palmar_centroids"]
AllHands_metadata = imagedb["HandInfo"]


def create_KMeans(model, img_list, label, c):
    feature_desc = None
    model_name = 'bag_'+model
    for descriptor in label_unlabel_bow.find():
        if descriptor["_id"] in img_list:
            if feature_desc is None:
                feature_desc = pd.DataFrame(descriptor[model_name])
                feature_desc = feature_desc.transpose()
            else:
                feature_desc = [feature_desc, pd.DataFrame(descriptor[model_name]).transpose()]
                feature_desc = pd.concat(feature_desc, axis=0, sort=False)

    feature_desc = feature_desc.values
    ret = KMeans(n_clusters=c, max_iter=1000).fit(feature_desc)
    centers = ret.cluster_centers_
    centroids_dict = {}
    centroids_dict[model] = centers.tolist()
    if label == 'dorsal':
        dorsal_centroids.insert_one(centroids_dict)
    else:
        palmar_centroids.insert_one(centroids_dict)

def dorsal_palmar_cluster(label_path, unlabelled_path, model, k, c):
    label_meta_collection = label_path
    unlabelled_meta_collection = unlabelled_path
    label_meta_obj = imagedb14[label_meta_collection]
    unlabel_meta_obj = imagedb14[unlabelled_meta_collection]
    label_img_list = []
    unlabel_img_list = []

    for descriptor in label_meta_obj.find():
        label_img_list.append(descriptor["imageName"])
    for descriptor in unlabel_meta_obj.find():
        unlabel_img_list.append(descriptor["imageName"])

    img_list = label_img_list + unlabel_img_list
    calculate_Bow_unlabelled_imgs(img_list, model, k)

    dorsal_img_list = []
    palmar_img_list = []
    for descriptor in label_meta_obj.find():
        aspect_of_hand = descriptor["aspectOfHand"].split(" ")
        if aspect_of_hand[0] == 'dorsal':
            dorsal_img_list.append(descriptor["imageName"])
        else:
            palmar_img_list.append(descriptor["imageName"])

    create_KMeans(model, dorsal_img_list, 'dorsal', c)
    create_KMeans(model, palmar_img_list, 'palmar', c)

def unlabelled_img_classification(unlabelled_path, model):
    dorsal_centers = []
    palmar_centers = []
    unlabelled_imgs_bow = []
    unlabelled_img_list = []
    model_name = 'bag_' + model

    unlabelled_meta_collection = unlabelled_path
    unlabel_meta_obj = imagedb14[unlabelled_meta_collection]
    for descriptor in unlabel_meta_obj.find():
        unlabelled_img_list.append(descriptor["imageName"])


    # fetching centroid of dorsal images from db
    for descriptor in dorsal_centroids.find():
        dorsal_centers.append(descriptor[model])

    # fetching centroid of palmar images from db
    for descriptor in palmar_centroids.find():
        palmar_centers.append(descriptor[model])

    img_list = []
    for descriptor in label_unlabel_bow.find():
        if descriptor['_id'] in unlabelled_img_list:
            unlabelled_imgs_bow.append(descriptor[model_name])
            img_list.append(descriptor["_id"])

    # classification of unlabelled images by calculating distance
    # from centers of dorsal and centers of palmar
    results = {}

    for i, desc in enumerate(unlabelled_imgs_bow):
        dorsal_dist = []
        palmar_dist = []
        for j, cen1 in enumerate(dorsal_centers[0]):
            euc_dis = np.square(np.subtract(unlabelled_imgs_bow[i], dorsal_centers[0][j]))
            dist = np.sqrt(euc_dis.sum(0))
            dorsal_dist.append(dist)
        min_dorsal_dist = min(dorsal_dist)

        for j ,cen2 in enumerate(palmar_centers[0]):
            euc_dis = np.square(np.subtract(unlabelled_imgs_bow[i], palmar_centers[0][j]))
            dist = np.sqrt(euc_dis.sum(0))
            palmar_dist.append(dist)
        min_palmar_dist = min(palmar_dist)

        if min_dorsal_dist > min_palmar_dist:
            results[img_list[i]] = "palmar"
        else:
            results[img_list[i]] = "dorsal"

    classification_result = []
    for key, value in results.items():
        temp = [key, value]
        classification_result.append(temp)

    df = pd.DataFrame(classification_result, columns=["Image Names", "Label"])
    # df.to_csv("D:/CSE515MultiMediaWebDB/Phase3_outputs/Task2_accuracy_set1_hog.csv", index=None)

    total_imgs = 0
    correct_imgs = 0
    for key, values in results.items():
        total_imgs += 1
        for descriptor in AllHands_metadata.find():
            if descriptor["imageName"] == key:
                aspect_of_hand = descriptor["aspectOfHand"].split(" ")
                true_label = aspect_of_hand[0]
                if true_label == values:
                    correct_imgs += 1

    accuracy = (correct_imgs / total_imgs) * 100
    print("accuracy is : %d" % accuracy)
    return results, accuracy

def Query_input(c, folder, classfiy):

    imagedb14.dorsal_centroids.drop()
    imagedb14.palmar_centroids.drop()
    imagedb14.label_unlabel_bow.drop()
    dorsal_palmar_cluster(folder, classfiy, "SIFT", 80, c)
    result = unlabelled_img_classification(classfiy, "SIFT")
    return result

# result = Query_input(5, "labelled_set2", "unlabelled_set1")
# vz.visualize_labelled_images(result[0], 0, '', 5, result[1])

# result = Query_input(10, "labelled_set2", "unlabelled_set1")
# vz.visualize_labelled_images(result[0], 0, '', 10, result[1])

# result = Query_input(5, "labelled_set2", "unlabelled_set2")
# vz.visualize_labelled_images(result[0], 0, '', 5, result[1])

result = Query_input(10, "labelled_set2", "unlabelled_set2")
vz.visualize_labelled_images(result[0], 0, '', 10, result[1])
