from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import pymongo
import os
from numpy import linalg
import math
import shutil
import Visualizer as vz
import Constants as const
import BOW_compute

client = pymongo.MongoClient('localhost', const.MONGODB_PORT)
imagedb = client["imagedb"]
mydb = imagedb["image_models"]
meta = imagedb["ImageMetadata"]
dr_name = 'PCA'


class P_CA(object):

    def createKLatentSymantics(self, model, k):
        #svd1 = TruncatedSVD(k)
        model_name = model
        model = "bag_" + model
        frames = []
        img_list = []
        for descriptor in imagedb.image_models.find():
            frames.append(descriptor[model])
            img_list.append(descriptor["_id"])

        frames = pd.DataFrame(frames)
        print(frames.shape)
        mean_vec = np.mean(frames, axis=0)
        cov_mat = np.cov(frames.T)
        print(cov_mat.shape)

        # Compute the eigen values and vectors using numpy
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)

        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        y = 0
        for p in eig_pairs:
            feature = pd.DataFrame(p[1])
            y += 1
            if y == 1:
                frame = feature
            else:
                frame = [frame, feature]
                frame = pd.concat(frame, axis=1, sort=False)

        visualizeArr = []
        # code for data latent semantics visualizer
        for i in range(k):
            col = frame.iloc[:, i]
            arr = []
            for j, val in enumerate(col):
                arr.append((j, val))
            arr.sort(key=lambda x: x[1], reverse=True)
            """ Only take the top 5 data objects to report for each latent semantic """
            visualizeArr.append(arr[:5])
            print("Printing term-weight pair for latent Semantic L{}:".format(i + 1))
            print(arr)
        visualizeArr = pd.DataFrame(visualizeArr)

        vz.visualize_feature_ls(visualizeArr, dr_name, model_name, '')
        feat_latent = np.transpose(eig_vecs)
        # code for feature latent semantics visualizer
        feature_latentSemantics = {}
        for l in range(k):
            results = []
            for descriptor in imagedb.image_models.find():
                res = feat_latent[l] @ descriptor[model]
                results.append((descriptor["_id"], res))
            results.sort(key=lambda x: x[1], reverse=True)
            feature_latentSemantics[l+1] = results[0][0]

        print(feature_latentSemantics)

        vz.visualize_ftr_ls_hdp(feature_latentSemantics, dr_name, model_name)


    def mSimilarImage(self, imgLoc, model, k, m):
        model_name = model
        img_list = []
        pca = PCA(k)
        # model = "bag_" + model
        feature_desc = []
        img_list = []
        for descriptor in imagedb.image_models.find():
            feature_desc.append(descriptor[model])
            img_list.append(descriptor["_id"])

        feature_desc_transformed = pca.fit_transform(feature_desc)
        print(feature_desc_transformed)

        head, tail = os.path.split(imgLoc)

        id = img_list.index(tail)

        rank_dict = {}
        for i, row in enumerate(feature_desc_transformed):
            if (i == id):
                continue
            euc_dis = np.square(np.subtract(feature_desc_transformed[id], feature_desc_transformed[i]))
            match_score = np.sqrt(euc_dis.sum(0))
            rank_dict[img_list[i]] = match_score

        # res_dir = os.path.join('..', 'output', model[4:], 'match')
        # if os.path.exists(res_dir):
        #     shutil.rmtree(res_dir)
        # os.mkdir(res_dir)
        count = 0
        print("\n\nNow printing top {} matched Images and their matching scores".format(m))
        # sorted_dict = sorted(rank_dict.items(), key=lambda item: item[1])
        head, tail = os.path.split(imgLoc)
        # vz.visualize_matching_images(tail, rank_dict, k, m, dr_name, model_name, '')
        vz.visualize_relevance_feedback(tail, rank_dict, m)
        for key, value in sorted(rank_dict.items(), key=lambda item: item[1]):
            if count < m:
                print(key + " has matching score:: " + str(value))
                # shutil.copy(os.path.join(head, key), res_dir)
                count += 1
            else:
                break

    def LabelLatentSemantic(self, label, model, k):
        model_name = model
        model = "bag_" + model
        if label == "left" or label == "right":
            search = "Orientation"
        elif label == "dorsal" or label == "palmar":
            search = "aspectOfHand"
        elif label == "Access" or label == "NoAccess":
            search = "accessories"
        elif label == "male" or label == "female":
            search = "gender"
        else:
            print("Please provide correct label")
            exit(1)

        frames = []
        img_list = []
        imageslist_Meta = []

        for descriptor in imagedb.ImageMetadata.find():
            if descriptor[search] == label:
                imageslist_Meta.append(descriptor["imageName"])

        for descriptor in imagedb.image_models.find():
            if descriptor["_id"] in imageslist_Meta:
                frames.append(descriptor[model])
                img_list.append(descriptor["_id"])

        frames = pd.DataFrame(frames)

        mean_vec = np.mean(frames, axis=0)
        cov_mat = np.cov(frames.T)

        # Compute the eigen values and vectors using numpy
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)

        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        y = 0
        for p in eig_pairs:
            feature = pd.DataFrame(p[1])
            y += 1
            if y == 1:
                frame = feature
            else:
                frame = [frame, feature]
                frame = pd.concat(frame, axis=1, sort=False)

        visualizeArr = []

        for i in range(k):
            col = frame.iloc[:, i]
            arr = []
            for k, val in enumerate(col):
                arr.append((k, val))
            arr.sort(key=lambda x: x[1], reverse=True)
            """ Only take the top 5 data objects to report for each latent semantic """
            visualizeArr.append(arr[:5])
            print("Printing term-weight pair for latent Semantic L{}:".format(i + 1))
            print(arr)
        visualizeArr = pd.DataFrame(visualizeArr)
        vz.visualize_feature_ls(visualizeArr, dr_name, model_name, label)

    def mSimilarImage_Label(self, imgLoc, label, model, k, m):
        model_name = model
        label_str = label
        if label == "left" or label == "right":
            search = "Orientation"
        elif label == "dorsal" or label == "palmar":
            search = "aspectOfHand"
        elif label == "Access" or label == "NoAccess":
            search = "accessories"
            if label == "Access":
                label = 1
                label_str = 'With Accessories'
            else:
                label = 0
                label_str = 'Without Accessories'

        elif label == "male" or label == "female":
            search = "gender"
        else:
            print("Please provide correct label")
            exit(1)

        pca = PCA(k)
        model = "bag_" + model
        img_list = []
        imageslist_Meta = []
        frames = []

        for descriptor in imagedb.ImageMetadata.find():
            if descriptor[search] == label:
                imageslist_Meta.append(descriptor["imageName"])

        # print(len(imageslist_Meta))

        for descriptor in imagedb.image_models.find():
            if descriptor["_id"] in imageslist_Meta:
                frames.append(descriptor[model])
                img_list.append(descriptor["_id"])

        feature_desc_transformed = pca.fit_transform(frames)
        print(feature_desc_transformed)

        head, tail = os.path.split(imgLoc)

        id = img_list.index(tail)

        rank_dict = {}
        for i, row in enumerate(feature_desc_transformed):
            if (i == id):
                continue
            euc_dis = np.square(np.subtract(feature_desc_transformed[id], feature_desc_transformed[i]))
            match_score = np.sqrt(euc_dis.sum(0))
            rank_dict[img_list[i]] = match_score

        # res_dir = os.path.join('..', 'output', model[4:], 'match')
        # if os.path.exists(res_dir):
        #     shutil.rmtree(res_dir)
        # os.mkdir(res_dir)
        count = 0
        print("\n\nNow printing top {} matched Images and their matching scores".format(m))
        vz.visualize_matching_images(tail, rank_dict, k, m, dr_name, model_name, label_str)
        for key, value in sorted(rank_dict.items(), key=lambda item: item[1]):
            if count < m:
                print(key + " has matching score:: " + str(value))
                # shutil.copy(os.path.join(head, key), res_dir)
                count += 1
            else:

                break

    def ImageClassfication(self, imgLoc, model, k):
        model_name = model
        result = {}
        model = "bag_" + model
        head, tail = os.path.split(imgLoc)
        query_desc = []
        flag = False
        for descriptor in imagedb.image_models.find():
            if descriptor["_id"] == tail:
                query_desc.append(descriptor[model])
                flag = True

        Labels = ["dorsal_left", "dorsal_right", "palmar_left", "palmar_right", "Access", "NoAccess", "male", "female"]

        if len(query_desc) == 0:
            query_desc = BOW_compute.BOW(imgLoc, model_name)

        for label in Labels:
            label_Desc = []
            desc_img_list = []
            imageslist_Meta = []
            id = -1

            if label in ["dorsal_left", "dorsal_right", "palmar_left", "palmar_right"]:
                for subject in imagedb.subjects.find():
                    for img in subject[label]:
                        label_Desc.append(imagedb.image_models.find({"_id": img})[0][model])
                        desc_img_list.append(img)

            elif label == "Access" or label == "NoAccess":
                search = "accessories"
                if label == "Access":
                    label = 1
                else:
                    label = 0

                for descriptor in imagedb.ImageMetadata.find():
                    if descriptor[search] == label:
                        imageslist_Meta.append(descriptor["imageName"])

                for descriptor in imagedb.image_models.find():
                    if descriptor["_id"] in imageslist_Meta:
                        label_Desc.append(descriptor[model])
                        desc_img_list.append(descriptor["_id"])

            elif label == "male" or label == "female":
                search = "gender"
                for descriptor in imagedb.ImageMetadata.find():
                    if descriptor[search] == label:
                        imageslist_Meta.append(descriptor["imageName"])

                for descriptor in imagedb.image_models.find():
                    if descriptor["_id"] in imageslist_Meta:
                        label_Desc.append(descriptor[model])
                        desc_img_list.append(descriptor["_id"])

            if not flag:
                label_Desc.append(query_desc)
                desc_img_list.append(tail)
                id = len(desc_img_list) - 1

            pca = PCA(k)
            lda_Obj = pca.fit(label_Desc)
            label_desc_transformed = lda_Obj.transform(label_Desc)
            # query_desc_transformed = lda_Obj.transform(query_desc)
            # print(query_desc_transformed[:10])
            print(label_desc_transformed)

            dist = []

            for i, db_desc in enumerate(label_desc_transformed):
                if desc_img_list[i] == tail:
                    continue
                euc_dis = np.square(np.subtract(db_desc, label_desc_transformed[id]))
                match_score = np.sqrt(euc_dis.sum())
                dist.append(match_score)

            result[label] = min(dist)

        classification = {}

        if result["dorsal_left"] > result["dorsal_right"]:
            semi_final1 = result["dorsal_right"]
            conclusion1 = "dorsal_right"
        else:
            semi_final1 = result["dorsal_left"]
            conclusion1 = "dorsal_left"

        if result["palmar_left"] > result["palmar_right"]:
            semi_final2 = result["palmar_right"]
            conclusion2 = "palmar_right"
        else:
            semi_final2 = result["palmar_left"]
            conclusion2 = "palmar_left"

        if semi_final1 > semi_final2:
            res = conclusion2.split("_")
            classification['Aspect of Hand:'] = res[0]
            classification['Orientation:'] = res[1]
            print(res[1])
            print(res[0])
        else:
            res = conclusion1.split("_")
            classification['Aspect of Hand:'] = res[0]
            classification['Orientation:'] = res[1]
            print(res[1])
            print(res[0])

        if result[1] > result[0]:
            classification['Accessories:'] = 'Without Accessories'
            print("NoAccess")
        else:
            classification['Accessories:'] = 'With Accessories'
            print("Access")

        if result["male"] > result["female"]:
            classification['Gender:'] = 'Female'
            print("female")
        else:
            classification['Gender:'] = 'Male'
            print("male")

        vz.visualize_classified_image(tail, classification, dr_name, model_name, k)