from sklearn.decomposition import TruncatedSVD
from scipy.linalg import svd
import Visualizer as vz
import Constants as const
import numpy as np
import pandas as pd
import pymongo
import os
import shutil
import BOW_compute

client = pymongo.MongoClient('localhost', const.MONGODB_PORT)
imagedb = client["imagedb"]
mydb = imagedb["image_models"]
meta = imagedb["ImageMetadata"]
dr_name = 'SVD'

class SVD(object):

    def createKLatentSymantics(self, model, k):
        #svd1 = TruncatedSVD(k)
        model_name = model
        model = "bag_" + model
        feature_desc = []
        img_list = []
        for descriptor in imagedb.image_models.find():
            feature_desc.append(descriptor[model])
            img_list.append(descriptor["_id"])

        #feature_desc_transformed = svd1.fit_transform(feature_desc)
        U, S, V = svd(feature_desc, full_matrices=False)

        visualizeArr = []
        # code for data latent semantics visualizer
        for i in range(k):
            col = U[:, i]
            arr = []
            for j, val in enumerate(col):
                arr.append((str(img_list[j]), val))
            arr.sort(key=lambda x: x[1], reverse=True)
            """ Only take the top 5 data objects to report for each latent semantic """
            visualizeArr.append(arr[:5])
            print("Printing term-weight pair for latent Semantic {}({}):".format(i + 1, S[i]))
            print(arr)
        visualizeArr = pd.DataFrame(visualizeArr)

        # code for feature latent semantics visualizer
        feature_latentSemantics = {}
        for l in range(k):
            results = []
            for descriptor in imagedb.image_models.find():
                res = V[l] @ descriptor[model]
                results.append((descriptor["_id"], res))
            results.sort(key=lambda x: x[1], reverse=True)
            feature_latentSemantics[l + 1] = results[0][0]

        print(feature_latentSemantics)

        vz.visualize_data_ls(visualizeArr, dr_name, model_name, '')

    def mSimilarImage(self, imgLoc, model, k, m):
        model_name = model
        img_list = []
        svd = TruncatedSVD(k)
        model = "bag_" + model
        feature_desc = []
        img_list = []
        for descriptor in imagedb.image_models.find():
            feature_desc.append(descriptor[model])
            img_list.append(descriptor["_id"])

        feature_desc_transformed = svd.fit_transform(feature_desc)

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
        vz.visualize_matching_images(tail, rank_dict, m, dr_name, model_name, '')
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

        feature_desc = []
        img_list = []
        imageslist_Meta = []

        for descriptor in imagedb.ImageMetadata.find():
            if descriptor[search] == label:
                imageslist_Meta.append(descriptor["imageName"])

        print(len(imageslist_Meta))

        for descriptor in imagedb.image_models.find():
            if descriptor["_id"] in imageslist_Meta:
                feature_desc.append(descriptor[model])
                img_list.append(descriptor["_id"])

        print(len(img_list))
        svd1 = TruncatedSVD(k)

        feature_desc_transformed = svd1.fit_transform(feature_desc)
        U, S, V = svd(feature_desc, full_matrices=False)

        visualizeArr = []

        for i in range(k):
            col = U[:, i]
            arr = []
            for k, val in enumerate(col):
                arr.append((str(img_list[k]), val))
            arr.sort(key=lambda x: x[1], reverse=True)
            """ Only take the top 5 data objects to report for each latent semantic """
            visualizeArr.append(arr[:5])
            print("Printing term-weight pair for latent Semantic {}({}):".format(i + 1, S[i]))
            print(arr)
        visualizeArr = pd.DataFrame(visualizeArr)
        vz.visualize_data_ls(visualizeArr, dr_name, model_name, label)

        return feature_desc_transformed

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

        svd = TruncatedSVD(k)
        model = "bag_" + model
        img_list = []
        imageslist_Meta = []
        feature_desc = []

        for descriptor in imagedb.ImageMetadata.find():
            if descriptor[search] == label:
                imageslist_Meta.append(descriptor["imageName"])

        print(len(imageslist_Meta))

        for descriptor in imagedb.image_models.find():
            if descriptor["_id"] in imageslist_Meta:
                feature_desc.append(descriptor[model])
                img_list.append(descriptor["_id"])

        feature_desc_transformed = svd.fit_transform(feature_desc)

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
        vz.visualize_matching_images(tail, rank_dict, m, dr_name, model_name, label_str)
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

        if len(query_desc) == 0:
            query_desc = BOW_compute.BOW(imgLoc, model_name)


        Labels = ["dorsal_left", "dorsal_right", "palmar_left", "palmar_right", "Access", "NoAccess", "male", "female"]

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
                        if (img == tail):
                            id = len(desc_img_list) -1



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
                        if (img == tail):
                            id = len(desc_img_list) -1

            elif label == "male" or label == "female":
                search = "gender"
                for descriptor in imagedb.ImageMetadata.find():
                    if descriptor[search] == label:
                        imageslist_Meta.append(descriptor["imageName"])

                for descriptor in imagedb.image_models.find():
                    if descriptor["_id"] in imageslist_Meta:
                        label_Desc.append(descriptor[model])
                        desc_img_list.append(descriptor["_id"])
                        if (img == tail):
                            id = len(desc_img_list) -1

            if not flag:
                label_Desc.append(query_desc)
                desc_img_list.append(tail)
                id = len(desc_img_list) -1

            svd = TruncatedSVD(k)
            svd_Obj = svd.fit(label_Desc)
            label_desc_transformed = svd_Obj.transform(label_Desc)


            dist = []

            for i, db_desc in enumerate(label_desc_transformed):
                if desc_img_list[i] == tail:
                    continue
                euc_dis = np.square(np.subtract(db_desc, label_desc_transformed[id]))
                match_score = np.sqrt(euc_dis.sum())
                dist.append(match_score)

            dist.sort()
            num = int(len(dist)/5)
            dist1 = dist[:num]
            ln = len(dist1)
            result[label] = sum(dist1)/ln

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

        vz.visualize_classified_image(tail, classification, dr_name, model_name,k)