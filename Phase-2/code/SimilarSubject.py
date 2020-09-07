from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pandas as pd
import pymongo
from sklearn.decomposition import NMF
import Visualizer as vz
import Constants as const
import os

client = pymongo.MongoClient('localhost', const.MONGODB_PORT)
imagedb = client["imagedb"]
mydb = imagedb["image_models"]


class Subject(object):

    def kl(self, p, q):
        """Kullback-Leibler divergence D(P || Q) for discrete distributions
        Parameters
        ----------
        p, q : array-like, dtype=float, shape=n
        Discrete probability distributions.
        """
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)

        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    def similar3Subjects(self, model, k, subjectId):
        model_name = model
        model = "bag_" + model
        subject_dict = {}
        dorsal_left_Desc= []
        dorsal_right_Desc = []
        palmar_left_Desc = []
        palmar_right_Desc = []

        for subject in imagedb.subjects.find():
            subject_dict[subject["_id"]] = [[] for i in range(4)]
            for img in subject["dorsal_left"]:
                dorsal_left_Desc.append(imagedb.image_models.find({"_id": img})[0][model])
                subject_dict[subject["_id"]][0].append(len(dorsal_left_Desc) - 1)

            for img in subject["dorsal_right"]:
                dorsal_right_Desc.append(imagedb.image_models.find({"_id": img})[0][model])
                subject_dict[subject["_id"]][1].append(len(dorsal_right_Desc) - 1)

            for img in subject["palmar_left"]:
                palmar_left_Desc.append(imagedb.image_models.find({"_id": img})[0][model])
                subject_dict[subject["_id"]][2].append(len(palmar_left_Desc) - 1)

            for img in subject["palmar_right"]:
                palmar_right_Desc.append(imagedb.image_models.find({"_id": img})[0][model])
                subject_dict[subject["_id"]][3].append(len(palmar_right_Desc) - 1)

        lda_dl = LatentDirichletAllocation(k, max_iter=25)
        lda_dr = LatentDirichletAllocation(k, max_iter=25)
        lda_pl = LatentDirichletAllocation(k, max_iter=25)
        lda_pr = LatentDirichletAllocation(k, max_iter=25)

        dorsal_left_Desc_transformed = lda_dl.fit_transform(dorsal_left_Desc)
        dorsal_right_Desc_transformed = lda_dr.fit_transform(dorsal_right_Desc)
        palmar_left_Desc_transformed = lda_pl.fit_transform(palmar_left_Desc)
        palmar_right_Desc_transformed = lda_pr.fit_transform(palmar_right_Desc)

        score_dict = {}
        for key in subject_dict:
            if key == subjectId:
                continue
            mx = 0
            params = 0
            if len(subject_dict[subjectId][0]) > 0 and len(subject_dict[key][0]) > 0:
                params += 1
                minKL = 10000000000
                minKLCM = 10000000000
                cnt = 0
                mxb=0
                for i in subject_dict[subjectId][0]:
                    for j in subject_dict[key][0]:
                        minKL = min(minKL, self.kl(dorsal_left_Desc_transformed[i], dorsal_left_Desc_transformed[j]))
                    cnt+=1
                    mxb+=minKL
                minKL = mxb/cnt
                mx += minKL

            if len(subject_dict[subjectId][1]) > 0 and len(subject_dict[key][1]) > 0:
                params += 1
                minKL = 10000000000
                cnt = 0
                mxb = 0
                for i in subject_dict[subjectId][1]:
                    for j in subject_dict[key][1]:
                        minKL = min(minKL, self.kl(dorsal_right_Desc_transformed[i], dorsal_right_Desc_transformed[j]))
                    cnt += 1
                    mxb += minKL
                minKL = mxb / cnt
                mx += minKL

            if len(subject_dict[subjectId][2]) > 0 and len(subject_dict[key][2]) > 0:
                params += 1
                minKL = 10000000000
                cnt = 0
                mxb = 0
                for i in subject_dict[subjectId][2]:
                    for j in subject_dict[key][2]:
                        minKL = min(minKL, self.kl(palmar_left_Desc_transformed[i], palmar_left_Desc_transformed[j]))
                    cnt += 1
                    mxb += minKL
                minKL = mxb / cnt
                mx += minKL

            if len(subject_dict[subjectId][3]) > 0 and len(subject_dict[key][3]) > 0:
                params += 1
                minKL = 10000000000
                cnt = 0
                mxb = 0
                for i in subject_dict[subjectId][3]:
                    for j in subject_dict[key][3]:
                        minKL = min(minKL, self.kl(palmar_right_Desc_transformed[i], palmar_right_Desc_transformed[j]))
                    cnt += 1
                    mxb += minKL
                minKL = mxb / cnt
                mx += minKL

            if params > 0:
                score_dict[key] = mx/params

        z = 0
        vz.visualize_similar_subjects(subjectId, score_dict, k, model_name)
        for key, value in sorted(score_dict.items(), key=lambda item: item[1]):
            if z < 3:
                print(str(key) + " has matching score:: " + str(value))
                z += 1
            else:
                break

    def similarSubjectsAssignment(self, model, k, subjectId, simMat, sub_list):

        model = "bag_" + model
        subject_dict = {}
        dorsal_left_Desc= []
        dorsal_right_Desc = []
        palmar_left_Desc = []
        palmar_right_Desc = []

        for subject in imagedb.subjects.find():
            subject_dict[subject["_id"]] = [[] for i in range(4)]
            for img in subject["dorsal_left"]:
                dorsal_left_Desc.append(imagedb.image_models.find({"_id": img})[0][model])
                subject_dict[subject["_id"]][0].append(len(dorsal_left_Desc) - 1)
            for img in subject["dorsal_right"]:
                dorsal_right_Desc.append(imagedb.image_models.find({"_id": img})[0][model])
                subject_dict[subject["_id"]][1].append(len(dorsal_right_Desc) - 1)
            for img in subject["palmar_left"]:
                palmar_left_Desc.append(imagedb.image_models.find({"_id": img})[0][model])
                subject_dict[subject["_id"]][2].append(len(palmar_left_Desc) - 1)
            for img in subject["palmar_right"]:
                palmar_right_Desc.append(imagedb.image_models.find({"_id": img})[0][model])
                subject_dict[subject["_id"]][3].append(len(palmar_right_Desc) - 1)

        lda_dl = LatentDirichletAllocation(k, max_iter=25)
        lda_dr = LatentDirichletAllocation(k, max_iter=25)
        lda_pl = LatentDirichletAllocation(k, max_iter=25)
        lda_pr = LatentDirichletAllocation(k, max_iter=25)

        dorsal_left_Desc_transformed = lda_dl.fit_transform(dorsal_left_Desc)
        dorsal_right_Desc_transformed = lda_dr.fit_transform(dorsal_right_Desc)
        palmar_left_Desc_transformed = lda_pl.fit_transform(palmar_left_Desc)
        palmar_right_Desc_transformed = lda_pr.fit_transform(palmar_right_Desc)

        score_dict = {}
        for key in subject_dict:
            if key == subjectId:
                simMat[sub_list.index(subjectId)][sub_list.index(key)] = 0
                continue
            mx = 0
            params = 0
            if len(subject_dict[subjectId][0]) > 0 and len(subject_dict[key][0]) > 0:
                params += 1
                minKL = 10000000000
                for i in subject_dict[subjectId][0]:
                    for j in subject_dict[key][0]:
                        minKL = min(minKL, self.kl(dorsal_left_Desc_transformed[i], dorsal_left_Desc_transformed[j]))
                mx += minKL

            if len(subject_dict[subjectId][1]) > 0 and len(subject_dict[key][1]) > 0:
                params += 1
                minKL = 10000000000
                for i in subject_dict[subjectId][1]:
                    for j in subject_dict[key][1]:
                        minKL = min(minKL, self.kl(dorsal_right_Desc_transformed[i], dorsal_right_Desc_transformed[j]))
                mx += minKL

            if len(subject_dict[subjectId][2]) > 0 and len(subject_dict[key][2]) > 0:
                params += 1
                minKL = 10000000000
                for i in subject_dict[subjectId][2]:
                    for j in subject_dict[key][2]:
                        minKL = min(minKL, self.kl(palmar_left_Desc_transformed[i], palmar_left_Desc_transformed[j]))
                mx += minKL

            if len(subject_dict[subjectId][3]) > 0 and len(subject_dict[key][3]) > 0:
                params += 1
                minKL = 10000000000
                for i in subject_dict[subjectId][3]:
                    for j in subject_dict[key][3]:
                        minKL = min(minKL, self.kl(palmar_right_Desc_transformed[i], palmar_right_Desc_transformed[j]))
                mx += minKL

            if params > 0:
                simMat[sub_list.index(subjectId)][sub_list.index(key)] = abs(mx/params)

    def rescaleToBasis(self, arr):
        np.seterr(divide='ignore', invalid='ignore')
        row_magnitude = np.sqrt(np.sum(np.square(arr), axis=1))
        rescaled_array = np.divide(arr, row_magnitude[:, None])
        return rescaled_array

    def createSSMatrix(self, k):
        num_ls = k
        sub_list = []
        sub_count = imagedb.subjects.find().count()
        #print(sub_count)
        simMat = [[0.0 for x in range(sub_count)] for y in range(sub_count)]
        for subject in imagedb.subjects.find():
            sub_list.append(subject["_id"])

        for subject in sub_list:
            self.similarSubjectsAssignment("SIFT", k, subject, simMat, sub_list)

        #print(simMat)
        df = pd.DataFrame(simMat)
        csv_path = os.path.join("..", "csv", "Sub2SubMatrix.csv")
        df.to_csv(csv_path, index=False, header=False)
        nm = NMF(k)
        W = nm.fit_transform(simMat)
        Mat = self.rescaleToBasis(W)

        visualizeArr = []

        for i in range(k):
            col = Mat[:, i]
            arr = []
            for k, val in enumerate(col):
                arr.append((str(sub_list[k]), val))
            arr.sort(key=lambda x: x[1], reverse=True)
            print("Printing term-weight pair for latent Semantic {}:".format(i + 1))
            print(arr)
            """ Append the sorted array for each latent semantic """
            visualizeArr.append(arr[:30])

        visualizeArr = pd.DataFrame(visualizeArr)
        vz.visualize_ss_matrix(visualizeArr, num_ls)
