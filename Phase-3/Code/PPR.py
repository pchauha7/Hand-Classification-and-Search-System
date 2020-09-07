import Constants as const
import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
# from NonNegativeMatrix import NM_F
# import Visualizer as vz
import pandas as pd
import os
import pymongo

client = pymongo.MongoClient('localhost', const.MONGODB_PORT)
imagedb = client["imagedb"]
imagedb14 = client["imagedb14"]
dr_name = 'NMF'

class PersonalizedPageRank(object):

    def calculateDistanceMatrix(self, transformed_matrix):
        sim_matrix = [[0 for j in range(len(transformed_matrix))] for i in range(len(transformed_matrix))]

        for i in range(len(transformed_matrix)):
            for j in range(len(transformed_matrix)):
                if i == j:
                    continue
                euc_dis = np.square(np.subtract(transformed_matrix[i], transformed_matrix[j]))
                match_score = np.sqrt(euc_dis.sum(0))
                sim_matrix[i][j] = abs(match_score)

        return sim_matrix

    # def calculateImageSimilarityGraph(self, k):
    #     feature_desc = []
    #     img_list = []
    #     for descriptor in imagedb14.image_models.find():
    #         feature_desc.append(descriptor["bag_LBP"])
    #         img_list.append(descriptor["_id"])
    #     nmf_ = NMF(n_components=30)
    #     W = nmf_.fit_transform(feature_desc)
    #
    #     dist_matrix = self.calculateDistanceMatrix(W)
    #
    #     sim_graph = [[-1 for j in range(len(dist_matrix))] for i in range(len(dist_matrix))]
    #
    #     for idx,row in enumerate(dist_matrix):
    #         new_row = []
    #         for i, val in enumerate(row):
    #             new_row.append([val, i])
    #         new_row.sort(key=lambda x:x[0])
    #         for item in new_row[1:k+1]:
    #             sim_graph[idx][item[1]] = item[0]
    #
    #     return img_list, sim_graph

    def calculateImageSimilarityGraph(self, k, csv1, csv2=""):
        feature_desc = []
        img_list = []
        csv_db1 = imagedb14[csv1]
        for row in csv_db1.find():
            feature_desc.append(imagedb14.image_models.find({"_id": row['imageName']})[0]["bag_SIFT"])
            # feature_desc.append(descriptor["HOG"])
            img_list.append(row['imageName'])
        if csv2 != "":
            csv_db2 = imagedb14[csv2]
            for row in csv_db2.find():
                feature_desc.append(imagedb14.image_models.find({"_id": row['imageName']})[0]["bag_SIFT"])
                img_list.append(row['imageName'])

        # nmf_ = NMF(n_components=30)
        # W = nmf_.fit_transform(feature_desc)

        # pca = PCA(55)
        # feature_desc_transformed = pca.fit_transform(feature_desc)

        dist_matrix = self.calculateDistanceMatrix(feature_desc)

        sim_graph = [[-1 for j in range(len(dist_matrix))] for i in range(len(dist_matrix))]

        for idx,row in enumerate(dist_matrix):
            new_row = []
            for i, val in enumerate(row):
                new_row.append([val, i])
            new_row.sort(key=lambda x:x[0])
            for item in new_row[1:k+1]:
                sim_graph[idx][item[1]] = item[0]

        return img_list, sim_graph

    def calculateImageSimilarityGraphOnImageList(self, img_list, k):
        feature_desc = []
        #img_list = []
        for img in img_list:
            feature_desc.append(imagedb.image_models.find({"_id": img})[0]["HOG"])
        # nmf_ = NMF(n_components=30)
        # W = nmf_.fit_transform(feature_desc)

        # print('Feature Descriptor')
        # print(feature_desc)
        # pca = PCA(55)
        # feature_desc_transformed = pca.fit_transform(feature_desc)

        dist_matrix = self.calculateDistanceMatrix(feature_desc)

        sim_graph = [[-1 for j in range(len(dist_matrix))] for i in range(len(dist_matrix))]

        for idx,row in enumerate(dist_matrix):
            new_row = []
            for i, val in enumerate(row):
                new_row.append([val, i])
            new_row.sort(key=lambda x:x[0])
            for item in new_row[1:k+1]:
                sim_graph[idx][item[1]] = item[0]

        return sim_graph

    def getPersonalizedPageRank(self, sim_graph, k, seed_values, tolerance=1.0e-5):
        np_graph = np.array(sim_graph)
        # print(np_graph.shape)
        graph_transpose = np_graph.transpose()
        # print(graph_transpose.shape)
        damping_factor = 0.85
        error = 1

        page_rank_current = np.array([0.0 for i in range(len(sim_graph))])
        initial_page_rank = np.array([0.0 for i in range(len(sim_graph))])
        self.set_seed_values(page_rank_current, seed_values, damping_factor)
        self.set_seed_values(initial_page_rank, seed_values, damping_factor)

        while error > tolerance:
            for index, row in enumerate(graph_transpose):
                edge_indexes = np.nonzero(row > -1)[0]
                # edge_indexes.toList()
                for edge_index in edge_indexes:
                    page_rank_current[index] += (initial_page_rank[edge_index] * damping_factor) / k

            error = np.linalg.norm(page_rank_current - initial_page_rank, 2)
            initial_page_rank = page_rank_current
            page_rank_current = np.array([0.0 for i in range(len(sim_graph))])
            self.set_seed_values(page_rank_current, seed_values, damping_factor)

        return initial_page_rank

    def set_seed_values(self, rank_array, seed_values, damping_factor):
        for index in seed_values:
            rank_array[index] = (1 - damping_factor)/3

    def getKDominantImagesUsingPPR(self, k, K, seedList, csv1, csv2=""):
        img_list, sim_graph = self.calculateImageSimilarityGraph(k, csv1, csv2)
        df = pd.DataFrame(sim_graph)
        csv_path = os.path.join("..", "csv", "pprGraph.csv")
        df.to_csv(csv_path, index=False, header=False)
        seed_values = []
        for image in seedList:
            seed_values.append(img_list.index(image))
        page_rank = self.getPersonalizedPageRank(sim_graph, k, seed_values)

        rank_dict = {}
        for i, image in enumerate(img_list):
            rank_dict[image] = page_rank[i]

        count = 0
        print("\n\nNow printing top {} matched Images and their matching scores".format(K))
        # sorted_dict = sorted(rank_dict.items(), key=lambda item: item[1])
        # vz.visualize_ppr_images(seedList, rank_dict, k, K, dr_name)
        for key, value in sorted(rank_dict.items(), key=lambda item: item[1], reverse=True):
            if count < K:
                print(key + " has matching score:: " + str(value))
                # shutil.copy(os.path.join(head, key), res_dir)
                count += 1
            else:
                break


    def ppr_classification(self, sim_graph, labels, k):
        page_rank_scores_for_label = {}
        for label in labels:
            page_rank_for_label = self.getPersonalizedPageRank(sim_graph, k, labels[label],tolerance=1.0e-5)
            page_rank_scores_for_label[label] = page_rank_for_label

        return page_rank_scores_for_label

    def classifyUnlabelledImagesUsingPPR(self, k, csv1, csv2):
        img_list, sim_graph = self.calculateImageSimilarityGraph(k, csv1, csv2)
        img_dict = {}
        for index, img in enumerate(img_list):
            img_dict[img] = index
        labels = {}
        labels['dorsal'] = []
        labels['palmar'] = []
        csv_db1 = imagedb14[csv1]

        for row in csv_db1.find():
            if 'dorsal' in row['aspectOfHand'] and row['imageName'] in img_dict:
                labels['dorsal'].append(img_dict[row['imageName']])
            if 'palmar' in row['aspectOfHand'] and row['imageName'] in img_dict:
                labels['palmar'].append(img_dict[row['imageName']])

        page_rank_scores_for_label = self.ppr_classification(sim_graph, labels, k)

        classification_result = {}
        csv_db2 = imagedb14[csv2]
        for row in csv_db2.find():
            if row['imageName'] in img_dict:
                if page_rank_scores_for_label['dorsal'][img_dict[row['imageName']]] > page_rank_scores_for_label['palmar'][img_dict[row['imageName']]]:
                    classification_result[row['imageName']] = 'dorsal'
                else:
                    classification_result[row['imageName']] = 'palmar'

        count = 0
        den = len(classification_result)
        for img in classification_result:
            try:
                if (classification_result[img] in imagedb14.HandInfo.find({"imageName": img})[0]['aspectOfHand']):
                    count += 1
            except:
                den -= 1

        print('could not find {} images in 11k Image set'.format(len(classification_result) - den))
        print('Number of items in dictionary: %d' % len(classification_result.items()))
        if den != 0:
            successRatio = (count / den) * 100
            print(successRatio)
        # classification_result = reversed(sorted(classification_result.keys()))
        # vz.visualize_labelled_images(classification_result, 0, 'PPR Based', 0, successRatio)
        return classification_result



    def relevanceFeedbackPPR(self, image_list, labels):
        print(labels)
        k = max(5, len(image_list)//15)
        sim_graph = self.calculateImageSimilarityGraphOnImageList(image_list, k)
        page_rank_scores_for_label = self.ppr_classification(sim_graph, labels, k)

        print(page_rank_scores_for_label)

        score_label = [(page_rank_scores_for_label["Relevant"][i] - page_rank_scores_for_label["Irrelevant"][i]) for i in range(len(page_rank_scores_for_label["Relevant"]))]
        print(score_label)
        rank_dict = {}
        for i, image in enumerate(image_list):
            rank_dict[image] = score_label[i]

        ans_list = []
        for key, value in sorted(rank_dict.items(), key=lambda item: item[1], reverse=True):
            print(key + " has matching score:: " + str(value))
            ans_list.append(key)

        return ans_list


# Uncomment below lines to run PPR cases

x = PersonalizedPageRank()
# x.getKDominantImagesUsingPPR(5, 10, ['Hand_0008333.jpg', 'Hand_0006183.jpg', 'Hand_0000074.jpg'], 'labelled_set2')
# x.getKDominantImagesUsingPPR(5, 10, ['Hand_0003457.jpg', 'Hand_0000074.jpg', 'Hand_0005661.jpg'], 'labelled_set2')

#
x.classifyUnlabelledImagesUsingPPR(5, 'labelled_set2', 'unlabelled_set2')
# vz.visualize_labelled_images(results[0], 0, 'PPR', 0, results[1])
# print()


