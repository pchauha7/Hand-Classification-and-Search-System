import math
from collections import defaultdict
from functools import reduce
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import Constants as const
import Visualizer as vz
import os

from LocalBinaryPatterns import LBP
from ColorMoments import CM
from SIFT import SIFT
from HOGmain import HOG

import numpy as np
import pandas as pd

IMAGE_ID_COL = 'ImageId'

import pymongo

client = pymongo.MongoClient('localhost', const.MONGODB_PORT)
imagedb = client["imagedb"]
mydb = imagedb["image_models"]


class LSH:

    def __init__(self, hash_obj, num_layers, num_hash, vec, b, w):
        self.hash_obj = hash_obj
        self.num_layers = num_layers
        self.num_hash = num_hash
        self.vec = vec
        self.b = b
        self.w = w

    def create_hash_table(self, img_vecs, verbose=False):
        hash_table = self.init_hash_table()
        nrows, ncols = img_vecs.shape[0], img_vecs.shape[1]
        for i in range(nrows):
            img_id, img_vec = img_vecs[i][0], np.array(img_vecs[i][1:])
            for idx, hash_vec in enumerate(hash_table):
                buckets = self.hash_obj.hash(img_vec, self.vec[idx], self.b[idx], self.w)
                for i in range(len(buckets)):
                    hash_vec[i][buckets[i]].add(img_id)

        return hash_table

    def init_hash_table(self):
        hash_table = []
        for i in range(self.num_layers):
            hash_layer = []
            for j in range(self.num_hash):
                hash_vec = defaultdict(set)
                hash_layer.append(hash_vec)
            hash_table.append(hash_layer)
        return hash_table

    def find_ann(self, query_point, hash_table, k):
        candidate_imgs = set()
        num_conjunctions = self.num_hash
        count_overall = 0
        for layer_idx, layer in enumerate(self.vec):
            hash_vec = hash_table[layer_idx]
            buckets = self.hash_obj.hash(query_point, layer, self.b[layer_idx], self.w)
            cand = hash_vec[0][buckets[0]].copy()
            count_overall = len(cand)
            for ix, idx in enumerate(buckets[1:num_conjunctions]):
                count_overall = count_overall + len(hash_vec[ix + 1][idx])
                cand = cand.intersection(hash_vec[ix + 1][idx])
            candidate_imgs = candidate_imgs.union(cand)
            # print("---------------  Candidate Images  ------------------")
            # print(candidate_imgs)
            if len(candidate_imgs) > 4 * k:
                print(f'Early stopping at layer {layer_idx} found {len(candidate_imgs)}')
                break
        if len(candidate_imgs) < k:
            if num_conjunctions > 1:
                self.num_hash -= 1
                # print(self.num_hash)
                # print('Reduced number of hashes')
                return self.find_ann(query_point, hash_table, k=k)
            else:
                print('Cannot reduce number of hashes')
        print("Overall Images: ", count_overall)
        print("Count of Unique Images: ", len(candidate_imgs))
        return candidate_imgs

    def post_process_filter(self, query_point, candidates, k):
        distances = [{IMAGE_ID_COL: row[0],
                      'distance': self.hash_obj.dist(query_point, row.drop(0))}
                     for idx, row in candidates.iterrows()]
        return sorted(distances, key=lambda x: x['distance'])[:k]


class l2DistHash:

    def hash(self, point, vec, b, w):
        val = np.dot(vec, point) + b
        val = val * 100
        res = np.floor_divide(val, w)
        return res

    def dist(self, point1, point2):
        v = (point1 - point2) ** 2
        return math.sqrt(sum(v))


final_desc = []
img_df = None


def run_lsh(input_vec, num_layers, num_hash):
    w = 300
    dim = 257
    vec = np.random.rand(num_layers, num_hash, dim - 1)
    b = np.random.randint(low=0, high=w, size=(num_layers, num_hash))
    l2_dist_obj = l2DistHash()
    lsh = LSH(hash_obj=l2_dist_obj, num_layers=num_layers, num_hash=num_hash, vec=vec, b=b, w=w)
    hashTable = lsh.create_hash_table(input_vec, verbose=False)
    return hashTable


def getFeature(query_path, model_name):
    if model_name == "CM":
        fd = CM(query_path)
    elif model_name == "HOG":
        fd = HOG(query_path)
    elif model_name == "SIFT":
        fd = SIFT(query_path)
    elif model_name == "LBP":
        fd = LBP(query_path)
    else:
        print("Error! Invalid Model Name")
        return []
    lst = fd.getFeatureDescriptors()
    return lst


def img_ann(img_df, query, k, num_layers, num_hash, layer_file_name=None):
    image_ids = img_df.iloc[:, 0]
    w = 50
    dim = img_df.shape[1]
    vec = np.random.rand(num_layers, num_hash, dim - 1)
    b = np.random.randint(low=0, high=w, size=(num_layers, num_hash))
    l2_dist_obj = l2DistHash()
    lsh = LSH(hash_obj=l2_dist_obj, num_layers=num_layers, num_hash=num_hash, vec=vec, b=b, w=w)
    # LSH index structure
    hash_table = lsh.create_hash_table(img_df.values)
    dump_file = os.path.join("..", "csv", "hashtable.csv")
    df = pd.DataFrame(hash_table)
    df.to_csv(dump_file, index=False, header=False)
    query_vec = img_df.loc[img_df[0] == query]
    query_vec = query_vec.iloc[:, 1:].values[0]

    candidate_ids = lsh.find_ann(query_point=query_vec, hash_table=hash_table, k=k)
    candidate_vecs = img_df.loc[img_df[0].isin(candidate_ids)]
    # print(len(candidate_ids))

    if not candidate_ids:
        return None
    dist_res = lsh.post_process_filter(query_point=query_vec, candidates=candidate_vecs, k=k)
    return dist_res


################################### MAIN FUNCTION #####################################

# imgdf = None
# k = 0

# n = imagedb.image_models.count()


# final_desc = []
# for descriptor in imagedb.image_models.find().limit(1000):
#     hog_desc = descriptor['HOG']
#     hog_concat_desc = [i for cm in hog_desc for i in cm]
#     desc = [descriptor['_id']] + hog_concat_desc
#     final_desc.append(desc)
# imgdf = pd.DataFrame(final_desc)
# imgdf.to_csv("output.csv", index = False, header = False)


# for i in range(1, 11):
#     final_desc = []
#     for descriptor in imagedb.image_models.find().skip(i*1000).limit(1000):
#         hog_desc = descriptor['HOG']
#         hog_concat_desc = [i for cm in hog_desc for i in cm]
#         desc = [descriptor['_id']] + hog_concat_desc
#         final_desc.append(desc)
#     imgdf = pd.DataFrame(final_desc)
#     with open("output.csv", 'a', newline = '') as f:
#         imgdf.to_csv(f, index = False, header = False)

# for descriptor in imagedb.image_models.find().skip(11000):
#         hog_desc = descriptor['HOG']
#         hog_concat_desc = [i for cm in hog_desc for i in cm]
#         desc = [descriptor['_id']] + hog_concat_desc
#         final_desc.append(desc)
# imgdf = pd.DataFrame(final_desc)
# with open("output.csv", 'a', newline = '') as f:
#     imgdf.to_csv(f, index = False, header = False)


# img_df = pd.read_csv("output.csv", nrows = 1000, header = None)
# print(img_df)

# pca = PCA(256)
# feature_desc_fit_transformed = pd.DataFrame(pca.fit_transform(np.array(img_df.iloc[:, 1:])))
# for i in range(1, 11):
#     print(i)
#     img_df = pd.read_csv("output.csv", skiprows=1000*i, nrows=1000, header = None)
#     new_feature_desc = pd.DataFrame(pca.transform(np.array(img_df.iloc[:, 1:])))
#     feature_desc_fit_transformed = feature_desc_fit_transformed.append(new_feature_desc, ignore_index = True)

# img_df = pd.read_csv("output.csv", skiprows=12000, header = None)
# new_feature_desc = pd.DataFrame(pca.transform(np.array(img_df.iloc[:, 1:])))
# feature_desc_fit_transformed = feature_desc_fit_transformed.append(new_feature_desc, ignore_index = True)

# feature_desc_fit_transformed.to_csv("output_pca.csv", index = False, header = False)

# feature_df = pd.read_csv("output_pca.csv", header = None)
# print(feature_df.shape)

# image_id_list = []
# for descriptor in imagedb.image_models.find():
#     image_id_list.append(descriptor['_id'])

# print(len(image_id_list))

# feature_df.insert(loc = 0, column = None, value = image_id_list)
# print(feature_df.shape)
# feature_df.to_csv("output_pca_final.csv", index = False, header = False)

csv_path = os.path.join('..', 'csv', 'output_pca_final.csv')
img_df = pd.read_csv(csv_path, header=None)
# print(img_df.shape)

# Task 5 Query 1
result = img_ann(img_df, 'Hand_0000674.jpg', 20, 10, 10)
vz.visualize_relevance_feedback('Hand_0000674.jpg', result, 20, 10, 10)
# Task 5 Query 2
# result = img_ann(img_df, 'Hand_0000674.jpg', 20, 10, 13)
# vz.visualize_relevance_feedback('Hand_0000674.jpg', result, 20, 10, 13)
# Task 5 Query 3
# result = img_ann(img_df, 'Hand_0000674.jpg', 20, 5, 10)
# vz.visualize_relevance_feedback('Hand_0000674.jpg', result, 20, 5, 10)
print(len(result))
print(result)
