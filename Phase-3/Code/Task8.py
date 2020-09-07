import glob
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import Visualizer as vz

class Task8(object):

    def metadataRead(self,filepath, k):
        metadataFile = pd.read_csv(filepath, sep=',', header=None)
        metadataFile = metadataFile.iloc[1:]
        # metadataFile.columns = ['id', 'age', 'gender', 'skinColor', 'accessories', 'nailPolish', 'aspectOfHand',
        #                         'imageName', 'irregularities']
        metadataFile.columns = ['id', 'age', 'gender', 'skinColor', 'accessories', 'nailPolish', 'aspectOfHand',
                                'Orientation', 'imageName']

        metadataFile = metadataFile[['imageName', 'aspectOfHand', 'Orientation', 'gender', 'accessories']]
        # print(metadataFile)
        self.convertToBinaryMatrix(metadataFile, k)


    def convertToBinaryMatrix(self,metaDataFrame, k):
        a = 0
        metaDataFrame['Orientation'] = metaDataFrame['Orientation'].replace(['left', 'right'], [0, 1])
        metaDataFrame['aspectOfHand'] = metaDataFrame['aspectOfHand'].replace(['dorsal', 'palmar'], [0, 1])
        metaDataFrame['gender'] = metaDataFrame['gender'].replace(['male', 'female'], [0, 1])
        imageList = metaDataFrame['imageName'].tolist()
        metaDataFrame = metaDataFrame[['aspectOfHand', 'Orientation', 'gender', 'accessories']]
        metaDataFrame['accessories'] = metaDataFrame['accessories'].astype(int)
        featureList = metaDataFrame.columns
        self.performNMF(metaDataFrame, k, imageList, featureList)


    def performNMF(self,metaDataFrame, k, imageList, featureList):
        a = 0
        b = 0
        metaDataMatrix = metaDataFrame.to_numpy()
        nmf_ = NMF(n_components=k, init='random', random_state=0)
        W = nmf_.fit_transform(metaDataMatrix)
        H = nmf_.components_

        W = self.rescaleToBasis(W)
        print("Image space")
        print(W)
        print("Metadata Space")
        print(H)
        img_space = []
        print("Top {} latent semantics in image-space".format(k))
        for i in range(k):
            col = W[:, i]
            arr = []
            for j, val in enumerate(col):
                arr.append((str(imageList[j]), val))
            arr.sort(key=lambda x: x[1], reverse=True)
            print("Printing latent Semantic {} in image-space:".format(i + 1))
            print(arr)
            img_space.append(arr[:50])
        img_space = pd.DataFrame(img_space)
        print(img_space.shape)
        vz.visualize_img_space(k, img_space)
        print("Top {} latent semantics in metadata-space".format(k))
        metadata_space = []
        for i in range(k):
            print(i)
            row = H[i]
            arr = []
            for j, val in enumerate(row):
                arr.append((str(featureList[j]), val))
            arr.sort(key=lambda x: x[1], reverse=True)
            print("Printing latent Semantic {} in metadata-space:".format(i + 1))
            print(arr)
            metadata_space.append(arr)
        metadata_space = pd.DataFrame(metadata_space)
        vz.visualize_metadata_space(k, metadata_space)


    def rescaleToBasis(self,arr):
        np.seterr(divide='ignore', invalid='ignore')
        row_magnitude = np.sqrt(np.sum(np.square(arr), axis=1))
        rescaled_array = np.divide(arr, row_magnitude[:, None])
        return rescaled_array

    def run_task_8(self,metadatacsvpath,k):
        self.metadataRead(metadatacsvpath, k)

# t8=Task8()
# t8.run_task_8("../csv/ImageMetadata.csv",4)
