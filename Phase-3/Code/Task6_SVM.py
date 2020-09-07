import pymongo
import Constants as const
from sklearn.decomposition import PCA
import numpy as np
import svm
from kernel import Kernel


client = pymongo.MongoClient('localhost', const.MONGODB_PORT)
imagedb = client["imagedb"]
mydb = imagedb["image_models"]
meta = imagedb["ImageMetadata"]


class SVMrel_feed:

    def relvancefeedbackSVM(self, image_list, relvance_dict):
        a = 0
        print(image_list)
        print(relvance_dict)
        model = "HOG"
        relevant_imagesList = []
        irrelevant_imagesList = []
        labels_training = []
        new_imageList = []
        mydb = imagedb["image_models"]
        _list = []
        print(image_list.__len__())
    #     Relevant and Irrelevant
    #     relevant_imagesList=relvance_dict["Relevant"]
    #     irrelevant_imagesList=relvance_dict["Irrelevant"]
        for index in relvance_dict["Relevant"]:
            new_imageList.append(image_list[index])
            labels_training.append(1)
            _list.append(index)
            relevant_imagesList.append(image_list[index])

        for index in relvance_dict["Irrelevant"]:
            new_imageList.append(image_list[index])
            labels_training.append(-1)
            _list.append(index)
            irrelevant_imagesList.append(image_list[index])

        # print('New Image List')
        # print(new_imageList)
        # print('Labels Training')
        # print(labels_training)
        # print('_list')
        # print(_list)

        feature_desc_train=[]
        for image in image_list:
            feature_desc_train.append(mydb.find({"_id": image})[0][model])
        feature_desc = np.asarray(feature_desc_train)
        print(feature_desc.shape)

        test_images = []
        k=20
        pca=PCA(k)
        feature_desc_all = pca.fit_transform(feature_desc)
        feature_desc_transformed=[]
        feature_desc_transformed_test=[]
        for index in range(0,20):
            if index in _list:
                feature_desc_transformed.append(feature_desc_all[index])
            else:
                feature_desc_transformed_test.append(feature_desc_all[index])
                test_images.append(image_list[index])

        feature_desc_transformed_training=np.asarray(feature_desc_transformed)
        feature_desc_transformed_testing=np.asarray(feature_desc_transformed_test)

        y = SVMrel_feed.classifySVM(self, feature_desc_transformed_training, labels_training,feature_desc_transformed_testing)
        # print('Test Images')
        # print(test_images)
        # print('Relevant ImageList')
        # print(relevant_imagesList)
        # print('Irrelevant ImageList')
        # print(irrelevant_imagesList)
        # print('y')
        # print(y)
        new_order=SVMrel_feed.return_func(self, relevant_imagesList,irrelevant_imagesList, test_images, y)
        print(new_order)
        print(new_order.__len__())
        return new_order

    def return_func(self, relevant_imagesList, irrelevant_imagesList, test_images, label_test):
        a=0
        newimage_list=[]
        for items in relevant_imagesList:
            newimage_list.append(items)

        index=0
        for index, item in enumerate(label_test):
            if item == 1:
                newimage_list.append(test_images[index])
        newimage_list.extend(irrelevant_imagesList)
        for index, item in enumerate(label_test):
            if item == -1:
                newimage_list.append(test_images[index])

        return newimage_list

    def classifySVM(self, trainingData, labels, testData):

        # classifer = svm.binary_classification_smo(kernel=Kernel.linear())
        classifier = svm.binary_classification_smo(kernel=Kernel._polykernel(5))  # try 5 and 10 for dimensions in polykernel
        # out_images = np.array(labels)
        for index, item in enumerate(labels):
            if item == 0:
                labels[index] = -1
        print(labels)
        labels_training = np.asarray(labels)
        # label_rep = np.where(labels <= 0, -1, 1)

        classifier.fit(trainingData, labels_training)

        # calculate transformed features of unlablled data
        # mydb = imagedb["image_models_unlabelled"]
        # pca = PCA(k)
        # testData_transformed = pca.fit_transform(testData)
        y = classifier.predict(testData)
        print(y)
        return y

#
# if __name__ == '__main__':
#     # t1 = SVMrel_feed()
#     # image_list = ['Hand_0000002.jpg','Hand_0000003.jpg','Hand_0000004.jpg','Hand_0000005.jpg','Hand_0000006.jpg','Hand_0000007.jpg','Hand_0000008.jpg','Hand_0000009.jpg','Hand_0000010.jpg','Hand_0000011.jpg','Hand_0000012.jpg','Hand_0000013.jpg','Hand_0000014.jpg','Hand_0000015.jpg','Hand_0000016.jpg','Hand_0000020.jpg','Hand_0000021.jpg','Hand_0000022.jpg','Hand_0000023.jpg','Hand_0000024.jpg']
#     # # t1.classify_DP(r"C:\Users\shadab\Documents\MWDB\project\phase3\phase3_sample_data\Unlabelled\Set3","LBP",20)
#     # rel_dict = {}
#     # rel_dict["Relevant"] = [0,1,2,3,4,5,8]
#     # rel_dict["Irrelevant"] = [10,11,12,13,14,15,19]
#     # t1.relvancefeedbackSVM(image_list,rel_dict)

#     relvance :
#
