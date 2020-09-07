import numpy as np
from sklearn.decomposition import NMF
import pymongo
import Constants as const
from sklearn.decomposition import PCA
import svm
from kernel import Kernel
import Visualizer as vz

client = pymongo.MongoClient('localhost', const.MONGODB_PORT)
imagedb = client["imagedb"]
mydb11k = imagedb["image_models"]
imagedb14 = client["imagedb14"]


class SVM:
    a = 0

    def preprocess_SVM(self, model, k, labelled_csv, unlablled_csv):
        a = 0
        model = model
        img_list = []
        pca = PCA(k)
        # model = "bag_" + model
        feature_desc = []
        img_list = []
        dorsal_imagelist = []
        palmar_imagelist = []
        imagedb1_4 = imagedb14[labelled_csv]
        for descriptor in imagedb1_4.find():
            if 'dorsal' in descriptor["aspectOfHand"]:
                dorsal_imagelist.append(descriptor["imageName"])
            else:
                palmar_imagelist.append(descriptor["imageName"])
        print(dorsal_imagelist)
        # for descriptor in imagedb.image_models.find():
        #     feature_desc.append(mydb.find({"_id": image})[0][model])
        #     feature_desc.append(descriptor[model])
        #     img_list.append(descriptor["_id"])

        for image in dorsal_imagelist:
            feature_desc.append(mydb11k.find({"_id": image})[0][model])
            img_list.append(image)
        for image in palmar_imagelist:
            feature_desc.append(mydb11k.find({"_id": image})[0][model])
            img_list.append(image)

        feature_desc_transformed = pca.fit_transform(feature_desc)
        # print(feature_desc_transformed)
        print(feature_desc_transformed)
        # print(feature_desc_transformed.)
        labels = SVM.labelData(self, img_list, labelled_csv)
        image_label_dict = SVM.classifySVM(self, feature_desc_transformed, labels, model, k, unlablled_csv)
        return image_label_dict

    def labelData(self, img_list, labelled_csv):
        a = 0
        labels = []
        imagedb1_4 = imagedb14[labelled_csv]
        print(labelled_csv)
        for image in img_list:
            print(image)
            if 'dorsal' in imagedb1_4.find({'imageName': image})[0]['aspectOfHand']:
                labels.append(0)
            else:
                labels.append(1)
        print(labels)
        # print(img_list)
        return labels

    def classifySVM(self, trainingData, labels, model, k, unlablled_csv):
        a = 0
        # classifer = svm.binary_classification_smo(kernel=Kernel.linear())
        classifier = svm.binary_classification_smo(
            kernel=Kernel._polykernel(5))  # try 5 and 10 for dimensions in polykernel
        # out_images = np.array(labels)
        for index, item in enumerate(labels):
            if item == 0:
                labels[index] = -1
        print(labels)
        labels_training = np.asarray(labels)
        # label_rep = np.where(labels <= 0, -1, 1)
        classifier.fit(trainingData, labels_training)

        # calculate transformed features of unlablled data
        imagedb1_4 = imagedb14[unlablled_csv]
        feature_descriptorUn = []
        unlabelled_imgList = []
        for descriptor in imagedb1_4.find():
            unlabelled_imgList.append(descriptor["imageName"])
        print(unlabelled_imgList)
        for image in unlabelled_imgList:
            feature_descriptorUn.append(mydb11k.find({"_id": image})[0][model])
        print(feature_descriptorUn)
        pca = PCA(k)
        feature_descriptorUnlabelled = np.asarray(feature_descriptorUn)
        testingData = pca.fit_transform(feature_descriptorUnlabelled)
        y = classifier.predict(testingData)
        print(y)

        svm_accuracy = SVM.accuracy(self, y, unlabelled_imgList)
        dict = {}
        for index, item in enumerate(unlabelled_imgList):
            if y[index] == -1:
                dict[item] = 'dorsal'
            else:
                dict[item] = 'palmar'
        print(dict)
        return dict, svm_accuracy

    def accuracy(self, pred_labels, unlabelled_imgList):
        a = 0
        metainfo = imagedb["HandInfo"]
        true_labels = []
        for image in unlabelled_imgList:
            a = 0
            if 'dorsal' in metainfo.find({"imageName": image})[0]["aspectOfHand"]:
                true_labels.append(-1)
            else:
                true_labels.append(1)

        num_correct = 0
        for i in range(len(pred_labels)):
            if pred_labels[i] == true_labels[i]:
                num_correct += 1
        print(num_correct / len(true_labels))
        return (num_correct / len(true_labels)) * 100


# if __name__ == '__main__':
#     s1 = SVM()
#     results = s1.preprocess_SVM("HOG", 20, "labelled_set2", "unlabelled_set2")
#     vz.visualize_labelled_images(results[0], 0, 'SVM', 0, results[1])
