from matplotlib import pyplot
import numpy as np
import pymongo
import Constants as const
import feature_bow_extractor as fbe
from sklearn.decomposition import PCA
import Visualizer as vz

client = pymongo.MongoClient('localhost', const.MONGODB_PORT)
imagedb = client["imagedb"]
mydb = imagedb["image_models"]
imagedb14 = client["imagedb14"]


class Task1:

    def classify_DP(self, model, k, labelled_csv, unlablled_csv):

        model = "bag_" + model
        dorsal_imagelist = []
        palmar_imagelist = []
        imagedb1_4 = imagedb14[labelled_csv]
        for descriptor in imagedb1_4.find():
            if 'dorsal' in descriptor["aspectOfHand"]:
                dorsal_imagelist.append(descriptor["imageName"])
            else:
                palmar_imagelist.append(descriptor["imageName"])

        imagedb1_4_bow = imagedb14["label_unlabel_bow"]
        feature_desc_d = []
        feature_desc_p = []
        print(dorsal_imagelist)
        for image in dorsal_imagelist:
            feature_desc_d.append(imagedb1_4_bow.find({"_id": image})[0][model])

        for image in palmar_imagelist:
            feature_desc_p.append(imagedb1_4_bow.find({"_id": image})[0][model])

        feature_desc_dorsal = np.asarray(feature_desc_d)
        feature_desc_palmar = np.asarray(feature_desc_p)
        print(feature_desc_dorsal.shape)

        pca = PCA(k)
        Ud = pca.fit_transform(feature_desc_dorsal)
        Up = pca.fit_transform(feature_desc_palmar)

        imagedb1_4 = imagedb14[unlablled_csv]
        unlabelled_imagelist = []
        for descriptor in imagedb1_4.find():
            unlabelled_imagelist.append(descriptor["imageName"])
        unlabelled_image_feature_descriptor = []

        for unlabelled_images in unlabelled_imagelist:
            unlabelled_image_feature_descriptor.append(imagedb1_4_bow.find({"_id": unlabelled_images})[0][model])

        unlabelled_image_fd = np.asarray(unlabelled_image_feature_descriptor)
        n_d, w_d = Up.shape
        n_p, w_p = Ud.shape
        sum_of_centroid_dorsal = np.sum(Ud, axis=0)
        sum_of_centroid_palmar = np.sum(Up, axis=0)

        centroid_dorsal = sum_of_centroid_dorsal / n_d
        centroid_palmar = sum_of_centroid_palmar / n_p

        dist_d = []
        dist_p = []

        for images in Ud:
            dist_d.append(np.linalg.norm(images - centroid_dorsal))

        for images in Up:
            dist_p.append(np.linalg.norm(images - centroid_palmar))

        pyplot.hist(dist_d, 50, color="darkblue", alpha=0.5, label='dorsal')

        pyplot.hist(dist_p, 50, color="yellow", alpha=0.5, label='palmar')
        pyplot.legend()
        pyplot.show()

        threshold = 10.45

        U_ul = pca.fit_transform(unlabelled_image_fd)
        n_ul, w_ul = U_ul.shape
        sum_of_centroid_unlabelled = np.sum(U_ul, axis=0)
        centroid_unlabelled = sum_of_centroid_unlabelled / n_ul
        dict_ul = {}
        pred_labels = []
        for index, images in enumerate(U_ul):
            if np.linalg.norm(images - centroid_unlabelled) > threshold:
                dict_ul[unlabelled_imagelist[index]] = 'dorsal'
                pred_labels.append(-1)
            else:
                dict_ul[unlabelled_imagelist[index]] = 'palmar'
                pred_labels.append(1)
        print(dict_ul)
        t1_accuracy = Task1.accuracy(self, pred_labels, unlabelled_imagelist)
        imagedb14.label_unlabel_bow.drop()

        return dict_ul, t1_accuracy

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

    def create_bow(self, model, k, labelled_csv, unlabelled_csv):
        label_meta_obj = imagedb14[labelled_csv]
        unlabel_meta_obj = imagedb14[unlabelled_csv]
        label_img_list = []
        unlabel_img_list = []
        for descriptor in label_meta_obj.find():
            label_img_list.append(descriptor["imageName"])
        for descriptor in unlabel_meta_obj.find():
            unlabel_img_list.append(descriptor["imageName"])
        img_list = label_img_list + unlabel_img_list
        fbe.calculate_Bow_unlabelled_imgs(img_list, model, k)


if __name__ == '__main__':
    t1 = Task1()

    # Task 1 Query 1
    # t1.create_bow("SIFT", 60, "labelled_set1", "unlabelled_set1")
    # results = t1.classify_DP("SIFT", 30, "labelled_set1", "unlabelled_set1")
    # vz.visualize_labelled_images(results[0], 30, '', 0, results[1])

    # # Task 1 Query 2
    # t1.create_bow("SIFT", 60, "labelled_set1", "unlabelled_set2")
    # results = t1.classify_DP("SIFT", 30, "labelled_set1", "unlabelled_set2")
    # vz.visualize_labelled_images(results[0], 30, '', 0, results[1])
    # # Task 1 Query 3
    # t1.create_bow("SIFT", 60, "labelled_set2", "unlabelled_set1")
    # results = t1.classify_DP("SIFT", 30, "labelled_set2", "unlabelled_set1")
    # vz.visualize_labelled_images(results[0], 30, '', 0, results[1])
    # # Task 1 Query 4
    t1.create_bow("SIFT", 60, "labelled_set2", "unlabelled_set2")
    results = t1.classify_DP("SIFT", 30, "labelled_set2", "unlabelled_set2")
    vz.visualize_labelled_images(results[0], 30, '', 0, results[1])
