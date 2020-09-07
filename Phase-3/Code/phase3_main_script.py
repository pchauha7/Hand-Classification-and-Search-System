import argparse
import os
import glob
import shutil
from os.path import isfile, join
import pandas as pd
import Constants as const


from Task1 import Task1
from LSH import LSH
from Task4SVM import SVM
from Task4DecisionTree import DecTree
from PPR import PersonalizedPageRank
from Task2 import Query_input
from feature_descriptor import *


#Parsing the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir',action="store", dest="dir",help="Provide directory name", default="None")
parser.add_argument('-l', '--label',action="store", dest="label",help="Provide labelled csv name", default="None")
parser.add_argument('-u', '--unlabel',action="store", dest="unlabel",help="Provide unlabelled csv name", default="None")
parser.add_argument('-i', '--imageid',action="store", dest="imageid",help="Provide image name", default="None")
parser.add_argument('-k', '--klatent',type=int, dest="klatent",help="Provide k value to get k latent symantics", default=20)
parser.add_argument('-c', '--centers',type=int, dest="centers",help="Provide centers count", default=10)
parser.add_argument('-L', '--layers',type=int, dest="layers",help="Provide layers count", default=10)
parser.add_argument('-m', '--mimage',type=int, dest="mimage",help="Provide m value to get m similar images", default=10)
parser.add_argument('-t', '--taskid',type=int, dest="taskid", help="Provide the task number", default=-1)
parser.add_argument('-I', '--list',type=list, action='store',dest='list',help='Pass the image list',required=True)
parser.add_argument('-T', '--type',action="store", dest="type",help="Provide type of classifier", default="None")


args = parser.parse_args()

if not 0 <= args.taskid <= 5:
    print("Please provide valid task Id using option -t OR --taskid")
    exit(1)

task_id = args.taskid
k = args.klatent
m = args.mimage
c = args.centers

if task_id == 0:
    if args.dir == "None":
        print("Please provide directory name")
        exit(1)
    path = args.dir
    labelled_path = os.join.path(path, 'Labelled')
    unlabelled_path = os.join.path(path, 'Unlabelled')

    for label_dir in os.walk(labelled_path):
        calculate_fd(label_dir)
    for unlabel_dir in os.walk(labelled_path):
        calculate_fd(label_dir)
    #Creating Bags for each model and saving into Database
    createKMeans("CM", 40)
    createKMeans("LBP", 40)
    createKMeans("SIFT", 40)
    exit(0)

elif task_id ==1:
    if args.label == "None" or args.unlabel == "None":
        print("Please provide proper csv names")
        exit(1)
    t1 = Task1()
    t1.create_bow("SIFT", 60, args.label, args.unlabel)
    image_classified = t1.classify_DP("SIFT", k, args.label, args.unlabel)
    exit(0)
elif task_id ==2:
    if args.label == "None" or args.unlabel == "None":
        print("Please provide proper csv names")
        exit(1)
    Query_input(c, args.label, args.unlabel)
    exit(0)
elif task_id ==3:
    if args.label == "None":
        print("Please provide Label")
        exit(1)
    if len(args.list) == 0:
        print("Please provide images in the list")
        exit(1)
    x = PersonalizedPageRank()
    x.getKDominantImagesUsingPPR(c, m, args.list, args.label)
    exit(0)
elif task_id ==4:
    clss_typ= args.type
    if clss_typ == "None":
        print("Please provide type of Classification")
        exit(1)
    if args.label == "None" or args.unlabel == "None":
        print("Please provide proper csv names")
        exit(1)
    if clss_typ == "PPR":
        x = PersonalizedPageRank()
        print(x.classifyUnlabelledImagesUsingPPR(5, args.label, args.unlabel))
    elif clss_typ == "SVM":
        x = SVM()
        x.preprocess_SVM("HOG", 20, args.label, args.unlabel)
    elif clss_typ == "DT":
        x = DecTree()
        x.preprocess_dectree("HOG", 20, args.label, args.unlabel)
    else:
        print("Not a valid Classifier")
        exit(1)
    exit(0)
elif task_id ==5:
    if args.dir == "None":
        print("Please provide directory name")
        exit(1)
    if args.imageid == "None":
        print("Please provide ImageID")
        exit(1)
    #image = os.path.join(args.dir,args.imageid)
    csv_path = os.path.join("..", "csv", "output_pca_final.csv")
    img_df = pd.read_csv(csv_path, header=None)
    lsh = LSH()
    result = lsh.img_ann(img_df, args.imageid, m, args.layers, k)
    print(result)
    exit(0)


