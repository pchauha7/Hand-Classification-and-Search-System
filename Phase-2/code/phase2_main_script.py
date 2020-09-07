import argparse
import os
import glob
import shutil
from os.path import isfile, join

from LocalBinaryPatterns import LBP
from ColorMoments import CM
from SIFT import SIFT
from HOGmain import HOG
from PrincipleComponentAnalysis import P_CA
from NonNegativeMatrix import NM_F
from SingularValueDecomposition import SVD
from LatentDirichletAllocation import LDA
from SimilarSubject import Subject
from Task8 import Task8
from feature_descriptor import *

#Parsing the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-M', '--model',action="store", dest="model",help="Provide any of these model: LBP, SIFT, CM, HOG", default="SIFT")
parser.add_argument('-T', '--technique',action="store", dest="technique",help="Provide any of these technique: PCA, LDA, NMF, SVD", default="SVD")
parser.add_argument('-d', '--dir',action="store", dest="dir",help="Provide directory name", default="None")
parser.add_argument('-l', '--label',action="store", dest="label",help="Provide label of image: left, access, dorsal etc", default="None")
parser.add_argument('-i', '--imageid',action="store", dest="imageid",help="Provide image name", default="None")
parser.add_argument('-k', '--klatent',type=int, dest="klatent",help="Provide k value to get k latent symantics", default=20)
parser.add_argument('-m', '--mimage',type=int, dest="mimage",help="Provide m value to get m similar images", default=10)
parser.add_argument('-s', '--subject',type=int, dest="subject",help="Provide subject Id", default=-1)
parser.add_argument('-t', '--taskid',type=int, dest="taskid", help="Provide the task number", default=-1)


args = parser.parse_args()

if not 0 <= args.taskid <=9:
    print("Please provide valid task Id using option -t OR --taskid")
    exit(1)

task_id = args.taskid
model = args.model
technique = args.technique

# md = None
# if model == "CM":
#     md = CM()
# elif model == "SIFT":
#     md = SIFT()
# elif model == "LBP":
#     md = LBP()
# elif model == "HOG":
#     md = HOG()
# else:
#     print("Please provide proper model name")
#     exit(1)

teq = None
if technique == "SVD":
    teq = SVD()
elif technique == "LDA":
    teq = LDA()
elif technique == "PCA":
    teq = P_CA()
elif technique == "NMF":
    teq = NM_F()
else:
    print("Please provide proper dimensionality reduction technique")
    exit(1)

k = args.klatent
m = args.mimage

if task_id == 0:
    if args.dir == "None":
        print("Please provide directory name")
        exit(1)
    path = args.dir
    calculate_fd(path)
    #Creating Bags for each model and saving into Database
    createKMeans("CM", 40)
    createKMeans("SIFT", 40)
    createKMeans("LBP", 40)
    createKMeans("HOG", 40)
    #Creating Meta data csv file
    meta_file = os.path.join("..", "csv","HandInfo.csv")
    onlyfiles = [f for f in os.listdir(path) if isfile(join(path, f))]
    df = pd.read_csv(meta_file)
    meta_data = []
    for index, row in df.iterrows():
        if row['imageName'] in onlyfiles:
            orient = row["aspectOfHand"].split(" ")
            meta_data.append([row["id"], row["age"], row["gender"], row["skinColor"], row["accessories"], row["nailPolish"],orient[0], orient[1], row["imageName"]])

    df1 = pd.DataFrame(meta_data,columns=["SubjectID", "age", "gender", "skinColor", "accessories", "nailPolish", "aspectOfHand","Orientation", "imageName"], index=None)
    csv_path = os.path.join("..", "csv", "ImageMetadata.csv")
    df1.to_csv(csv_path, index=None)
    # try:
    #     p = os.subprocess.Popen(['mongoimport', '--port', "27018", '--db', 'imagedb', '--type', 'csv', '--file', csv_path, '--headerline'])
    # except:
    #     print("Could not execute import DB operation properly, please do it again if required.")

    #subjectMeta()
    exit(0)

elif task_id ==1:
    teq.createKLatentSymantics(model, k)
    exit(0)
elif task_id ==2:
    if args.imageid == "None":
        print("Please provide ImageID")
        exit(1)
    image = args.imageid
    teq.mSimilarImage(image, model, k, m)
    exit(0)
elif task_id ==3:
    if args.label == "None":
        print("Please provide Label")
        exit(1)
    teq.LabelLatentSemantic(args.label, model, k)
    exit(0)
elif task_id ==4:
    if args.imageid == "None":
        print("Please provide ImageID")
        exit(1)
    if args.label == "None":
        print("Please provide Label")
        exit(1)
    image = args.imageid
    teq.mSimilarImage_Label(image, args.label, model, k, m)
    exit(0)
elif task_id ==5:
    if args.dir == "None":
        print("Please provide directory name")
        exit(1)
    if args.imageid == "None":
        print("Please provide ImageID")
        exit(1)
    image = os.path.join(args.dir,args.imageid)
    shutil.copy(image, const.DB_IMG_PATH)
    teq.ImageClassfication(image, model,k)
    exit(0)

elif task_id ==6:
    if args.subject == -1:
        print("Please provide proper subject id")
        exit(1)
    sub = Subject()
    sub.similar3Subjects("SIFT", 20, args.subject)
    exit(0)

elif task_id ==7:
    sub = Subject()
    sub.createSSMatrix(k)
    exit(0)

elif task_id ==8:
    t8 = Task8()
    csv_path = os.path.join("..", "csv", "ImageMetadata.csv")
    t8.run_task_8(csv_path, k)
    exit(0)

elif task_id ==9:
    subjectMeta()
    exit(0)


