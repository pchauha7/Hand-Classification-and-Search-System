import argparse
import os
import glob
import shutil
from LocalBinaryPatterns import LBP
from ColorMoments import CM
from SIFT import SIFT
from HOGmain import HOG

#Parsing the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',action="store", dest="model",help="Provide any of these model: LBP, SIFT, CM, HOG", default="CM")
parser.add_argument('-d', '--dir',action="store", dest="dir",help="Provide directory name", default="None")
parser.add_argument('-r', '--ranking',action="store_true", dest="ranking",help="Enable ranking of images")
parser.add_argument('-i', '--imageLoc',action="store", dest="imageLoc",help="Provide image name", default="None")
parser.add_argument('-k', '--kimage',type=int, dest="kimage",help="Provide k value to get k similar images", default=-1)
parser.add_argument('-s', '--single_task',action="store_true", dest="single_task",help="Enable task 1 for single image")


args = parser.parse_args()

if args.dir == "None":
    print("Please provide directory name")
    exit(1)

curpath = os.path.dirname(os.path.abspath(__file__))

dirpath = os.path.join(curpath, args.dir)

if not os.path.exists(dirpath):
    print("Please provide proper directory location")
    exit(1)

imagePath = os.path.join(curpath, args.dir, args.imageLoc)

##   TASK1 #####
if args.single_task:
    if args.imageLoc == "None" or not os.path.exists(imagePath):
        print("Please provide proper directory location")
        exit(1)

    if args.model == 'CM':
        md = CM(imagePath)
        lst = md.getFeatureDescriptors()
        md.createFeatureOutputFile(lst)

    elif args.model == 'LBP':
        md = LBP(imagePath)
        lst = md.getFeatureDescriptors()
        md.createFeatureOutputFile(lst)

    elif args.model == 'SIFT':
        md = SIFT(imagePath)
        lst = md.getFeatureDescriptors()
        md.createFeatureOutputFile(lst)
        
    elif args.model == 'HOG':
        md = HOG(imagePath)
        lst = md.getFeatureDescriptors()
        md.createFeatureOutputFile(lst)

    else:
        print("Please provide proper model name")
        exit(1)

    exit(0)

if args.ranking:
    if args.imageLoc == "None" or not os.path.exists(imagePath):
        print("Please provide proper image location using '-i'")
        exit(1)
    if int(args.kimage) == -1:
        print("Please provide k value using '-k'")
        exit(1)

####### TASK 2 #######
if args.model == 'CM':
    for image in glob.glob(os.path.join(dirpath, "*.jpg")):
        md = CM(image)
        lst = md.getFeatureDescriptors()
        md.createFeatureOutputFile(lst)

elif args.model == 'LBP':
    for image in glob.glob(os.path.join(dirpath, '*.jpg')):
        md = LBP(image)
        lst = md.getFeatureDescriptors()
        md.createFeatureOutputFile(lst)

elif args.model == 'SIFT':
    for image in glob.glob(os.path.join(dirpath, "*.jpg")):
        md = SIFT(image)
        lst = md.getFeatureDescriptors()
        md.createFeatureOutputFile(lst)
        
elif args.model == 'HOG':
    for image in glob.glob(os.path.join(dirpath, "*.jpg")):
        md = HOG(image)
        lst = md.getFeatureDescriptors()
        md.createFeatureOutputFile(lst)

else:
    print("Please provide proper model name")
    exit(1)

# Compare the images based on provided arguments

######## TASK 3 #######################
if args.model == 'CM' and args.ranking:
    rank_dict = {}
    head = ""
    for image in glob.glob(os.path.join(dirpath, "*.jpg")):
        md = CM(image)
        x = md.compareImages(imagePath)
        if x == -1:
            continue
        head, tail = os.path.split(image)
        rank_dict.update({tail: x})

    k = 0
    res_dir = os.path.join(curpath, '..', 'output', 'CM', 'match')
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)
    os.mkdir(res_dir)

    print("\n\nNow printing top {} matched Images and their matching scores".format(args.kimage))
    for key, value in sorted(rank_dict.items(), key=lambda item: item[1]):
        if k < args.kimage:
            print(key + " has matching score:: " + str(value))
            shutil.copy(os.path.join(head, key), res_dir)
            k += 1
        else:
            break

elif args.model == 'LBP' and args.ranking:
    rank_dict = {}
    for image in glob.glob(os.path.join(dirpath, '*.jpg')):
        md = LBP(image)
        x = md.compareImages(imagePath)
        if x == -1:
            continue
        head, tail = os.path.split(image)
        rank_dict.update({tail: x})

    k = 0
    res_dir = os.path.join(curpath, '..', 'output', 'LBP', 'match')
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)
    os.mkdir(res_dir)

    print("\n\nNow printing top {} matched Images and their ranks".format(args.kimage))
    for key, value in sorted(rank_dict.items(), key=lambda item: item[1], reverse=True): # For cosine similarity
    # for key, value in sorted(rank_dict.items(), key=lambda item: item[1]):             # For Euclidean Distance
        if k < args.kimage:
            print(key + " has matching score:: " + str(value))
            shutil.copy(os.path.join(head, key), res_dir)
            k += 1
        else:
            break

elif args.model == 'SIFT' and args.ranking:
    rank_dict = {}
    head = ""
    for image in glob.glob(os.path.join(dirpath, "*.jpg")):
        md = SIFT(image)
        x = md.compareImages(imagePath)
        if x == -1:
            continue
        head, tail = os.path.split(image)
        rank_dict.update({tail: x})

    k = 0
    res_dir = os.path.join(curpath, 'output', 'SIFT', 'match')
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)
    os.mkdir(res_dir)
    print("\n\nNow printing top {} matched Images and their matching scores".format(args.kimage))
    for key, value in sorted(rank_dict.items(), key=lambda item: item[1], reverse=False):
        if k < args.kimage:
            print(key + " has matching score:: " + str(value))
            shutil.copy(os.path.join(head, key), res_dir)
            k += 1
        else:
            break
            
elif args.model == 'HOG' and args.ranking:
    rank_dict = {}
    for image in glob.glob(os.path.join(dirpath, '*.jpg')):
        md = HOG(image)
        x = md.compareImages(imagePath)
        if x == -1:
            continue
        head, tail = os.path.split(image)
        rank_dict.update({tail: x})

    k = 0
    res_dir = os.path.join(curpath, '..', 'output', 'HOG', 'match')
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)
    os.mkdir(res_dir)

    print("\n\nNow printing top {} matched Images and their ranks".format(args.kimage))
    for key, value in sorted(rank_dict.items(), key=lambda item: item[1], reverse=True):  # For cosine similarity
        # for key, value in sorted(rank_dict.items(), key=lambda item: item[1]):             # For Euclidean Distance
        if k < args.kimage:
            print(key + " has matching score:: " + str(value))
            shutil.copy(os.path.join(head, key), res_dir)
            k += 1
        else:
            break
