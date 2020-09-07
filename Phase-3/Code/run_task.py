import cm
#import hog
import cv2
import hog
import pymongo

myclient = pymongo.MongoClient('localhost', 27018)
mydb = myclient["imagedb"]
mycol = mydb["image_models"]
keyExists = False


# To implement task 1 functionality
def call_task1(imageID, model):  # Call cm functionality of task 1
    # Create a cursor object to find if the imageID already exists in the database
    cursor = mydb.image_models.find({"_id": imageID})

    if model == 'CM':
        # Check if color moments for imageID has already been calculated

        # If the keyExists and CM has already been calculated for the given imageID, retrieve it from the DB
        if cursor.count() > 0:
            if "CM" in cursor[0].keys():
                print(cursor[0]["CM"])
            else:
                feature_descriptor = cm.calculate_cm(imageID, 1)
                mydb.image_models.update_one({"_id" : imageID}, {"$push": {"CM" : feature_descriptor}})
        else:
            feature_descriptor = cm.calculate_cm(imageID, 1)
            dict = {}
            dict["_id"] = imageID
            dict["CM"] = feature_descriptor
            rec = mydb.image_models.insert_one(dict)
            if feature_descriptor is None:
                print("Error!!!!")

    elif model == 'HOG':  # Call hog functionality of task 1
        # Check if hog for imageID has already been calculated
        # If the keyExists and CM has already been calculated for the given imageID, retrieve it from the DB
        if cursor.count() > 0:
            if "HOG" in cursor[0].keys():
                print(cursor[0]["HOG"])
            else:
                feature_descriptor = hog.calculate_hog(imageID, 1)
                mydb.image_models.update_one({"_id" : imageID}, {"$push": {"HOG" : feature_descriptor}})
        else:
            feature_descriptor = hog.calculate_hog(imageID, 1)
            dict = {}
            dict["_id"] = imageID
            dict["HOG"] = feature_descriptor
            rec = mydb.image_models.insert_one(dict)
            if feature_descriptor is None:
                print("Error!!!!")

    else:
        print("Model not found!!!!")

# To implement task 2 functionality
def call_task2(path):
    # Color Moments implementation
    feature_descriptor_cm = cm.display_cm_task2(path, 2) # Call cm functionality of task 2
    # HOG implementation
    feature_descriptor_hog = hog.display_hog_task2(path, 2) # Call cm functionality of task 2


# To implement task 3 functionality
def call_task3(imageID, model, k):
    if model == "CM":
        cm.display_cm_task3(imageID, k)  # Call cm functionality of task 3
    elif model == "HOG":
        hog.display_hog_task3(imageID, k)  # Call cm functionality of task 3
    else:
        print("Model not found!!!!!")

# The main code starts here
# for document in mycol.find():
#     print (document)
task = int(input("Enter Task: "))
# Task 1: Pass ImageID, Model to extract feature descriptor
if task == 1:
    imageID = input("Enter Image ID: ")
    model = input("Enter model: ")
    call_task1(imageID, model)
# Task 2: Pass path to folder containing images
elif task == 2:
    path = input("Enter Path: ")
    call_task2(path)
# Task 3: Pass imageID, model and k to find k most similar images
elif task == 3:
    imageID = input("Enter Image ID: ")
    model = input("Enter model: ")
    k = int(input("Enter value of k: "))
    call_task3(imageID, model, k)
else:
    print("Model not found!!!!")
