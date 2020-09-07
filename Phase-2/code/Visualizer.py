import tkinter as tk
import cv2
import os
import pymongo
from HorizontalScrollableFrame import HSF
from VerticalScrollableFrame import VSF
import Constants as const
from PIL import Image
from PIL import ImageTk

""" Constants.py should hold the path containing all the images"""
img_dir = const.DB_IMG_PATH
thumbnail_size = (160, 120)
metadata_width = 900
imgspace_width = 1300
ls_width = 1125
subj_width = 1300
data_ls_height = 780
ftr_hdp_width =1350
ftr_hdp_height = 650
ftr_ls_height = 225
subject_ls_height = 850
client = pymongo.MongoClient('localhost', const.MONGODB_PORT)
imagedb = client["imagedb"]


def get_subject_imgnames(subject):
    img_names = []

    for img in subject['dorsal_left']:
        img_names.append(img)
    for img in subject['dorsal_right']:
        img_names.append(img)
    for img in subject['palmar_left']:
        img_names.append(img)
    for img in subject['palmar_right']:
        img_names.append(img)

    # first_img = 0
    """ Code to take at most one image from each aspect
    if len(subject['dorsal_left']) > 0:
        img_names.append(subject['dorsal_left'][first_img])
    if len(subject['dorsal_right']) > 0:
        img_names.append(subject['dorsal_right'][first_img])
    if len(subject['palmar_left']) > 0:
        img_names.append(subject['palmar_left'][first_img])
    if len(subject['palmar_right']) > 0:
        img_names.append(subject['palmar_right'][first_img])"""

    return img_names

def create_thumbnail(img_id):
    # Load an image using OpenCV
    img_path = os.path.join(img_dir, img_id)
    # print('Loading image at path: %s' % img_path)
    cv_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    tn_img = cv2.resize(cv_img, thumbnail_size, interpolation=cv2.INTER_AREA)
    return tn_img


def visualize_matching_images(q_img, images_data, k, m, technique, fm, label):
    photos = []
    imgdata_to_visualize = []
    # Create a window
    window = tk.Tk()
    title_txt = "Visualization of %s Most Related Images for %s with %s Feature Descriptors" % (str(m), technique, fm)
    if label != '':
        title_txt = title_txt + ' and label: ' + label
    window.title(title_txt)
    """ Add an additional Label to display number of Latent Semantics"""
    row = tk.Frame(window, relief=tk.RIDGE, borderwidth=2)
    cur_row = 0
    ls_lbl = tk.Label(row, text='Using Top-%s Latent Semantics' % str(k))
    row.grid(row=cur_row, column=0, columnspan=2)
    ls_lbl.grid(row=cur_row, column=0, columnspan=2)
    cur_row += 1
    q_header = tk.Frame(window, relief=tk.RIDGE, borderwidth=2)
    q_lbl = tk.Label(q_header, text='Query Image')
    q_name = tk.Label(q_header, text='Query Image ID')
    """ rowspan set to 11 as that is the max count of rows for images 
        and data that will be stored for 5 or more similar images."""
    q_header.grid(row=0, column=0, columnspan=2, rowspan=11)
    q_lbl.grid(row=0, column=0)
    q_name.grid(row=0, column=1)
    # q_row = tk.Frame(window)
    q_cimg = create_thumbnail(q_img)
    q_canvas = tk.Canvas(q_header, width=thumbnail_size[0], height=thumbnail_size[1])
    q_photo = ImageTk.PhotoImage(image=Image.fromarray(q_cimg))
    # Add a PhotoImage to the Canvas
    q_canvas.create_image(0, 0, image=q_photo, anchor=tk.NW)
    # print('Giving label %s to last image loaded' % q_img)
    # print()
    q_id = tk.Label(q_header, text=q_img)
    q_canvas.grid(row=1, column=0)
    q_id.grid(row=1, column=1)

    img_label = tk.Label(window, text='Image ID')
    img_score = tk.Label(window, text='Matching Score')
    img_label.grid(row=0, column=2)
    img_score.grid(row=0, column=3)
    cur_row = 1
    img_col = 2
    score_col = 3
    count = 0
    for key, value in sorted(images_data.items(), key=lambda item: item[1]):
        if count < m:
            """ This will only happen if we have more than 5 similar images to visualize.
                In this case repeat the header frame above next 5 similar images. """
            if cur_row == 0:
                img_label = tk.Label(window, text='Image ID')
                img_score = tk.Label(window, text='Matching Score')
                img_label.grid(row=cur_row, column=img_col)
                img_score.grid(row=cur_row, column=score_col)
                cur_row += 1

            row = tk.Frame(window, relief=tk.RIDGE, borderwidth=2)

            tn_img = create_thumbnail(key)

            # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
            height, width, no_channels = tn_img.shape

            # Create a canvas that can fit the above image
            canvas = tk.Canvas(row, width=width, height=height)
            # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
            photo = ImageTk.PhotoImage(image=Image.fromarray(tn_img))
            photos.append(photo)
            # Add a PhotoImage to the Canvas
            canvas.create_image(0, 0, image=photos[count], anchor=tk.NW)
            # print('Giving label %s to last image loaded' % key)
            # print()
            img_match_lbl = tk.Label(window, text='Related Image #%s' % str(count + 1))
            match_label = tk.Label(window, text=key)
            img_match_lbl.grid(row=cur_row, column=img_col)
            match_label.grid(row=cur_row, column=score_col)
            """ After displaying the label of the image to be displayed up the current row value. """
            cur_row += 1
            label = tk.Label(row, text=str(value))
            row.grid(row=cur_row, column=img_col, columnspan=2)
            canvas.grid(row=cur_row, column=img_col)
            label.grid(row=cur_row, column=score_col)
            """ After adding the image thumbnail and score up the current row value. """
            cur_row += 1
            imgdata_to_visualize.append((canvas, label))
            """ Up the count after adding the image thumbnail and score. """
            count += 1
            """ If we have 5 images go ahead and reset current row to 0 so if more images occur we can display
                those images in the following column. Also increase the columns for images and scores by 2. """
            if count % 5 == 0:
                cur_row = 0
                img_col += 2
                score_col += 2
        else:
            break

    window.mainloop()


def visualize_data_ls(data_ls, technique, fm, label):
    """ Function to visualize the Data to Latent Semantics Matrix.
        Note k value is not needed here as the data_ls is a list of dataFrames of length k. """
    photos = []

    # Create a window
    window = tk.Tk()
    k = len(data_ls.values)
    title_txt = "Visualization of Top-%s Data-Latent Semantics for %s with %s Feature Descriptors" % (k, technique, fm)
    if label != '':
        title_txt = title_txt + ' and label: ' + label
    window.title(title_txt)

    frame = HSF(window, ls_width, data_ls_height)

    v_row = 0
    img_col = 0
    lbl_col = 1
    ls_count = 1
    p_count = 0
    """ ls_list is a list of tuples of (images, scores) """
    for ls_list in data_ls.values:
        ls_label = tk.Label(frame.scrollable_frame, text='Latent Semantic %s' % ls_count)
        ls_label.grid(row=v_row, column=img_col, columnspan=2)
        """ After displaying the latent semantic label up the current row value. """
        v_row += 1
        for img, score in ls_list:
            row = tk.Frame(frame.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
            tn_img = create_thumbnail(img)
            # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
            height, width, no_channels = tn_img.shape
            # Create a canvas that can fit the above image
            canvas = tk.Canvas(row, width=width, height=height)
            # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
            photo = ImageTk.PhotoImage(image=Image.fromarray(tn_img))
            photos.append(photo)
            canvas.create_image(0, 0, image=photos[p_count], anchor=tk.NW)
            """ Up the photo count by 1 so next image will be retrieved from the correct index """
            p_count += 1
            # print('Giving label %s to last image loaded' % img)
            # print()
            match_label = tk.Label(frame.scrollable_frame, text=img)
            match_label.grid(row=v_row, column=img_col, columnspan=2)
            """ After displaying the label of the image to be displayed up the current row value. """
            v_row += 1
            label = tk.Label(row, text=str(round(score, 8)))
            row.grid(row=v_row, column=img_col, columnspan=2)
            canvas.grid(row=v_row, column=img_col)
            label.grid(row=v_row, column=lbl_col)
            """ After adding the image thumbnail and score up the current row value. """
            v_row += 1

        """ After going through each list reset the row back to 0, and up the Latent Semantic count by 1
            Also increase the image column and label column each by 2. """
        v_row = 0
        ls_count += 1
        img_col += 2
        lbl_col += 2

    frame.pack(expand=True, fill='both')
    window.mainloop()


def visualize_feature_ls(feature_ls, technique, fm, label):
    """ Function to visualize the Feature to Latent Semantics Matrix.
        Note k value is not needed here as the data_ls is a list of dataFrames of length k. """
    # Create a window
    window = tk.Tk()
    k = len(feature_ls.values)
    title_txt = "Visualization of Top-%s Feature-Latent Semantics for %s with %s Feature Descriptors" % (k, technique, fm)
    if label != '':
        title_txt = title_txt + ' and label: ' + label
    window.title(title_txt)

    frame = HSF(window, ls_width, ftr_ls_height)

    v_row = 0
    ftr_col = 0
    lbl_col = 1
    ls_count = 1
    """ ls_list is a list of tuples of (images, scores) """
    for ls_list in feature_ls.values:
        ls_label = tk.Label(frame.scrollable_frame, text='Latent Semantic %s' % ls_count)
        ls_label.grid(row=v_row, column=ftr_col, columnspan=2)
        """ After displaying the latent semantic label up the current row value. """
        v_row += 1
        row = tk.Frame(frame.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
        feature_id = tk.Label(row, text="Feature Identifier", width=15)
        score_id = tk.Label(row, text="Feature Score", width=15)
        row.grid(row=v_row, column=ftr_col, columnspan=2)
        feature_id.grid(row=v_row, column=ftr_col)
        score_id.grid(row=v_row, column=lbl_col)
        """ After adding identifier labels up the current row value. """
        v_row += 1
        for feature, score in ls_list:
            data_row = tk.Frame(frame.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
            feature_label = tk.Label(data_row, text=str(feature), width=14)
            score_label = tk.Label(data_row, text=str(round(score, 8)), width=16)
            data_row.grid(row=v_row, column=ftr_col, columnspan=2)
            feature_label.grid(row=v_row, column=ftr_col)
            score_label.grid(row=v_row, column=lbl_col)
            """ After adding the image thumbnail and score up the current row value. """
            v_row += 1

        """ After going through each list reset the row back to 0, and up the Latent Semantic count by 1
            Also increase the image column and label column each by 2. """
        v_row = 0
        ls_count += 1
        ftr_col += 2
        lbl_col += 2

    frame.pack(expand=True, fill='both')
    window.mainloop()


def visualize_ftr_ls_hdp(feature_ls, technique, fm):
    """ Function to visualize the Feature to Latent Semantics of Images with Highest Dot Product. """
    photos = []

    # Create a window
    window = tk.Tk()
    k = len(feature_ls.items())

    title_txt = "Visualization of Top-%s Feature-Latent Semantics for %s with %s Feature Descriptors" % (k, technique, fm)
    window.title(title_txt)

    frame = VSF(window, ftr_hdp_width, ftr_hdp_height)

    v_row = 0
    img_col = 0
    hdp_label = tk.Label(frame.scrollable_frame, text='Displaying Images with Highest Dot Product to each of the Top-'
                                                      '%s Latent Semantics' % str(k))
    hdp_label.grid(row=v_row, column=img_col, columnspan=10)
    cur_row = 1
    img_lbl_col = 1
    p_count = 0

    """ feature_ls is a dictionary containing Latent Semantic numbers and image with HDP to that latent semantic """
    for ls, img in feature_ls.items():
        """ v_row should be always at whatever the value of cur_row is at. """
        v_row = cur_row

        ls_label = tk.Label(frame.scrollable_frame, text='Latent Semantic %s' % ls)
        ls_label.grid(row=v_row, column=img_col, columnspan=2)
        """ After displaying the latent semantic label up the current row value. """
        v_row += 1

        row = tk.Frame(frame.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
        tn_img = create_thumbnail(img)
        # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
        height, width, no_channels = tn_img.shape
        # Create a canvas that can fit the above image
        canvas = tk.Canvas(row, width=width, height=height)
        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        photo = ImageTk.PhotoImage(image=Image.fromarray(tn_img))
        photos.append(photo)
        canvas.create_image(0, 0, image=photos[p_count], anchor=tk.NW)
        """ Up the photo count by 1 so next image will be retrieved from the correct index """
        p_count += 1
        # print('Giving label %s to last image loaded' % img)
        # print()
        row.grid(row=v_row, column=img_col, columnspan=2)
        canvas.grid(row=v_row, column=img_col)
        match_label = tk.Label(row, text=img)
        match_label.grid(row=v_row, column=img_lbl_col)

        img_col += 2
        img_lbl_col += 2
        """ If we have 5 images go ahead and reset img_col to 0 so if more images occur we can display
            those images in the following row. Also increase the row by 2. Skipping over label and image """
        if p_count % 5 == 0:
            cur_row += 2
            img_col = 0
            img_lbl_col = 1

    frame.pack(expand=True, fill='both')
    window.mainloop()


def visualize_classified_image(q_img, classification, technique, fm, k):
    window = tk.Tk()
    title_txt = "Classification of Unlabeled Query Image for %s with %s Feature Descriptors" % (technique, fm)
    window.title(title_txt)
    q_header = tk.Frame(window, relief=tk.RIDGE, borderwidth=2)
    q_lbl = tk.Label(q_header, text='Query Image')
    q_name = tk.Label(q_header, text='Query Image ID')
    """ rowspan set to 4 as that is the max count of rows for classification data."""
    q_header.grid(row=0, column=0, columnspan=2, rowspan=5)
    q_lbl.grid(row=0, column=0)
    q_name.grid(row=0, column=1)
    # q_row = tk.Frame(window)
    q_cimg = create_thumbnail(q_img)
    q_canvas = tk.Canvas(q_header, width=thumbnail_size[0], height=thumbnail_size[1])
    q_photo = ImageTk.PhotoImage(image=Image.fromarray(q_cimg))
    # Add a PhotoImage to the Canvas
    q_canvas.create_image(0, 0, image=q_photo, anchor=tk.NW)
    # print('Giving label %s to last image loaded' % q_img)
    # print()
    q_id = tk.Label(q_header, text=q_img)
    q_canvas.grid(row=1, column=0)
    q_id.grid(row=1, column=1)

    cur_row = 0
    lbl_col = 2
    val_col = 3
    """ Add an additional Label to display number of Latent Semantics"""
    row = tk.Frame(window, relief=tk.RIDGE, borderwidth=2)
    ls_lbl = tk.Label(row, text='Using Top-%s Latent Semantics' % str(k))
    row.grid(row=cur_row, column=lbl_col, columnspan=2)
    ls_lbl.grid(row=cur_row, column=lbl_col, columnspan=2)
    cur_row += 1
    for label, value in classification.items():
        row = tk.Frame(window, relief=tk.RIDGE, borderwidth=2)
        c_label = tk.Label(row, text=label, width=15)
        c_value = tk.Label(row, text=value, width=15)
        row.grid(row=cur_row, column=lbl_col, columnspan=2)
        c_label.grid(row=cur_row, column=lbl_col)
        c_value.grid(row=cur_row, column=val_col)
        cur_row += 1

    window.mainloop()


def visualize_similar_subjects(q_subj, subject_dict, k, fm):
    print('Visualization for Similar Subjects called')
    # Create a window
    window = tk.Tk()
    frame = VSF(window, subj_width, data_ls_height)
    title_txt = "Visualization of 3 Most Related Subjects using LDA with %s Feature Descriptors and k of %s" % (fm, str(k))
    window.title(title_txt)
    q_holder = tk.Frame(frame.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
    q_lbl = tk.Label(q_holder, text='Query Subject %s' % q_subj)
    q_name = tk.Label(q_holder, text='Query Image IDs')
    """ rowspan set to 5 as that is the max count of rows for images 
        and data that will be stored for at most 4 images per subject. """
    q_holder.grid(row=0, column=0, columnspan=2, rowspan=5)
    q_lbl.grid(row=0, column=0)
    q_name.grid(row=0, column=1)
    query_subject = imagedb.subjects.find_one({'_id': q_subj})
    img_names = get_subject_imgnames(query_subject)

    q_photos = []
    q_count = 0
    cur_row = 1
    img_col = 0
    id_col = 1
    for img in img_names:
        q_img = create_thumbnail(img)
        q_canvas = tk.Canvas(q_holder, width=thumbnail_size[0], height=thumbnail_size[1])
        q_photo = ImageTk.PhotoImage(image=Image.fromarray(q_img))
        q_photos.append(q_photo)
        # Add a PhotoImage to the Canvas
        q_canvas.create_image(0, 0, image=q_photos[q_count], anchor=tk.NW)
        q_count += 1
        # print('Giving label %s to last image loaded' % q_img)
        # print()
        q_id = tk.Label(q_holder, text=img)
        q_canvas.grid(row=cur_row, column=img_col)
        q_id.grid(row=cur_row, column=id_col)
        cur_row += 1

    subj_photos = []
    subj_row = 0
    s_count = 0
    s_photo_count = 0
    subject_num = 1
    s_img_col = img_col + 2
    s_id_col = id_col + 2
    for subject, score in sorted(subject_dict.items(), key=lambda item: item[1]):
        """ Only show top 3 similar subjects """
        if s_count < 3:
            subj_holder = tk.Frame(frame.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
            subject_lbl = tk.Label(subj_holder, text='Similar Subject #%s: %s' % (subject_num, subject))
            subject_score = tk.Label(subj_holder, text='Score: %s' % str(score)) # (round(score, 12)))
            subj_holder.grid(row=subj_row, column=s_img_col, columnspan=2, rowspan=5)
            subject_lbl.grid(row=subj_row, column=s_img_col)
            subject_score.grid(row=subj_row, column=s_id_col)
            """ Up the row after the labels """
            subj_row += 1
            s_collection = imagedb.subjects.find_one({'_id': subject})
            s_imgs = get_subject_imgnames(s_collection)

            for img in s_imgs:
                s_img = create_thumbnail(img)
                s_canvas = tk.Canvas(subj_holder, width=thumbnail_size[0], height=thumbnail_size[1])
                s_photo = ImageTk.PhotoImage(image=Image.fromarray(s_img))
                subj_photos.append(s_photo)
                # Add a PhotoImage to the Canvas
                s_canvas.create_image(0, 0, image=subj_photos[s_photo_count], anchor=tk.NW)
                # print('Giving label %s to last image loaded' % s_img)
                # print()
                q_id = tk.Label(subj_holder, text=img)
                s_canvas.grid(row=subj_row, column=s_img_col)
                q_id.grid(row=subj_row, column=s_id_col)
                """ Up the count after adding the image thumbnail """
                s_photo_count += 1

                subj_row += 1

            """ Move the columns over 2 after adding each subject. """
            s_img_col += 2
            s_id_col += 2
            """ Also up the subject number """
            subject_num += 1
            s_count += 1
            """ Reset the row to 0 after each subject finishes."""
            subj_row = 0

    frame.pack(expand=True, fill='both')
    window.mainloop()


def visualize_ss_matrix(ss_ls, k):
    """ Function to visualize the Subject Similarity Matrix """
    # Create a window
    window = tk.Tk()
    title_txt = "Visualization of Top-%s Latent Semantics as subject-weight pairs" % str(k)
    window.title(title_txt)

    frame = HSF(window, ls_width, subject_ls_height)

    v_row = 0
    ftr_col = 0
    lbl_col = 1
    ls_count = 1
    """ ls_list is a list of tuples of (images, scores) """
    for ls_list in ss_ls.values:
        ls_label = tk.Label(frame.scrollable_frame, text='Latent Semantic %s' % ls_count)
        ls_label.grid(row=v_row, column=ftr_col, columnspan=2)
        """ After displaying the latent semantic label up the current row value. """
        v_row += 1
        row = tk.Frame(frame.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
        feature_id = tk.Label(row, text="Subject Identifier", width=15)
        score_id = tk.Label(row, text="Subject Score", width=15)
        row.grid(row=v_row, column=ftr_col, columnspan=2)
        feature_id.grid(row=v_row, column=ftr_col)
        score_id.grid(row=v_row, column=lbl_col)
        """ After adding identifier labels up the current row value. """
        v_row += 1
        for feature, score in ls_list:
            data_row = tk.Frame(frame.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
            feature_label = tk.Label(data_row, text=str(feature), width=14)
            score_label = tk.Label(data_row, text=str(round(score, 8)), width=16)
            data_row.grid(row=v_row, column=ftr_col, columnspan=2)
            feature_label.grid(row=v_row, column=ftr_col)
            score_label.grid(row=v_row, column=lbl_col)
            """ After adding the image thumbnail and score up the current row value. """
            v_row += 1

        """ After going through each list reset the row back to 0, and up the Latent Semantic count by 1
            Also increase the image column and label column each by 2. """
        v_row = 0
        ls_count += 1
        ftr_col += 2
        lbl_col += 2

    frame.pack(expand=True, fill='both')
    window.mainloop()


def visualize_img_space(k, img_space):
    print("Visualizing Image Space")
    photos = []

    # Create a window
    window = tk.Tk()
    title_txt = "Visualization of Top-%s Latent Semantics in the Image-space" % str(k)
    window.title(title_txt)

    frame = VSF(window, imgspace_width, data_ls_height)

    v_row = 0
    img_col = 0
    lbl_col = 1
    ls_count = 1
    p_count = 0
    """ ls_list is a list of tuples of (images, scores) """
    for ls_list in img_space.values:
        ls_label = tk.Label(frame.scrollable_frame, text='Latent Semantic %s' % ls_count)
        ls_label.grid(row=v_row, column=img_col, columnspan=2)
        """ After displaying the latent semantic label up the current row value. """
        v_row += 1
        for img, score in ls_list:
            row = tk.Frame(frame.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
            tn_img = create_thumbnail(img)
            # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
            height, width, no_channels = tn_img.shape
            # Create a canvas that can fit the above image
            canvas = tk.Canvas(row, width=width, height=height)
            # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
            photo = ImageTk.PhotoImage(image=Image.fromarray(tn_img))
            photos.append(photo)
            canvas.create_image(0, 0, image=photos[p_count], anchor=tk.NW)
            """ Up the photo count by 1 so next image will be retrieved from the correct index """
            p_count += 1
            # print('Giving label %s to last image loaded' % img)
            # print()
            match_label = tk.Label(frame.scrollable_frame, text=img)
            match_label.grid(row=v_row, column=img_col, columnspan=2)
            """ After displaying the label of the image to be displayed up the current row value. """
            v_row += 1
            label = tk.Label(row, text=str(round(score, 8)))
            row.grid(row=v_row, column=img_col, columnspan=2)
            canvas.grid(row=v_row, column=img_col)
            label.grid(row=v_row, column=lbl_col)
            """ After adding the image thumbnail and score up the current row value. """
            v_row += 1

        """ After going through each list reset the row back to 0, and up the Latent Semantic count by 1
            Also increase the image column and label column each by 2. """
        v_row = 0
        ls_count += 1
        img_col += 2
        lbl_col += 2

    frame.pack(expand=True, fill='both')
    window.mainloop()


def visualize_metadata_space(k, metadata_space):
    print("Visualizing Metadata Space")
    # Create a window
    window = tk.Tk()
    title_txt = "Visualization of Top-%s Latent Semantics in the Metadata-space" % str(k)
    window.title(title_txt)

    frame = HSF(window, metadata_width, ftr_ls_height)

    v_row = 0
    ftr_col = 0
    lbl_col = 1
    ls_count = 1
    """ ls_list is a list of tuples of (images, scores) """
    for ls_list in metadata_space.values:
        ls_label = tk.Label(frame.scrollable_frame, text='Latent Semantic %s' % ls_count)
        ls_label.grid(row=v_row, column=ftr_col, columnspan=2)
        """ After displaying the latent semantic label up the current row value. """
        v_row += 1
        row = tk.Frame(frame.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
        feature_id = tk.Label(row, text="Subject Identifier", width=15)
        score_id = tk.Label(row, text="Subject Score", width=15)
        row.grid(row=v_row, column=ftr_col, columnspan=2)
        feature_id.grid(row=v_row, column=ftr_col)
        score_id.grid(row=v_row, column=lbl_col)
        """ After adding identifier labels up the current row value. """
        v_row += 1
        for feature, score in ls_list:
            data_row = tk.Frame(frame.scrollable_frame, relief=tk.RIDGE, borderwidth=2)
            feature_label = tk.Label(data_row, text=str(feature), width=14)
            score_label = tk.Label(data_row, text=str(round(score, 8)), width=16)
            data_row.grid(row=v_row, column=ftr_col, columnspan=2)
            feature_label.grid(row=v_row, column=ftr_col)
            score_label.grid(row=v_row, column=lbl_col)
            """ After adding the image thumbnail and score up the current row value. """
            v_row += 1

        """ After going through each list reset the row back to 0, and up the Latent Semantic count by 1
            Also increase the image column and label column each by 2. """
        v_row = 0
        ls_count += 1
        ftr_col += 2
        lbl_col += 2

    frame.pack(expand=True, fill='both')
    window.mainloop()
