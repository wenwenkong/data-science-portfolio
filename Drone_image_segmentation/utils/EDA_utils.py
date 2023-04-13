import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def get_classes_RGB(df, col_class, col_R, col_G, col_B):
    '''
    Obtain classes names and the corresponding RGB information

    Parameters
    -----------
    df: dataframe
        contains information of the class names and the RGB information

    col_class: string
        name of the df column of class names

    col_R: string
        name of the df column of the red channel

    col_G: string
        name of the df column of the green channel

    col_B: string
        name of the df column of the blue channel

    Returns
    -----------
    CLASSES: list of strings
        contains the class names

    COLORS: list of nested lists
        each nested list contains R, G, B information corresponding to one class

    DICT: dictionary 
        contains paired values of CLASSES and COLORS
    '''

    CLASSES = df[col_class]

    COLORS=[]

    for i in range(0, len(CLASSES)):
        COLORS.append([int(df[col_R][i]), int(df[col_G][i]), int(df[col_B][i])])

    DICT = {}
    for i, label in enumerate(CLASSES):
        DICT[label] = COLORS[i]

    return CLASSES, COLORS, DICT

def plot_class_colorlabel(CLASSES, COLORS, title):
    '''
    Plot color chart of each semantic class
    
    Parameters
    ----------
    CLASSES: list of strings
        each string denote a class name
        for example, CLASSES = ['unlabeled', 'person']
    
    COLORS: list of nested lists
        each nested list contains RGB information of one class
        for example, COLORS = [[0, 0, 0], [255, 255, 255]]
    
    title: string
        title of the plot
    
    Returns
    ----------
    A matplotlib type color chart
    '''
    
    fig, ax = plt.subplots(figsize=(30, 2))
    
    for i in range(len(CLASSES)):
        ax.barh(0, 1, left=i, color=[c/255 for c in COLORS[i]], edgecolor='black')
        ax.text(i + 1, 1, CLASSES[i], ha='center', va='center', fontsize=30, rotation=45)
        
    ax.set_xlim(0, len(CLASSES))
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title(title, x=0.5, y=2, fontsize=40)
    plt.show()

def panel_one_pair(raw_path, raw_file, labeled_path, labeled_file, title):
    '''
    Panel plot of one paried images
    
    Parameters
    -----------
    raw_path: string
        folder path of the raw images
        
    raw_file: string
        file name of the raw image 
    
    labeled_path: string
        folder path of the labeled images
        
    labeled_file: string
        file name of the labeled image
        
    title: string
        title of the panel
    
    Returns
    -----------
    Panel plot
    
    '''
    
    raw_image = Image.open(raw_path + raw_file)
    labeled_image = Image.open(labeled_path + labeled_file)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    axs[0].imshow(raw_image)
    axs[0].set_title('Original Image ('+raw_file+')')
    
    axs[1].imshow(labeled_image)
    axs[1].set_title('Labeled Image ('+labeled_file+')')
    
    plt.suptitle(title, y = 0.9, fontsize=20, fontweight='bold')
    plt.show()

def panel_multiple_pairs(pairs, raw_path, labeled_path, title):
    '''
    Panel plot of multiple paired images 
    
    Parameters
    -----------
    pairs: list of tuples
        each tuple contains the paired file names
        example: [('173.jpg', '173.png'),
                  ('385.jpg', '385.png'),
                  ('460.jpg', '460.png'),
                  ('018.jpg', '018.png'),
                  ('430.jpg', '430.png')]
    
    raw_path: string
        folder path of original images
    
    labeled_path: string
        folder path of labeled images
    
    title: string
        suptitle of the panel plot
    
    Returns
    -----------
    Panel plot
    '''
    
    fig, axs = plt.subplots(len(pairs), 2, figsize=(10, 20))
    
    for i, (raw_file, labeled_file) in enumerate(pairs):
        raw_image = Image.open(os.path.join(raw_path, raw_file))
        labeled_image = Image.open(os.path.join(labeled_path, labeled_file))
        
        axs[i, 0].imshow(raw_image)
        axs[i, 0].set_title('Original Image ('+raw_file+')')
        
        axs[i, 1].imshow(labeled_image)
        axs[i, 1].set_title('Labeled Image ('+labeled_file+')')
        
    plt.suptitle(title, y = 0.9, fontsize=20, fontweight='bold')
    plt.show()

def width_height_aspect(path, files):
    '''
    Extract the width, height, and aspect ratio of the image dataset

    Parameters
    ----------
    path: string
        folder path where the files live

    files: list of strings
        file names of the images

    Returns
    ----------
    width: list of numerical values

    height: list of numerical values

    aspect: list of numerical values

    '''

    width, height, aspect = [], [], []

    for file in files:
        img = Image.open(path + file)

        temp_width, temp_height = img.size
        temp_aspect = temp_width / temp_height

        width.append(temp_width)
        height.append(temp_height)
        aspect.append(temp_aspect)

    return width, height, aspect

def class_counts(path, files, CLASSES, COLORS):
    '''
    Get the label counts based on the RGB match between pixels and label colors
    
    Parameters
    ----------
    path: string
        folder path where files live
        
    files: list of strings
        each string denotes the name of an image file
        
    CLASSES: list of strings
        each string denotes a class label
    
    COLORS: list of arrays
        each array contains the RGB information for a class label
    
    Returns
    ----------
    sorted_label_counts: dictionary
        contains the class label and label counts across all files

        the dictionary is sorted in an descending order based on label counts
    
    '''
    
    label_counts = {label: 0 for label in CLASSES}
    
    for file in files:
        im_cv = cv2.imread(path + file)
        
        im_rgb= cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        
        im = im_rgb.reshape(im_rgb.shape[0] * im_rgb.shape[1], 3)
        
        for label, label_RGB in zip(CLASSES, COLORS):
            matches = np.all(im == label_RGB, axis=1)
            label_counts[label] += np.count_nonzero(matches)
   
    label_counts_sorted_by_counts = sorted(label_counts.items(), key=lambda x:x[1], reverse=True)
    sorted_label_counts = dict(label_counts_sorted_by_counts)

    return sorted_label_counts
