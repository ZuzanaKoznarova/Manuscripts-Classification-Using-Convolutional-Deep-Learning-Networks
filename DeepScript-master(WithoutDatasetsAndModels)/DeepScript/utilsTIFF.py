from __future__ import print_function

import glob
import random
import os
import copy
from itertools import product

from keras.preprocessing.image import ImageDataGenerator

from DeepScript import augment

SEED = 7687655 # seed for the random
import numpy as np
np.random.seed(SEED)
random.seed(SEED)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

EPS=10e-8 # small constant to prevent dividing


sns.set_style("whitegrid", {'axes.grid' : False})

from sklearn.feature_extraction.image import extract_patches_2d

from skimage import io
from skimage.transform import resize, downscale_local_mean
from scipy.misc import imsave
import scipy.misc

from keras.utils import np_utils

# parameters for augmentation the dataset
AUGMENTATION_PARAMS = {
    'zoom_range': (0.75, 1.25),
    'rotation_range': (-10, 10),
    'shear_range': (-15, 15),
    'translation_range': (-12, 12),
    'do_flip': False,
    'allow_stretch': False,
}

# augmentation of dataset for test dataset
NO_AUGMENTATION_PARAMS = {
    'zoom_range': (1.0, 1.0),
    'rotation_range': (0, 0),
    'shear_range': (0, 0),
    'translation_range': (0, 0),
    'do_flip': False,
    'allow_stretch': False,
}


def load_dir(dir_name, ext='.tif'):     # fce for loading training data
    X, Y = [], []   # X is array for images, Y represents labels for the images
    fns = glob.glob(dir_name+'/*' + ext)    # load name of files from diven directory with given extension
    if(len(fns)==0):    # because we used two different kinds of the file extension and we do not want to send it as an argument
        fns = glob.glob(dir_name + '/*' + '.tiff')  # load name of files from diven directory with given extension
    random.shuffle(fns) # shuffle the filenames to do not after that have in batches only same categories
    for fn in fns: # go throw the filenames
        img = np.array(io.imread(fn), dtype='float32') # load a concrete file and convert it to the appropriate format
        i_h, i_w = img.shape[:2] # get the image shape
        scaled = img / np.float32(255.0) #
        if i_h > 302: # two times bigger then Image for learning
            scaled = downscale_local_mean(scaled, factors=(2,2))  # to downscale the loaded images
        X.append(scaled) # add an downscaled image to the array
        Y.append(os.path.basename(fn).split('_')[0]) # get the class from the whole filename and add it to the Y file
    return X, Y # array of images(shuffled and probably downscaled) and their labels


def load_labels(dir_name, ext='.tif'): # fce for loading training data
    Y = [] # array for labels of the images in the input directory
    fns = glob.glob(dir_name+'/*' + ext) # load name of files from diven directory with given extension
    if(len(fns)==0): # because we used two different kinds of the file extension and we do not want to send it as an argument
        fns = glob.glob(dir_name + '/*' + '.tiff') # load name of files from diven directory with given extension
    random.shuffle(fns) # shuffle the filenames to do not after that have in batches only same categories
    for fn in fns: # go throw the filenames
        Y.append(os.path.basename(fn).split('_')[0]) # get the class from the whole filename and add it to the Y file
    return Y # array of the labels

def load_part_of_dir(dir_name, ext='.tif'): # fce for loading training data
    X, Y = [], [] # X is array for images, Y represents labels for the images
    fns = glob.glob(dir_name+'/*' + ext) # load name of files from diven directory with given extension
    if(len(fns)==0): # because we used two different kinds of the file extension and we do not want to send it as an argument
        fns = glob.glob(dir_name + '/*' + '.tiff') # load name of files from diven directory with given extension
    random.shuffle(fns) # shuffle the filenames to do not after that have in batches only same categories
    for fn in fns: # go throw the filenames
        countOfPicture = 0 # variable to count number of loaded images
        for yy in Y: # go throw the labels to load only concrete number of images from each category
            if os.path.basename(fn).split('_')[0]==yy: # find out the concrete category
                countOfPicture = countOfPicture+1 # count number of members from choosen category
                print('From this category one more picture')
        if countOfPicture<3: # add the image only if there is already zero or one image from this category
            img = np.array(io.imread(fn), dtype='float32') # load concrete picture
            i_h, i_w = img.shape[:2] # get shape of the loaded image
            scaled = img / np.float32(255.0) # get the right format
            if i_h > 302: # two times bigger then image for learning
                scaled = downscale_local_mean(scaled, factors=(2,2)) # downscaled the image with given
            X.append(scaled) # add the image to the array X
            Y.append(os.path.basename(fn).split('_')[0]) # add the label to the array Y
    return X, Y # return array of choosen images and labels

def augment_test_image(image, nb_rows, nb_cols, nb_patches, white):
    X = [] # array of test images
    patches = extract_patches_2d(image=image, # get patches from the images
                                 patch_size=(nb_rows, nb_cols),
                                 max_patches=nb_patches)
    
    for patch in patches: # go throw the patches and do operations for every single patch
        patch = augment.perturb(patch, NO_AUGMENTATION_PARAMS, target_shape=(nb_rows, nb_cols)) # augmentation of the patch with zero augmentation, it crop only the picture
        if white: # if we want to use the whitening method
            cov = np.dot(patch.T, patch) / patch.shape[1] # calculate the covarience matrix
            d, E, _ = np.linalg.svd(cov) # calculate the eigenvector E and eigenvalue d
            D = np.diag(1. / np.sqrt(E + EPS))
            W = np.dot(np.dot(d, D), d.T) # calculation of whitening matrix
            X_white = np.dot(patch, W) # application the whitening on the image
            X_white = X_white.reshape((1, X_white.shape[0], X_white.shape[1])) # reshape on appropriate shape for input to the CNN model
            X.append(X_white) # add whitened image to the array X
        else:
            patch = patch.reshape((1, patch.shape[0], patch.shape[1])) # reshape on appropriate shape for input to the CNN model
            X.append(patch) # add image to the array X
    
    return np.array(X, dtype='float32') # modify the type and add to the output array


def augment_train_images(images, categories, nb_rows, nb_cols, nb_patches, white): # augment the images using the AUGMENTATION_PARAMS
    print('augmenting train!')
    X, Y = [], [] # array for images(X) and for labels(Y)
    for idx, (image, category) in enumerate(zip(images, categories)): # go throw all images
        if idx % 500 == 0: # every 500 images print how much pictures were already loaded
            print('   >', idx)
        i_h, i_w = image.shape[:2] # get the shape of actual image
        if (i_h > 301) & (i_w>301):  # for using another size of patches for small pictures
            patches = extract_patches_2d(image=image, # get patches from actual image
                                     patch_size=(nb_rows * 2, nb_cols * 2),
                                     max_patches=nb_patches)
        elif (i_h > 221) & (i_w>221):
            patches = extract_patches_2d(image=image,  # get patches from actual image
                                     patch_size=(220, 220),
                                     max_patches=nb_patches)
        else:
            patches = extract_patches_2d(image=image, # get patches from actual image
                                         patch_size=(nb_rows, nb_cols),
                                         max_patches=nb_patches)

        for patch in patches: # go throw every single patch in patches
            patch = augment.perturb(patch, AUGMENTATION_PARAMS, target_shape=(nb_rows, nb_cols)) # applicate the augmentation with AUGMENTATION_PARAMS on concrete patch
            if white: # if was choosen whitening then will be run this part of script
                cov = np.dot(patch.T, patch)/patch.shape[1] # calculate the covarience matrix
                #   d = (lambda1, lambda2, ..., lambdaN)
                d, E, _ = np.linalg.svd(cov) # calculate the eigenvector E and eigenvalue d
                D = np.diag(1. / np.sqrt(E + EPS))
                W = np.dot(np.dot(d, D), d.T) # calculation of whitening matrix

                X_white = np.dot(patch, W) # application the whitening on the image
                X_white = X_white.reshape((1, X_white.shape[0], X_white.shape[1])) # reshape on appropriate shape for input to the CNN model
                X.append(X_white) # add whitened image to the array X
            else:
                patch = patch.reshape((1, patch.shape[0], patch.shape[1])) # reshape on appropriate shape for input to the CNN model
                X.append(patch) # add image to the array X
            Y.append(category) # add label to the array Y

    X = np.array(X, dtype='float32') # modify the type and add to the output array X for images
    Y = np.array(Y, dtype='int8') # modify the type and add to the output array Y for labels

    return X, Y # return the images array X and labels array Y



def plot_confusion_matrix(cm, target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.tick_params(labelsize=6)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=90)
    plt.yticks(tick_marks, target_names)
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black',
                 fontsize=5)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


