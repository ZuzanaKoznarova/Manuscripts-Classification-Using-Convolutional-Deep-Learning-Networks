"""
Module to preprocess the ICDAR17 data:
    - create a train/dev split from the training data and preprocess these
    -for using the right dataset set Parameters CSV_NAME, DATA_SET_NAME
"""


from __future__ import print_function

import os
import shutil

SEED = 1066987
import random
import numpy as np
from PIL import Image
import csv

np.random.seed(SEED)
random.seed(SEED)
# set name of the CSV file where are available the names and theirs labels
CSV_NAME='icdar17color_labels_train.txt'
#CSV_NAME='icdar17_labels_test.txt'
#CSV_NAME='icdar17_labels_both.txt'
# set name of the part of the dataset
DATA_SET_NAME='train'
#DATA_SET_NAME='test'
#DATA_SET_NAME='both'
from sklearn.cross_validation import train_test_split
    

def train_dev_split(input_dir = os.getcwd()+'\\data\\icdar17\\dataset\\test',
#os.getcwd()+'data/CLaMM_train',
                    metafile = CSV_NAME,
                    target_classes = None,
                    test_size = 0.2,
                    random_state = 863863,
                    output_path = os.getcwd()+'\\data\\icdar17',
                    cut=False):

    print('Creating train / dev split for data from:', input_dir)
    if target_classes:
        print('\t-> restricting to classes:', target_classes)

    filenames, categories, categories2 = [], [], []

    for line in open(os.sep.join((input_dir, metafile)), 'r'):

        line = line.strip()


        if line and not line.startswith('FileName'):
            filename, category= line.split(' ')
            category = category.lower().replace('_', '-')
            if target_classes:
                if category not in target_classes:
                    continue

                if not os.path.exists(os.sep.join((input_dir, filename))):
                    raise ValueError('%s not found!' % (filename))

            filenames.append(filename)
            categories.append(category)

    print('categories:', sorted(set(categories)))

    if cut:
        # for using the in thesis described cutting method(2x4 images, width x height 224x224, gray scale, tiff)
        CNTW = 2
        CNTH = 4
        all_names = filenames
        filenames = [] #rewrite me the file which this order
        categories=[]
        file_name_and_category=[]
        width_new = 302
        height_new = 302
        for f in all_names:
            image_name = f
            print(f)
            img = Image.open(input_dir + '\\' + image_name).convert('L')
            width_original, height_original = img.size
            for x in range(0, CNTW):
                for y in range(0, CNTH):
                    left = round((width_original - (CNTW * width_new)) / 1.5) + x * width_new
                    top = round((height_original - CNTH * height_new) / 5) + y * height_new
                    if ((left + width_new) < width_original and (top + height_new) < height_original) and (top + height_new) > 0 and ((left + width_new)) > 0 and top > 0 and left > 0:
                        box = (left, top, left + width_new, top + height_new)
                        area = img.crop(box)
                        name = image_name.split('.')[0] + str(x) + str(y) + '.tiff'
                        area.save(output_path + '\\' + name, 'tiff')
                        category = image_name.split('-')[0]
                        categories.append(category)
                        filenames.append(name)
                        file_name_and_category.append(name + ';' + image_name.split('-')[0])

        name_of_file = 'labels.csv'
        with open(output_path + '\\' + name_of_file, 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter='\n',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(file_name_and_category)
    else: # classic preprocesing without cutting (gray scale, tiff)
        all_names = filenames
        filenames = []
        categories = []
        file_name_and_category = []
        for f in all_names:
            image_name = f
            print(f)
            img = Image.open(input_dir + '\\' + image_name).convert('L')
            name = image_name.split('.')[0] + 'gray' + '.tiff'
            img.save(output_path + '\\' + name, 'tiff')
            category = image_name.split('-')[0]
            categories.append(category)
            filenames.append(name)
            file_name_and_category.append(name + ';' + image_name.split('-')[0])

        name_of_file = 'labels.csv'
        with open(output_path + '\\' + name_of_file, 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter='\n',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(file_name_and_category)



    assert len(filenames) == len(categories)

    print('\t-> splitting', len(filenames), 'items in total')
    #here it start to split
    train_fns, dev_fns, train_categories, dev_categories = \
            train_test_split(filenames, categories,
                             test_size=test_size,
                             random_state=random_state,
                             stratify=categories)

    print('\n# train images:', len(train_fns))
    print('# dev images:', len(dev_fns))

    # make new directories for dev, train dataset, if they do not exist and save there the dev, train images
    input_dir = output_path
    if not os.path.exists(os.sep.join((output_path, 'train'))):
        os.mkdir(os.sep.join((output_path, 'train')))
    else:
        print('INFO: File train already exist, be careful, maybe you rewrite now your already splited data')
    if not os.path.exists(os.sep.join((output_path, 'dev'))):
        os.mkdir(os.sep.join((output_path, 'dev')))
    else:
        print('INFO: File dev already exist, be careful, maybe you rewrite now your already splited data')

    for fn, category in zip(train_fns, train_categories):
        in_ = os.sep.join((input_dir, fn))
        out_ = os.sep.join((output_path, 'train', category + '_' + fn))
        shutil.copyfile(in_, out_)

    for fn, category in zip(dev_fns, dev_categories):
        in_ = os.sep.join((input_dir, fn))
        out_ = os.sep.join((output_path, 'dev', category + '_' + fn))
        shutil.copyfile(in_, out_)



if __name__ == '__main__':

    # which classes
    target_classes = None

    # create path for the train, dev splits
    output_path = os.path.dirname(os.getcwd() + '\\data\\icdar17\\dataset\\icdar17cernobileRozdeleno302x302\\' + DATA_SET_NAME )
    try:
        shutil.rmtree(output_path)
    except:
        pass

    os.mkdir(output_path)

    # create train-dev split:
    train_dev_split(input_dir=os.getcwd()+'\\data\\icdar17\\dataset\\'+DATA_SET_NAME, #change only train, test or both
                    test_size = 0.2,
                    random_state = SEED,
                    output_path = output_path,
                    target_classes = target_classes,
                    cut=True)

