'''Test the model
    with or without whitening
'''
from __future__ import print_function

SEED = 7687655
import random
import os
import pickle
import glob
import sys

sys.setrecursionlimit(10000)

import numpy as np

np.random.seed(SEED)
random.seed(SEED)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid", {'axes.grid': False})

from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.transform import downscale_local_mean
from skimage import io
from keras.models import model_from_json
from DeepScript import utilsTIFF

NB_ROWS, NB_COLS = 150, 150
MODEL_NAME = 'final'
NB_TEST_PATCHES = 30


def testGeneral(mname, way, whitening=False):
    test_images, test_categories = [], []
    for fn in glob.glob(way + '/*.tif'):
        img = np.array(io.imread(fn), dtype='float32')
        scaled = img / np.float32(255.0)
        scaled = downscale_local_mean(scaled, factors=(2, 2))
        test_images.append(scaled)
        test_categories.append(os.path.basename(fn).split('_')[0])

    label_encoder = pickle.load(open('models/' + mname + '/label_encoder.p', 'rb'))
    print('-> Working on classes:', label_encoder.classes_)

    test_int_labels = label_encoder.transform(test_categories)

    model = model_from_json(open('models/' + mname + '/architecture.json').read())

    model.load_weights('models/' + mname + '/weights.hdf5')

    test_preds = []

    for img in test_images:
        X = utilsTIFF.augment_test_image(image=img,
                                         nb_rows=NB_ROWS,
                                         nb_cols=NB_COLS,
                                         nb_patches=NB_TEST_PATCHES,
                                        white=whitening)
        preds = np.array(model.predict(X), dtype='float64')
        av_pred = preds.mean(axis=0)
        test_preds.append(np.argmax(av_pred, axis=0))

    print('Test accuracy:', accuracy_score(test_int_labels, test_preds))

    # confusion matrix
    plt.clf()
    T = label_encoder.inverse_transform(test_int_labels)
    P = label_encoder.inverse_transform(test_preds)
    cm = confusion_matrix(T, P, labels=label_encoder.classes_)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)
    sns.plt.figure()
    utilsTIFF.plot_confusion_matrix(cm_normalized, target_names=label_encoder.classes_)
    sns.plt.savefig('models/' + mname + '/test_conf_matrix.pdf')


if __name__ == '__main__':
    arguments = sys.argv
    if (len(arguments) == 3):
        # name of file and also way to file, type of model, way to learning data
        testGeneral(arguments[1], arguments[2])
    elif (len(arguments) == 4):
        # name of file and also way to file, type of model, way to learning data, boolean finetune (True=fineTune, False=train), name of the file with pretrain weights
        testGeneral(arguments[1], arguments[2], arguments[3])
    else:
        print('wrong number of inputs!!!')
    print(arguments)