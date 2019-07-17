'''Training the model
    Pretraining the model
    Fine-tune the model
    Training with whitening
'''
from __future__ import print_function

import random
import shutil
import pickle
import gc
import csv
import os
import sys


sys.setrecursionlimit(10000)
SEED = 1066987
import numpy as np

np.random.seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.misc import imsave
from keras.utils import np_utils
from DeepScript import utilsTIFF, vgg16test
from DeepScript import resnet50, vgg16
from keras import backend as K



NB_ROWS, NB_COLS = 150, 150
BATCH_SIZE = 30
NB_EPOCHS = 100
NB_TRAIN_PATCHES = 100
NB_TEST_PATCHES = 30
MODEL_NAME = 'resnet50'  # name of the model e.g. ResNet50White
MODEL_TYPE = 'resnet50'  # resnet50 or vgg16
PRINT_ERROR = True





def trainGeneral(mname, mtype, way, fineTune=False, mnamePre='none',whitening=False):  # define function for training the neuronal network
    K.set_image_dim_ordering('th')
    # train_images, train_categories = utils.load_dir(os.getcwd()+'\\data\\splits\\train') # load from local computer
    # dev_images, dev_categories = utils.load_dir(os.getcwd()+'\\data\\splits\\dev') # load from local computer
    train_images, train_categories = utilsTIFF.load_dir(os.getcwd() + way + '/train')  # load from server
    print('train caterories')
    print(train_categories)
    print(os.getcwd() + way + '/train')
    dev_images, dev_categories = utilsTIFF.load_dir(os.getcwd() + way +'/dev')
    all_acc = []  # list for each categories

    try:
        os.mkdir('models')
    except:
        pass

    try:
        shutil.rmtree('models/' + mname)
    except:
        pass

    os.mkdir('models/' + mname)

    label_encoder = LabelEncoder().fit(train_categories)
    train_y_int = label_encoder.transform(train_categories)
    dev_y_int = label_encoder.transform(dev_categories)
    nb_classes = len(label_encoder.classes_)
    train_Y = np_utils.to_categorical(train_y_int, nb_classes)

    print('-> Working on', len(label_encoder.classes_), 'classes:', label_encoder.classes_)

    pickle.dump(label_encoder, open('models/' + mname + '/label_encoder.p', 'wb'))
    if (fineTune):
        if mtype == 'resnet50':
            model = resnet50.ResNet50(weights=mnamePre,
                                      nb_classes=len(label_encoder.classes_),
                                      nb_rows=NB_ROWS,
                                      nb_cols=NB_COLS)
        elif mtype == 'vgg16':
            model = vgg16test.VGG16(nb_classes=len(label_encoder.classes_), # zkusit testnout
                                nb_rows=NB_ROWS,
                                nb_cols=NB_COLS,
                                nb_classes_pretrain=mnamePre)
        else:
            raise ValueError('Unsupported model type: ' + mtype)
    else:
        if mtype == 'resnet50':
            model = resnet50.ResNet50(weights=None,
                                      nb_classes=len(label_encoder.classes_),
                                      nb_rows=NB_ROWS,
                                      nb_cols=NB_COLS)
        elif mtype == 'vgg16':
            model = vgg16.VGG16(nb_classes=len(label_encoder.classes_),
                                nb_rows=NB_ROWS,
                                nb_cols=NB_COLS)
        else:
            raise ValueError('Unsupported model type: ' + mtype)

    model.summary()
    print(model.summary())

    with open('models/' + mname + '/architecture.json', 'w') as F:
        F.write(model.to_json())

    best_dev_acc = 0.0
    all_acc.append(best_dev_acc)
    print('-> building dev inputs once:')
    dev_inputs = []
    for idx, img in enumerate(dev_images):
        print('imageSizeVsechno: ' + str(img.shape))

        i = utilsTIFF.augment_test_image(image=img,
                                     nb_rows=NB_ROWS,
                                     nb_cols=NB_COLS,
                                     nb_patches=NB_TEST_PATCHES,
                                     white=whitening)
        dev_inputs.append(i)
    saved_before=0 # before how much epochs was the model saved
    for e in range(NB_EPOCHS):
        saved_before=saved_before+1
        tmp_train_X, tmp_train_Y = utilsTIFF.augment_train_images(images=train_images,
                                                              categories=train_Y,
                                                              nb_rows=NB_ROWS,
                                                              nb_cols=NB_COLS,
                                                              nb_patches=NB_TRAIN_PATCHES,
                                                                  white=whitening)

        print('epocha:' + str(e))

        for idx, p in enumerate(tmp_train_X):
            p = p.reshape((p.shape[1], p.shape[2]))
            imsave('models/' + mname + '/' + str(idx) + '.png', p)  # save quadratisch picture
            if idx >= 30:
                break
        model.fit(tmp_train_X, tmp_train_Y,
                  batch_size=BATCH_SIZE,
                  nb_epoch=1,
                  shuffle=True)

        dev_preds = []
        for inp in dev_inputs:
            pred = model.predict(inp, batch_size=BATCH_SIZE)
            pred = pred.mean(axis=0)
            dev_preds.append(np.argmax(pred, axis=0))

        # calculate accuracy:
        curr_acc = accuracy_score(dev_preds, dev_y_int)
        print('  curr val acc:', curr_acc)

        # save weights, if appropriate:
        if curr_acc > best_dev_acc:
            print('    -> saving model')
            model.save_weights('models/' + mname + '/weights.hdf5',
                               overwrite=True)
            best_dev_acc = curr_acc
            saved_before=0

        # print eror graph
        if PRINT_ERROR:
            x = np.arange(e + 1)
            all_acc.append(curr_acc)
            nameOfFile = 'Accurancy' + mname + '.csv'
            with open(nameOfFile, 'a', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=' ',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(all_acc)

        # half learning rate:
        if e and e % 10 == 0:
            old_lr = K.get_value(
                model.optimizer.lr)  # solution from https://github.com/fchollet/keras/issues/898 marc-moreaux commented on 13 Mar • edited
            new_lr = old_lr * 0.5
            K.set_value(model.optimizer.lr, new_lr)
            print('\t- Lowering learning rate > was:', old_lr, ', now:', new_lr)
        if saved_before>10:
            e=NB_EPOCHS+1


if __name__ == '__main__':
    #name of file and also way to file, type of model, way to learning dat
    arguments=sys.argv
    print(arguments)
    if(len(arguments)==4):
        # name of file and also way to file, type of model, way to learning data
        trainGeneral(arguments[1],arguments[2], arguments[3])
    elif(len(arguments)==5):
        # name of file and also way to file, type of model, way to learning data, False, 'none', whitening
        trainGeneral(arguments[1], arguments[2], arguments[3], False, 'none', arguments[4])
    elif (len(arguments)==6):
        # name of file and also way to file, type of model, way to learning data, boolean finetune (True=fineTune, False=train), name of the file with pretrain weights
        trainGeneral(arguments[1], arguments[2], arguments[3], arguments[4], arguments[5])
    else:
        print('wrong number of inputs!!!')
    gc.collect()
