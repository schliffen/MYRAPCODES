#
# This script is for design deep learning for new in out
#
import os, sys, glob
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
#
import cv2
import argparse
import keras
import autokeras
import numpy as np
import tensorflow as tf
from keras import applications
from keras.models import load_model
from autokeras.utils import pickle_from_file

import os.path as osp
from sklearn.metrics import accuracy_score
import datetime

d = datetime.datetime.today()
uid = d.isoformat()

ap = argparse.ArgumentParser()
ap.add_argument("-dt", "--trn_data", default='/home/rapsodo/BASKETBALL/00_DATASETS/BallConfirmationData/Training/',
                help="path to train dataset")
ap.add_argument("-dv", "--val_data", default='/home/rapsodo/BASKETBALL/00_DATASETS/BallConfirmationData/Testing/',
                help="path to val dataset")
ap.add_argument("-dm", "--model", default='./models/',
                help="path to val dataset")

args = ap.parse_args()

from autokeras  import ImageClassifier

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                                   rotation_range=15, width_shift_range=.3,
                                   height_shift_range=.3, shear_range=.3,
                                   zoom_range=.2, horizontal_flip=True, vertical_flip=True,
                                   # zca_whitening=False, zca_epsilon=1e-3,
                                   rescale=1./255, fill_mode='nearest'
                                   )
validation_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                                   rotation_range=15, width_shift_range=.3,
                                   height_shift_range=.3, shear_range=.3,
                                   zoom_range=.2, horizontal_flip=True, vertical_flip=True,
                                   # zca_whitening=False, zca_epsilon=1e-3,
                                   rescale=1./255, fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    args.trn_data,
    target_size=(40, 40),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)
validation_generator = validation_datagen.flow_from_directory(
    args.val_data,
    target_size=(40, 40),
    batch_size=256,
    class_mode='categorical',
    shuffle=True)



# from keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

trdata_X = []
trdata_Y = []
cntr = 0
for image in train_generator:
    plt.imshow(image[0][0])
    trdata_X.append(image[0][0])
    if image[1][0][0] == 1:
        trdata_Y.append(1)
    else:
        trdata_Y.append(0)
    cntr +=1
    if cntr == 4000:
        break

tsdata_X = []
tsdata_Y = []
cntr = 0
for image in train_generator:
    # plt.imshow(image[0][0])
    tsdata_X.append(image[0][0])
    if image[1][0][0] == 1:
        tsdata_Y.append(1)
    else:
        tsdata_Y.append(0)
    cntr +=1
    if cntr == 1000:
        break

# trdata_X = np.array(trdata_X)
# trdata_Y = np.array(trdata_Y)
tsdata_X = np.array(tsdata_X)
tsdata_Y = np.array(tsdata_Y)

if __name__ == '__main__':

    # loading keras model and autokeras model
    # 1 - loading autokeras model
    # ap.model + 'ball_autokeras_model_1_0.h5'
    auto_model = pickle_from_file(args.model + 'ball_autokeras_model_1_0.h5')
    # 2 - loading autokeras model
    # keras model does not work properly
    keras_model = load_model(args.model + 'ball_keras_model_1_1.h5')



    y = auto_model.evaluate(tsdata_X, tsdata_Y)
    print(y * 100)

    y = keras_model.evaluate(tsdata_X, tsdata_Y)
    print(y * 100)

    # Evaluating accuracy with sklearn

    y_prediction = auto_model.predict(tsdata_X)
    sk_acc_measure = accuracy_score(y_true=tsdata_Y, y_pred=y_prediction)
    print('evaluating with sklearn: ', sk_acc_measure)



    y_prediction = keras_model.predict(tsdata_X)
    sk_acc_measure = accuracy_score(y_true=tsdata_Y, y_pred=y_prediction)
    print('evaluating with sklearn: ', sk_acc_measure)




