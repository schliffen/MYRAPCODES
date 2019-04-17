#
# The goal of this dicument is to load saved model and retrain it to make it work under coreML
#
#
#
#
# import the necessary packages
import argparse
import keras
import os
import cv2
import random
from IPython.display import clear_output
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import array_to_img, img_to_array, load_img
#
# from pyimagesearch.preprocessing import ImageToArrayPreprocessor
# from pyimagesearch.preprocessing import AspectAwarePreprocessor
# from pyimagesearch.datasets import SimpleDatasetLoader
# from pyimagesearch.nn.conv import MiniVGGNet
# --------------
from keras.optimizers import SGD
from imutils import paths
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
# --------------
from keras import optimizers
from keras.applications.mobilenet import MobileNet
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout, TimeDistributed, GlobalAveragePooling2D, Input, BatchNormalization, Bidirectional
from keras.layers import Concatenate, Reshape, Conv2D, Activation, ZeroPadding2D, DepthwiseConv2D, GlobalMaxPooling1D, RepeatVector, MaxPooling2D
# --------------
from keras.preprocessing.image import  ImageDataGenerator
train_datagen = ImageDataGenerator(rotation_range=30, width_shift_range=.1,
                                   height_shift_range=.1, shear_range=.2, zoom_range=.2, horizontal_flip=True,
                                   fill_mode='nearest', zca_whitening=True)
validation_datagen = ImageDataGenerator(rescale=1./255)
# -------------
ap = argparse.ArgumentParser()
ap.add_argument("-dt", "--trn_data", default='/home/ali/BASKETBALL_DATA/ball_noball/train/',
                help="path to train dataset")
ap.add_argument("-dv", "--val_data", default='/home/ali/BASKETBALL_DATA/ball_noball/validation/',
                help="path to val dataset")
ap.add_argument("-m", "--mdl", default='mobnet_ball_fin_15',
                help="path to val dataset")
ap.add_argument("-d", "--ddir", default='./models/',
                help="path to val dataset")
# ap.add_argument('-bd', '--bdr', default='/home/ali/BASKETBALL_DATA/ball_noball/', help='ball no ball data')
# -------------
args = ap.parse_args()
#
# loading the model here
# Serializing model to jason
js_file = open(args.ddir + args.mdl +'.json','r')
loaded_json_model = js_file.read()
js_file.close()
# loaded_model = model_from_json(loaded_json_model)
#
# # loading weights to the new model
# # loaded_model.load_weights('model_sl01_w.h5')
# #
#
from keras.applications import mobilenet
from keras.utils.generic_utils import CustomObjectScope
from keras.models import model_from_json
# from keras.layers.advanced_activations import ReLU
# import keras.applications.mobilenet.mobilenet._depthwise_conv_block as DepthwiseConv2D
# mobilenet.mobilenet.
# ,'DepthwiseConv2D': mobilenet.DepthwiseConv2D
with CustomObjectScope({'relu6': mobilenet.mobilenet.relu6}):
    loaded_model = model_from_json(loaded_json_model)
    # allmodel = load_model('complete_model.h5')
    # loading weights to the new model
    loaded_model.load_weights(args.ddir + args.mdl + '.h5')


loaded_model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])

weigth_file="./weights/mobnet_ball_fin_15-17.hdf5"
checkpoint = ModelCheckpoint(weigth_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tbCallBack = TensorBoard(log_dir='./Graph/mobnet_ball_fin_15', histogram_freq=0, write_graph=True, write_images=True)
callbacks_list = [checkpoint,tbCallBack]
#

train_generator = train_datagen.flow_from_directory(
    args.trn_data,
    target_size=(68, 68),
    batch_size=32,
    color_mode='rgb',
    shuffle=True,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    args.val_data,
    target_size=(68, 68),
    batch_size=16,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True)

#
# Train the model
history = loaded_model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples/train_generator.batch_size ,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples/validation_generator.batch_size,
    callbacks=callbacks_list,
    verbose=1)

# saving the model
model_json = loaded_model.to_json()
with open("./models/mobnet_ball_fin_15.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
loaded_model.save_weights("./models/mobnet_ball_fin_15.h5")
print("Saved model to disk")

print("Saved model to disk")
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()