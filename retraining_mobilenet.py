#
# The goal of this script is to retrain mobilenet again and save it for later use
#
#
# save and load practice
#
import os, sys
import tensorflow as tf
import matplotlib
# matplotlib.use('Agg') # this is for keeping matplotlib at Agg mode
#
import matplotlib.pyplot as plt
import numpy as np
import keras
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras import applications
from keras.layers import Dense, LSTM, Flatten, Dropout, TimeDistributed, GlobalAveragePooling2D, Input, BatchNormalization, Bidirectional
from keras.layers import Concatenate, Reshape, Conv2D, Activation, ZeroPadding2D, DepthwiseConv2D,  RepeatVector, Conv3D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing import image
import argparse
# fix random seed for reproductability
seed = 3
np.random.seed(seed)
argp = argparse.ArgumentParser()
#<<<<<<< HEAD
argp.add_argument('-d', '--ddir', default='/home/ali/BASKETBALL_DATA/testing_classification', help='direction of testing data')
argp.add_argument('-md', '--mdir', default='/home/ali/CLionProjects/rapsodo_ball_classification/', help='direction of main directory')
argp.add_argument('-m', '--mdl', default='models/mobnet_in-out_002', help='direction of testing data')
argp.add_argument('-l', '--lbl', default= ['in', 'out'])

args = argp.parse_args()
# #
# --------------------------------------------------------------------------------------
#
#
X, y = load_digits(n_class=10, return_X_y=True)
trX, tsX, trY, tsY = train_test_split(X,y, test_size=.1)
# introducing the mobilenet
#applications.mobilenet.MobileNet()
monet  = applications.mobilenet.MobileNet(weights='imagenet', include_top= False, dropout=.3, input_shape=(128,128,3))
# selecting trainable layers
for layer in monet.layers[-15:]:
    layer.trainable = True
#
input = Input((128, 128, 3))
#
# model 1 the convolution layer
#
#
# cnn_model = Sequential()
# cnn_model.add(Conv2D(1024, (5, 5), strides=(2, 2), padding='same',
#                      use_bias=False, input_shape=(128, 128, 3)))
# cnn_model.add(BatchNormalization())
# cnn_model.add(Activation('relu'))
# cnn_model.add(ZeroPadding2D(padding=(1, 1)))
# cnn_model.add(DepthwiseConv2D((3, 3), padding='valid', depth_multiplier=1, strides=(1, 1), use_bias=False))
# cnn_model.add(BatchNormalization())
# cnn_model.add(Activation('relu'))
# cnn_model.add(Conv2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False))
# cnn_model.add(BatchNormalization())
# cnn_model.add(Activation('relu'))
# cnn_model.add(ZeroPadding2D(padding=(1, 1)))
# cnn_model.add(DepthwiseConv2D((3, 3), padding='valid', depth_multiplier=1, strides=(1, 1), use_bias=False))
# cnn_model.add(BatchNormalization())
# cnn_model.add(Activation('relu'))
# cnn_model.add(Conv2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False))
# cnn_model.add(BatchNormalization())
# cnn_model.add(Activation('relu'))
# cnn_model.add(ZeroPadding2D(padding=(1, 1)))
# cnn_model.add(DepthwiseConv2D((3, 3), padding='valid', depth_multiplier=1, strides=(1, 1), use_bias=False))
# cnn_model.add(BatchNormalization())
# cnn_model.add(Activation('relu'))
# cnn_model.add(Conv2D(512, (1, 1), strides=(2, 1), padding='same', use_bias=False))
# cnn_model.add(BatchNormalization())
# cnn_model.add(Activation('relu'))
# cnn_model.add(GlobalAveragePooling2D(data_format='channels_last'))
# cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# cnn_model.summary()
#
# # the model
#
#out2 = cnn_model(input)

# out = monet(input)
# out = GlobalAveragePooling2D(data_format='channels_last')(out)
# output = Dense(2, kernel_initializer="uniform", activation='softmax')(out)
# model = Model(input, output)

model = Sequential()
model.add(Dense(32, input_shape=(128,128), activation='relu'))
model.add(Flatten())
model.add(Dense(1, kernel_initializer="uniform", activation='softmax'))
#
# compiling the model/
#
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
model.summary()
#do training here!

# model = creat_model()
#
print('saving training checkpoints as hdf5')
# NOTE: hd just saves the weights!
file_path = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# hyperparameters
epochs = 10
#
# =---------------------------- creating data generator --------------------------
from keras.preprocessing.image import  ImageDataGenerator
batch_size = 2
train_datagen = ImageDataGenerator(rescale=2./255)
test_datagen = ImageDataGenerator(rescale=2./255)
train_generator = train_generator = test_datagen.flow_from_directory(
    directory=args.ddir,
    target_size=(128, 128),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
test_generator = test_generator = test_datagen.flow_from_directory(
    directory=args.ddir,
    target_size=(128, 128),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

# with tf.device('/GPU:0'):
history = model.fit_generator(generator=train_generator,
                              validation_data= test_generator,
                              callbacks=callbacks_list,
                              epochs=epochs,
                              verbose=0) #
# evaluation
scores = model.evaluate(X, y, verbose=0) #
# printing
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# creating newly trained mobilenet
final_model = Model(input, model.layers[-2].output)
final_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Serializing model to jason
model_jason = final_model.to_json()
with open('final_model_train_01.json', 'w') as j_file:
    j_file.write(model_jason)
# Serialize weights to HDF5
final_model.save_weights('final_model_train_01_w.h5')






# # ------------------------------------- Until here!
# model.save('complete_model.h5')
# # now the turn is loading the model and creating it
# print("[INFO] loading model...")
# model = load_model(args["model"])

# Serializing model to jason
# js_file = open('model_train_01.json','r')
# loaded_json_model = js_file.read()
# js_file.close()
# # loaded_model = model_from_json(loaded_json_model)
# #
# from keras.applications.mobilenet import mobilenet
# from keras.utils.generic_utils import CustomObjectScope
# from keras.models import model_from_json
# # import keras.applications.mobilenet.mobilenet._depthwise_conv_block as DepthwiseConv2D
# # mobilenet.mobilenet.
# # ,'DepthwiseConv2D': mobilenet.DepthwiseConv2D
# with CustomObjectScope({'relu6': mobilenet.mobilenet.relu6}):
#     loaded_model = model_from_json(loaded_json_model)
#     # allmodel = load_model('complete_model.h5')
#     # loading weights to the new model
#     loaded_model.load_weights('model_train_01_w.h5')
# #
# coreml_model = coremltools.converters.keras.convert(loaded_model)
# coreml_model.save('coreml_train_01.mlmodel')
# -----------------------------------------------------------------------------

