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
import _pickle as pickle
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
import coremltools
# fix random seed for reproductability
seed = 3
np.random.seed(seed)

# #
# --------------------------------------------------------------------------------------
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#
# Reading and splitting the data
data_dir = '/home/yizhuo/Documents/deep_learning/inout_classification/myvenv/'
data_name = 'sample_video_data.pickle'
label_name = 'sample_video_label.pickle'
with open(data_dir + data_name, 'rb') as f:
    gold_data = pickle.load(f)
with open(data_dir + label_name, 'rb') as f:
    gold_label = pickle.load(f)

# load pima indians dataset
# digits = load_digits()

gold_data = gold_data.reshape(-1,40)
#
trX, tsX, trY, tsY = train_test_split(gold_data, gold_label)
#
#
data_x =[]
for j in range(trX.shape[0]):
    frame=[]
    for i in range(trX.shape[1]):
        # reshaping image for feeding to the net
        try:
            img = trX[j][i].astype(np.float32).reshape(224,224,3)
        except:
            img = np.zeros((224,224,3))
        # img = tf.expand_dims(img.reshape(224,224,3),0)
        # out = artif_model(img)

        frame.append(img)
    print(' shot processed and added ')
    if not frame == []:
        frame = np.array(frame)
        data_x.append(frame)
data_x = np.array(data_x)
#
#
input = Input((None, 224,224, 3))
#
# model 1 the convolution layer
#
#
cnn_model = Sequential()
cnn_model.add(Conv2D(1024, (5, 5), strides=(2, 2), padding='same',
                 use_bias=False, input_shape=(224, 224, 3)))
cnn_model.add(BatchNormalization())
cnn_model.add(Activation('relu'))
cnn_model.add(ZeroPadding2D(padding=(1, 1)))
cnn_model.add(DepthwiseConv2D((3, 3), padding='valid', depth_multiplier=1, strides=(1, 1), use_bias=False))
cnn_model.add(BatchNormalization())
cnn_model.add(Activation('relu'))
cnn_model.add(Conv2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False))
cnn_model.add(BatchNormalization())
cnn_model.add(Activation('relu'))
cnn_model.add(ZeroPadding2D(padding=(1, 1)))
cnn_model.add(DepthwiseConv2D((3, 3), padding='valid', depth_multiplier=1, strides=(1, 1), use_bias=False))
cnn_model.add(BatchNormalization())
cnn_model.add(Activation('relu'))
cnn_model.add(Conv2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False))
cnn_model.add(BatchNormalization())
cnn_model.add(Activation('relu'))
cnn_model.add(ZeroPadding2D(padding=(1, 1)))
cnn_model.add(DepthwiseConv2D((3, 3), padding='valid', depth_multiplier=1, strides=(1, 1), use_bias=False))
cnn_model.add(BatchNormalization())
cnn_model.add(Activation('relu'))
cnn_model.add(Conv2D(512, (1, 1), strides=(2, 1), padding='same', use_bias=False))
cnn_model.add(BatchNormalization())
cnn_model.add(Activation('relu'))
cnn_model.add(GlobalAveragePooling2D(data_format='channels_last'))
cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn_model.summary()
#
# Model 2
#
out2 = TimeDistributed(cnn_model)(input)

# input2 = RepeatVector(40)(out2)

out = Bidirectional(LSTM(units=512, input_shape=(None, 512), return_sequences=True,
                         kernel_initializer="uniform", activation='relu',
                         dropout=.2, forget_bias_init='one',
                         inner_activation='sigmoid', go_backwards=True))(out2)
out =  Bidirectional(LSTM(units=512, return_sequences=True, kernel_initializer="uniform",
                          activation='relu', go_backwards=True,
                          forget_bias_init='one', inner_activation='sigmoid', dropout=.2))(out)
out =  Bidirectional(LSTM(units=256, return_sequences=True, kernel_initializer="uniform",
                          dropout=.2, forget_bias_init='one', inner_activation='sigmoid',
                          activation='relu', go_backwards=True))(out)
out = LSTM(units=50, input_shape=(None, 1024), return_sequences=False, kernel_initializer="uniform",
           forget_bias_init='one', inner_activation='sigmoid',
           dropout=.2, activation='relu')(out)
out = Dropout(rate=.4)(out)
out = Dense(50, kernel_initializer="uniform", activation='tanh')(out)
out = Dropout(rate=.4)(out)
output = Dense(1, kernel_initializer="uniform", activation='softmax')(out)
#
model = Model(input, output)
#
print('saving training checkpoints as hdf5')
# NOTE: hd just saves the weights!
file_path = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# hyperparameters
from keras.optimizers import SGD
epochs = 10
learning_rate = 10
decay_rate = learning_rate/epochs
momentum = .99
sgd = SGD(lr = learning_rate, momentum=momentum, decay= decay_rate, nesterov= True)
#
# compiling the model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# evaluating with the model
model.summary()
# with tf.device('/GPU:0'):
history = model.fit(data_x, trY, validation_split=.1, callbacks=callbacks_list,  epochs=epochs, batch_size=2, verbose=0) #
# evaluation
scores = model.evaluate(tsX, tsY, verbose=0) #
# printing
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Serializing model to jason
model_jason = model.to_json()
with open('model_train_01.json', 'w') as j_file:
    j_file.write(model_jason)
# Serialize weights to HDF5
model.save_weights('model_train_01_w.h5')
# #
# model.save('complete_model.h5')
# # now the turn is loading the model and creating it
# print("[INFO] loading model...")
# model = load_model(args["model"])

# Serializing model to jason
js_file = open('model_train_01.json','r')
loaded_json_model = js_file.read()
js_file.close()
# loaded_model = model_from_json(loaded_json_model)
#
from keras.applications.mobilenet import mobilenet
from keras.utils.generic_utils import CustomObjectScope
from keras.models import model_from_json
# import keras.applications.mobilenet.mobilenet._depthwise_conv_block as DepthwiseConv2D
# mobilenet.mobilenet.
# ,'DepthwiseConv2D': mobilenet.DepthwiseConv2D
with CustomObjectScope({'relu6': mobilenet.mobilenet.relu6}):
    loaded_model = model_from_json(loaded_json_model)
    # allmodel = load_model('complete_model.h5')
    # loading weights to the new model
    loaded_model.load_weights('model_train_01_w.h5')
#
coreml_model = coremltools.converters.keras.convert(loaded_model)
coreml_model.save('coreml_train_01.mlmodel')
# -----------------------------------------------------------------------------


