#
# in this sandbox im trying to test sequential training system
#
import numpy as np
from numpy import zeros
from random import randint
from random import random
from matplotlib import pyplot
import _pickle as pickle
import keras
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
# load data generator
from video_seq_generation import data_generator
from keras.layers import Dense, LSTM, Flatten, Dropout, TimeDistributed, GlobalAveragePooling2D, Input, BatchNormalization, Bidirectional
from keras.layers import Concatenate, Reshape, Conv2D, Activation, ZeroPadding2D, DepthwiseConv2D, GlobalMaxPooling1D, RepeatVector
# ---------------
import tensorflow as tf

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# config.gpu_options.allow_growth = True
# config.gpu_options.allocator_type = 'BFC'
# sess = tf.Session(config = config)


# arguments
data_dir = '/home/rapsodo/BASKETBALL/V_Data/'
data_name = 'video_ba_02_data.pickle'
label_name = 'video_ba_02_label.pickle'
with open(data_dir + data_name, 'rb') as f:
    gold_data = pickle.load(f)
with open(data_dir + label_name, 'rb') as f:
    gold_label = pickle.load(f)
    # splitting data
trX, tsX, trY, tsY = train_test_split(gold_data, gold_label)
del gold_data, gold_label
# TODO: 1 make labels as vectors
#
# TODO: 2 I will apply ball confirmation in order to recognize frames including ball
# generator for LSTM
# def data_generator(tX, tY, batch_size = 8):
#
#     data_x = []
#     data_y = []
#
#     while True:
#         for j in range(batch_size):
#             shot=[]
#             for i in range(len(tX[j])): # trX.shape[1]
#                 img = tX[j][i].astype(np.float32)
#                 shot.append(img)
#             shot = np.array(shot)
#             # shot = shot.reshape(40,224224,3)
#             # creating batch data and label
#             data_x.append(shot)
#             if tY[j] ==1:
#                 data_y.append(np.array([0, 1]))
#             else:
#                 data_y.append(np.array([1, 0]))
#         yield np.array(data_x), np.array(data_y)

# ----------------------
# Preparing the data for CNN
# ------------------------
# dsize = 100
# data =[]
# for j in range(dsize):
#     for i in range(len(trX[j])):
#         img = trX[j][i].astype(np.float32)
#         img = img.reshape(224, 224, 3)
#         data.append(img)
#
#
# data = np.array(data).reshape(-1,40,224,224,3)
# # del data
# data_y =[]
# for j in range(dsize):
#     if trY[j] ==1:
#         data_y.append(np.array([0, 1]))
#     else:
#         data_y.append(np.array([1, 0]))

# Generator for CNN-LSTM
def data_generator(tX, tY, batch_size = 8):

    data_x = []
    data_y = []

    while True:
        for j in range(batch_size):
            shot = []
            for i in range(len(tX[j])):
                img = tX[j][i].astype(np.float32)
                img = img.reshape(224, 224, 3)
                shot.append(img)
            data_x.append(np.array(shot))

            # creating batch data and label

            if tY[j] ==1:
                data_y.append(np.array([0, 1]))
            else:
                data_y.append(np.array([1, 0]))

        yield np.array(data_x), np.array(data_y)

# a,b = data_generator(tsX, tsY, batch_size = 8)

# batch_size = 8
# niter = tsX.shape[0] // batch_size
# data_x = []
# data_y =[]
# for i in range(niter):
#     for j in range(batch_size):
#         shot=[]
#         for i in range(len(tsX[j])): # trX.shape[1]
#             img = tsX[j][i].astype(np.float32)
#             shot.append(img)
#         shot = np.array(shot)
#         # shot = shot.reshape(40,224224,3)
#         # creating batch data and label
#         data_x.append(shot)
#         if tsY[j] ==1:
#             data_y.append(np.array([0, 1]))
#         else:
#             data_y.append(np.array([1, 0]))
#
# data_x = np.array(data_x)
# data_y = np.array(data_y)

#
# A simple network to test the yield system with our data
#
input = Input((None, 224,224, 3))

model_0 = Sequential()
model_0.add(Conv2D(512, (5, 5), strides=(2, 2), padding='same',
                   use_bias=False, input_shape=(224,224,3)))
model_0.add(BatchNormalization())
model_0.add(Activation('relu'))
model_0.add(ZeroPadding2D(padding=(1, 1)))
model_0.add(DepthwiseConv2D((3, 3), padding='valid', depth_multiplier=1, strides=(1, 1), use_bias=False))
model_0.add(BatchNormalization())
model_0.add(Activation('relu'))
model_0.add(Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False))
model_0.add(BatchNormalization())
model_0.add(Activation('relu'))
model_0.add(GlobalAveragePooling2D(data_format='channels_last'))
model_0.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_0.summary()
#
# Model 2
#
out2 = TimeDistributed(model_0)(input)

out = LSTM(1, input_shape=(40, 256))(out2)
out = Dense(10, activation='sigmoid') (out)
out = Dense(2, activation='sigmoid')(out)


model = Model( input, out)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

# print(model.summary())



# fit generator

train_gen = data_generator(trX[:50], trY[:50], batch_size = 2)
test_gen  = data_generator(tsX[:10], tsY[:10], batch_size = 2)


# model.fit(data, data_y, epochs=2)

# with tf.device('GPU:0'):
model.fit_generator(generator=train_gen,
                        steps_per_epoch=10,
                        epochs=50
    #                     # validation_data=test_gen,
    #                     # validation_steps=10
                        )