#
# full LSTM version 2
#
import os, sys, time, datetime
# from lstm_network import LSTM_RNN_Network
# from video_seq_generation import vid_sample
import tensorflow as tf
import pickle
import argparse
import matplotlib.pyplot as plt
from random import random, randint
from numpy import array, zeros
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
from keras.layers import Concatenate, Reshape, Conv2D, Activation, ZeroPadding2D, DepthwiseConv2D, GlobalMaxPooling1D, RepeatVector, MaxPooling2D
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing import image
import coremltools

# import matplotlib
# matplotlib.use('Agg')
#
# Setting argumant parser
ap = argparse.ArgumentParser()
ap.add_argument('-d','--logdir', type=str, default='/home/ali/CLionProjects/p_01/report/log_dir', help='direction to tensorboard log')
ap.add_argument('-v','--vidir', type=str, default='/home/ali/CLionProjects/p_01/unsorted_data/v1', help='direction to video files')
ap.add_argument('-s','--source', default='/home/ali/CLionProjects/p_01/Vid_data/vid_01/',   help='Path to the source metadata file')
ap.add_argument('--stopwords_file', default='/home/ali/CLionProjects/p_01/report/stopwords.txt', help='Path to stopwords file')
ap.add_argument('--summaries_dir', default='/home/ali/CLionProjects/p_01/report/summaries_dir/', help='Path to stopwords file')
ap.add_argument('-m','--model', default='MobileNet', nargs="?",
                # type=named_model,
                help='Name of the pre-trained model to use')
ag = ap.parse_args()

# creating sample videos -----------------------------------------------
#
data_dir = '/home/ali/CLionProjects/in_out_detection/data/'
data_name = 'sample_video_data.pickle'
label_name = 'sample_video_label.pickle'
#
label_name = 'sample_video_label.pickle'
# with open(data_dir + data_name, 'rb') as f:
#     gold_data = pickle.load(f)
# with open(data_dir + label_name, 'rb') as f:
#     gold_label = pickle.load(f)


# gold_data = gold_data.reshape(-1,40)
#
# trX, tsX, trY, tsY = train_test_split(gold_data, gold_label)
#
# data_x = []
# for j in range(trX.shape[0]):
#
#     frame=[]
#
#     for i in range(trX.shape[1]):
#
#         img = trX[j][i].astype(np.float32)
#
#         frame.append(img)
#
#     frame = np.array(frame)
#     data_x.append(frame)


# configure problem
size = 50
# define the model
def ls_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(2, 2,2, activation='relu'),
                              input_shape=(None,size, size,1)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))
    return model

model = ls_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())


# fit model
# vsamp = vid_sample(size=size)
# X, y = vsamp.generate_examples(500)
# doing validation split
# model.fit(X, y, batch_size=8, nb_epoch=10)

# evaluate model
# X, y = vsamp.generate_examples(10)
# loss, acc = model.evaluate(X, y, verbose=0)
# print('loss: %f, acc: %f' % (loss, acc*100))

# prediction on new data
# X, y = vsamp.generate_examples(1)
# yhat = model.predict_classes(X, verbose=0)
# expected = "Right" if y[0]==1 else "Left"
# predicted = "Right" if yhat[0]==1 else "Left"
# print('Expected: %s, Predicted: %s' % (expected, predicted))





# -------------------------------------------------------------
summaries_dir = '{0}/{1}'.format(ag.summaries_dir, datetime.datetime.now().strftime('%d_%b_%Y-%H_%M_%S'))
train_writer = tf.summary.FileWriter(ag.summaries_dir + '/train')
validation_writer = tf.summary.FileWriter(summaries_dir + '/validation')
#
model_name = 'sample_train' + str(int(time.time()))
model_dir = '{0}/{1}'.format(ag.log_dir, model_name)
# check wether dir exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
# ----------------------------------
# here to get lstm data

