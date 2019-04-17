#
# generating sample videos to test the model
#
# this wowrks for images, for videos I should create custom version
#
import keras
import numpy as np
from numpy import zeros
from random import randint
from random import random
from matplotlib import pyplot
import _pickle as pickle
from keras.models import Sequential
from sklearn.model_selection import train_test_split
#
from keras.layers import Dense, LSTM, Flatten, Dropout, TimeDistributed, GlobalAveragePooling2D, Input, BatchNormalization, Bidirectional
from keras.layers import Concatenate, Reshape, Conv2D, Activation, ZeroPadding2D, DepthwiseConv2D, GlobalMaxPooling1D, RepeatVector

#
# Reading the data
#
from sklearn.datasets import load_digits
digits = load_digits()
trX, tsX, trY, tsY = train_test_split(digits.data, digits.target)
# image feeding system

def data_generator(tX, tY):
    batch_size = 8
    dim = (tX.shape[0], tX.shape[1])
    n_channels = 64
    n_classes = 10

    shuffle = True

    iteration = dim[0] //batch_size

    # for indx in range(iteration):
    # while True:
    while True:
        X = np.empty((batch_size,1, dim[1]))
        y = np.empty(batch_size, dtype=int)
        if shuffle == True:
            np.random.shuffle(tX)
        for i in range(batch_size):
            rndx = np.random.randint(0, dim[0],1)
            X[i,] = tX[rndx].reshape(1,dim[1])
            y[i] = tY[rndx]
        yield X, keras.utils.to_categorical(y, num_classes=n_classes)

# xx,yy = data_generator(trX, trY)


