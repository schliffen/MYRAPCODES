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
# generate the next frame in the sequence
#
# class vid_sample():
#     def __init__(self, size):
#         self.size = size
#
#     def next_frame(self, last_step, last_frame, column):
#         # define the scope of the next step
#         lower = max(0, last_step-1)
#         upper = min(last_frame.shape[0]-1, last_step+1)
#         # choose the row index for the next step
#         step = randint(lower, upper)
#         # copy the prior frame
#         frame = last_frame.copy()
#         # add the new step
#         frame[step, column] = 1
#         return frame, step
#     #
#     # generate a sequence of frames of a dot moving across an image
#     #
#     def build_frames(self):
#         frames = list()
#         # create the first frame
#         frame = zeros((self.size, self.size))
#         step = randint(0, self.size-1)
#         # decide if we are heading left or right
#         right = 1 if random() < 0.5 else 0
#         col = 0 if right else self.size-1
#         frame[step, col] = 1
#         frames.append(frame)
#         # create all remaining frames
#         for i in range(1, self.size):
#             col = i if right else self.size-1-i
#             frame, step = self.next_frame(step, frame, col)
#             frames.append(frame)
#         return frames, right
#
#     # generate multiple sequences of frames and reshape for network input
#     def generate_examples(self, n_patterns):
#         X, y = list(), list()
#         for _ in range(n_patterns):
#             frames, right = self.build_frames()
#             X.append(frames)
#             y.append(right)
#         # resize as [samples, timesteps, width, height, channels]
#         X = np.array(X).reshape(n_patterns, self.size, self.size, self.size, 1)
#         y = np.array(y).reshape(n_patterns, 1)
#         return X, y
# # generate sequence of frames
# size = 5
# frames, right = build_frames(size)
# # plot all feames
# pyplot.figure()
# for i in range(size):
#     # create a grayscale subplot for each frame
#     pyplot.subplot(1, size, i+1)
#     pyplot.imshow(frames[i], cmap='Greys')
#     # turn of the scale to make it cleaer
#     ax = pyplot.gca()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# # show the plot
# pyplot.show()
#
# creating a simple model here
#

class Data_generator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=8, dim=(32,32,32), n_channels=3,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch o__data_generationf data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')
            print(ID)
            # print(list_IDs_temp[ID])
            X[i,] = list_IDs_temp[int(ID[0])]

            # Store class
            y[i] = self.labels[int(ID[0])]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
#
#


