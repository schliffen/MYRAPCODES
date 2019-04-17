#
#
#import keras
import numpy as np
from random import random, randint
from numpy import array, zeros
from matplotlib import pyplot
import _pickle as pickle
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from sklearn.model_selection import train_test_split
# load data generator
from video_seq_generation import data_generator
#
from keras.layers import Dense, LSTM, Flatten, Dropout, TimeDistributed, GlobalAveragePooling2D, Input, BatchNormalization, Bidirectional
from keras.layers import Concatenate, Reshape, Conv2D, Activation, ZeroPadding2D, DepthwiseConv2D, GlobalMaxPooling1D, RepeatVector, MaxPooling2D
#




#
# creating sample video
# generate the next frame in the sequence
def next_frame(last_step, last_frame, column):
    # define the scope of the next step
    lower = max(0, last_step-1)
    upper = min(last_frame.shape[0]-1, last_step+1)
    # choose the row index for the next step
    step = randint(lower, upper)
    # copy the prior frame
    frame = last_frame.copy()
    # add the new step
    frame[step, column] = 1
    return frame, step
# generate a sequence of frames of a dot moving across an image
def build_frames(size):
    frames = list()
    # create the first frame
    frame = zeros((size,size))
    step = randint(0, size-1)
    # decide if we are heading left or right
    right = 1 if random() < 0.5 else 0
    col = 0 if right else size-1
    frame[step, col] = 1
    frames.append(frame)
    # create all remaining frames
    for i in range(1, size):
        col = i if right else size-1-i
        frame, step = next_frame(step, frame, col)
        frames.append(frame)
    return frames, right
# generate multiple sequences of frames and reshape for network input
def generate_examples(size, n_patterns):
    X, y = list(), list()
    for _ in range(n_patterns):
        frames, right = build_frames(size)
        X.append(frames)
        y.append(right)
        # resize as [samples, timesteps, width, height, channels]
    X = array(X).reshape(n_patterns, size, size, size, 1)
    y = array(y).reshape(n_patterns, 1)
    return X, y
#
def data_generator(tX, tY):

    batch_size = 8
    n_classes = 2
    dim = tX.shape
    shuffle = True

    iteration = dim[0] // batch_size

    # for indx in range(iteration):
    # while True:
    while True:
        X = np.empty((batch_size, dim[1], dim[2],dim[3], 1))
        y = np.empty(batch_size, dtype=int)
        for i in range(batch_size):
            X[i,] = tX[i]
            y[i] = tY[i]
        yield np.array(X) , np.array(keras.utils.to_categorical(y, num_classes=n_classes))
        # X = np.empty((batch_size, dim[1]))
        # y = np.empty(batch_size, dtype=int)
        # if shuffle == True:
        #     np.random.shuffle(tX)
        # for i in range(batch_size):
        #     rndx = np.random.randint(0, dim[0],1)
        #     X[i,] = tX[rndx].reshape(1,dim[1])
        #     y[i] = tY[rndx]
        # yield X, keras.utils.to_categorical(y, num_classes=n_classes)
#
#

# configure problem
size = 200
# define the model
def creat_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(2, (2,2), activation='relu'),
                              input_shape=(None,size,size,1)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['acc'])
    return model

model = creat_model()
# using efficient wrapper -> this estimator is used for cross validation
estimator = KerasClassifier(build_fn=creat_model, epochs = 10, batch_size = 2, verbose = 0 )



# fit model
X, y = generate_examples(size, 500)
train_gen = data_generator(X, y)
# creating feed generator
# a,b = data_generator(X, y)
X, y = generate_examples(size, 100)
eval_gen = data_generator(X, y)

history = model.fit_generator(train_gen,
                              validation_data=eval_gen,
                              validation_steps=20,
                              nb_epoch=80, samples_per_epoch=10)
#
# evaluate model
#


# loss, acc = model.evaluate(X, y, verbose=0)
# print('loss: %f, acc: %f' % (loss, acc*100))
# prediction on new data
X, y = generate_examples(size, 1)
yhat = model.predict_classes(X, verbose=0)
expected = "Right" if y[0]==1 else "Left"
predicted = "Right" if yhat[0]==1 else "Left"
print('Expected: %s, Predicted: %s' % (expected, predicted))





