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
import cv2
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Sequential
from sklearn.model_selection import train_test_split
# load data generator
# from video_seq_generation import data_generator
from keras.layers import Dense, LSTM, Flatten, Dropout, TimeDistributed, GlobalAveragePooling2D, Input, BatchNormalization, Bidirectional
from keras.layers import Concatenate, Reshape, Conv2D, Activation, ZeroPadding2D, DepthwiseConv2D, GlobalMaxPooling1D, RepeatVector, MaxPooling2D
# ---------------
#from keras.utils.training_utils import multi_gpu_model
#from tensorflow.python.client import device_lib
from keras import backend as K

import tensorflow as tf
# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type    == 'GPU']
#num_gpu = len(get_available_gpus())
#print('number of gpus', num_gpu)



# arguments
data_dir = '/home/ali/BASKETBALL_DATA/Frames_sorted_by_shot/'
data_2  = 'data_02/video_ba_03_data-02.pickle'
label_2 = 'data_02/video_ba_03_label-02.pickle'
#
data_3  = 'data_03/video_ba_03_data-03.pickle'
label_3 = 'data_03/video_ba_03_label-03.pickle'

def data_loader(data_dir, data_name, label_name):
    with open(data_dir + data_name, 'rb') as f:
        gold_data = pickle.load(f)
    with open(data_dir + label_name, 'rb') as f:
        gold_label = pickle.load(f)
        # splitting data
    trX, tsX, trY, tsY = train_test_split(gold_data, gold_label, random_state=3, shuffle=True, train_size=.95)
    return trX, trY, tsX, tsY



# TODO: make labels as vectors

def data_gen(tX, tY, batch_size = 4):
    data_x = []
    data_y = []
    for j in range(batch_size):
        shot=[]
        for i in range(len(tX[j])): # trX.shape[1]
            img = tX[j][i].astype(np.float32)
            img = img.reshape(28,28,3)
            shot.append(img)
        shot = np.array(shot)
        # creating batch data and label
        data_x.append(shot)
        if tY[j] ==1:
            data_y.append(np.array([0, 1]))
        elif tY[j] == 0:
            data_y.append(np.array([1, 0]))
    return np.array(data_x), np.array(data_y)

# fit generator
def data_generator(tX, tY, batch_size = 1):
    data_x = []
    data_y = []
    while True:
        for k in range(len(tX)):
            for j in range(batch_size):
                shot=[]
                for i in range(len(tX[k][j])): # trX.shape[1]
                    img = tX[k][j][i].astype(np.float32)/255
                    img = img.reshape(28,28,3)
                    shot.append(img)
                shot = np.array(shot)
                # creating batch data and label
                data_x.append(shot)
                if tY[k][j] ==1:
                    data_y.append(np.array([0, 1]))
                elif tY[k][j] == 0:
                    data_y.append(np.array([1, 0]))
            yield np.array(data_x), np.array(data_y)
# # to test the model
tx0, ty0,sx0,sy0 = data_loader(data_dir, data_2, label_2)
tx1, ty1,sx1,sy1 = data_loader(data_dir, data_3, label_3)
a,b = data_generator([tx0, tx1], [ty0, ty1], batch_size = 1)

#
# A simple network to test the yield system with our data
#
size = 28
model = Sequential()
model.add(TimeDistributed(Conv2D(32, (2,2), activation='relu'),
                          input_shape=(None,size,size,3)))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(32, stateful=False))
model.add(Dense(2, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# print(model.summary())


print('saving training checkpoints as hdf5')
# NOTE: hd just saves the weights!



weigth_file="./weights/PreTrain_weights-improvement-cpu-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(weigth_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tbCallBack = TensorBoard(log_dir='./Graph/Pretrain_283cpu', histogram_freq=0, write_graph=True, write_images=True)
callbacks_list = [checkpoint,tbCallBack]



tx0, ty0,sx0,sy0 = data_loader(data_dir, data_2, label_2)
tx1, ty1,sx1,sy1 = data_loader(data_dir, data_3, label_3)

# fit generator
train_gen = data_generator([tx0, tx1], [ty0, ty1], batch_size = 1)
test_gen  = data_generator([sx0, sx1], [sy0, sy1], batch_size = 1)

nb_epochs = 50
# for j in range(nb_epochs):
model.fit_generator(generator=train_gen,
                    steps_per_epoch=40,
                    # validation_data=test_gen,
                    # validation_steps=10,
                    callbacks=callbacks_list,
                    epochs=50,
                    shuffle=True)
    # model.reset_states()
#     print('epoch: ', j)
# score = model.evaluate_generator(test_gen, nb_validation_samples/batch_size, workers=12)
# saving the model
model_json = model.to_json()
with open("./models/model_small_frms-05.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("./models/model_small_frms-05.h5")
print("Saved model to disk")
#

# memory error check
# Which value do you use for the half device?
# If it start with GPU, do you use the flag lib.cnmem?
# If not using it will speed up the computation and "fix" this problem.



# scores = model.predict_generator(test_gen, 50, workers=1)

