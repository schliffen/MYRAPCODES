#
# in this sandbox im trying to test sequential training system
#
import numpy as np
from numpy import zeros
from random import randint
from random import random
import matplotlib.pyplot as plt
import _pickle as pickle
import keras
import cv2
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import  ImageDataGenerator
#
#
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
# #num_gpu = len(get_available_gpus())
#print('number of gpus', num_gpu)
#
aug = ImageDataGenerator(rotation_range=30, width_shift_range=.1,
                         height_shift_range=.1, shear_range=.2, zoom_range=.2, horizontal_flip=True,
                         fill_mode='nearest')


# arguments
save_dir = '/home/ali/BASKETBALL_DATA/Frames_sorted_by_shot/'
data_dir = '/home/ali/BASKETBALL_DATA/Frames_sorted_by_shot/'
data  = 'Data_equalized_input_181009_.pickle'
label = 'Data_equalized_label_181009_.pickle'
#

with open(data_dir + data, 'rb') as f:
    gold_data = np.array(pickle.load(f))
with open(data_dir + label, 'rb') as f:
    gold_label = pickle.load(f)
    # splitting data


gold_data = list(gold_data.reshape((-1,40,gold_data.shape[1])))

nin = 0
nout = 0
for item in gold_label:
    if item == 1:
        nin +=1
    elif item == 0:
        nout +=1
#
# print('in: %2f%%, out:%2f%%' %(nin, nout))
#
# trX, tsX, trY, tsY = train_test_split(gold_data, gold_label, random_state=3, shuffle=True, train_size=.96)
#
def data_gen():
    aug.fit(np.array(gold_data).reshape((-1,28,28,3)))
    for i in range(len(gold_data)):
        batch = gold_data[i]
        batchx = batch.reshape(40,28,28,3)
        batchy = [gold_label[i]]*40

        for augX, _ in aug.flow(batchx, batchy, batch_size=40):
            augX = augX.reshape(1,40,28*28*3)
            yield augX, [gold_label[i]]

def data_generator(tX, tY, batch_size = 1):
    data_x = []
    data_y = []
    while True:
        # for k in range(len(tX)):
        for j in range(batch_size):
            shot=[]
            for i in range(tX[j].shape[0]): # trX.shape[1]
                img = tX[j][i].astype(np.float32)/255
                img = img.reshape(28*28*3)
                shot.append(img)
            shot = np.array(shot)
            # creating batch data and label
            data_x.append(shot)
            if tY[j] == 1:
                data_y.append(np.array([0, 1]))
            elif tY[j] == 0:
                data_y.append(np.array([1, 0]))
            # data_y.append(tY[j])
        yield np.array(data_x), np.array(data_y)



# for a,b in data_generator(gold_data, gold_label, batch_size = 1):
#     print(b)
#
# TODO: make labels as vectors
# checking the data equivallency


# tx0, ty0,sx0,sy0 = data_loader(data_dir, data_2, label_2)
# tx1, ty1,sx1,sy1 = data_loader(data_dir, data_3, label_3)
# a,b = data_generator([tx0, tx1], [ty0, ty1], batch_size = 1)

#
# A simple network to test the yield system with our data
#
size = 28
model = Sequential()
#model.add(TimeDistributed(Conv2D(32, (2,2), activation='relu'),
#                          input_shape=(None,size,size,3)))
#model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
#model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape =(40,size*size*3) ))
model.add(Dropout(.5))
model.add(LSTM(50, stateful=False))
model.add(Dropout(.5))
# model.add(Dense(100, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dropout(.2))
# model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))
# model.add(Dense(1,activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# print(model.summary())


print('saving training checkpoints as hdf5')
# NOTE: hd just saves the weights!



weigth_file="./weights/PreTrain_weights-improvement-cpu-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(weigth_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tbCallBack = TensorBoard(log_dir='./Graph/Pretrain_283cpu', histogram_freq=0, write_graph=True, write_images=True)
callbacks_list = [checkpoint,tbCallBack]



# tx0, ty0,sx0,sy0 = data_loader(data_dir, data, label)
# tx1, ty1,sx1,sy1 = data_loader(data_dir, data_3, label_3)

# fit generator
train_gen = data_generator(gold_data, gold_label, batch_size = 1)
# test_gen  = data_generator([sx0, sx1], [sy0, sy1], batch_size = 1)

nb_epochs = 50
# for j in range(nb_epochs):
model.fit_generator(generator=train_gen,
                    steps_per_epoch=len(gold_data),
                    # validation_data=test_gen,
                    # validation_steps=10,
                    callbacks=callbacks_list,
                    epochs=30,
                    shuffle=True,
                    verbose=1)
# model.reset_states()
#     print('epoch: ', j)
# score = model.evaluate_generator(test_gen, nb_validation_samples/batch_size, workers=12)
# saving the model
model_json = model.to_json()
with open("./models/model_small_frms-tst.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("./models/model_small_frms-tst.h5")
print("Saved model to disk")
#

# memory error check
# Which value do you use for the half device?
# If it start with GPU, do you use the flag lib.cnmem?
# If not using it will speed up the computation and "fix" this problem.



# scores = model.predict_generator(test_gen, 50, workers=1)

